# -*- coding: utf-8 -*-
"""Exact Python replication of TIGRE's FDK pipeline (raw_noise_FDK.m recipe):

  p = -log(proj/(max+1))                       [v, u, angle] from 200 tifs
  Parker weight (q=1, eps=0) over (u, beta)    [tapers first/last angles]
  ramp filter along u: Kak-Slaney kernel |fft|^2 construction, filt_len=1024,
  centered placement, scale = angle_step/dU * DSD/DSO
  parallel backprojection: img(x,y) = sum_a p_f[v(z), u(x,y;a), a]
  (cosine DSD weight omitted: max deviation 0.09% at this geometry)

--verify : reproduce clean test recon and I0=1e3 stats vs existing TIGRE data
--gen    : generate the new flux level (train z=122 + test z=32 folders)
"""
import os
import sys
import glob

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
from PIL import Image
import torch

RAW = r'H:/6248R/data/gluedbox/raw/proj'
FDK = r'H:/6248R/CodeMATLAB/result/FDK'
NANG = 200
ANGLES = np.linspace(0.0, np.pi, NANG)          # MATLAB linspace(0,1*pi,200)
DSD, DSO = 600.3, 600.2
DU = 0.1
NDET = 512


def load_projs():
    files = [os.path.join(RAW, '%03d.tif' % i) for i in range(1, NANG + 1)]
    return np.stack([np.asarray(Image.open(f), dtype=np.float64) for f in files],
                    axis=-1)                    # [v, u, a]


def clean_proj(proj):
    P = proj.max() + 1.0
    return -np.log(np.maximum(proj, 1.0) / P)


def noise_proj(proj, I0, seed=2024):
    """addCTnoise(proj,'Poisson',I0) then Beer-Lambert normalize."""
    rng = np.random.RandomState(seed)
    P = proj.max()
    Im = I0 * np.exp(-proj / P)
    Im = rng.poisson(Im).astype(np.float64) + rng.normal(0.0, 0.5, Im.shape)
    Im[Im <= 0] = 1e-6
    pn = np.log(I0 / Im) * P
    return -np.log(np.maximum(pn, 1.0) / (pn.max() + 1.0))


def parker_weight():
    """TIGRE ParkerWeight with q=1, epsilon=0 for our parallel 0..pi scan."""
    us = np.arange(-NDET / 2 + 0.5, NDET / 2) * DU
    alpha = np.arctan(us / DSD)
    alpha = -alpha                      # angles incremental -> flip
    delta = abs(alpha[-1] - alpha[0]) / 2

    def S(x):
        x = np.asarray(x, dtype=np.float64)
        y = np.zeros_like(x)
        m = np.abs(x) < 0.5
        y[m] = 0.5 * (1 + np.sin(np.pi * x[m]))
        y[x >= 0.5] = 1.0
        return y

    B = lambda a: 2 * delta - 2 * a + 0.0
    w = np.zeros((NANG, NDET))
    for ii, ang in enumerate(ANGLES):
        beta = abs(ang - ANGLES[0])
        w[ii] = 0.5 * (S(beta / B(alpha) - 0.5)
                       + S((beta - 2 * delta + 2 * alpha) / B(alpha) + 0.5)
                       - S((beta - np.pi + 2 * alpha) / B(-alpha) - 0.5)
                       - S((beta - np.pi - 2 * delta) / B(-alpha) + 0.5))
    return w.T                            # [u, angle]


def ramp_filter(p):
    """TIGRE filtering: ramp along u axis, filt_len=1024, centered placement."""
    filt_len = 1024
    nn = np.arange(-(filt_len // 2), filt_len // 2)
    h = np.zeros(filt_len, dtype=np.float64)
    h[filt_len // 2] = 0.25
    odd = (nn % 2) == 1
    h[odd] = -1.0 / (np.pi * nn[odd]) ** 2
    fk = np.abs(np.fft.fft(h)) * 2
    filt = fk[:filt_len // 2 + 1].copy()
    w_axis = 2 * np.pi * np.arange(filt_len // 2 + 1) / filt_len
    filt[w_axis > np.pi] = 0
    filt = np.concatenate([filt, filt[-2:0:-1]])

    astep = np.pi / (NANG - 1)
    scale = astep / DU * (DSD / DSO)

    lo = filt_len // 2 - NDET // 2        # 256
    f = np.zeros((p.shape[0], filt_len, p.shape[2]), dtype=np.float64)
    f[:, lo:lo + NDET, :] = p
    F = np.fft.fft(f, axis=1) * filt[None, :, None]
    out = np.real(np.fft.ifft(F, axis=1))[:, lo:lo + NDET, :] * scale
    return out


def backproject(p_filt, z_mm, device='cuda'):
    """Parallel backprojection of one slice; p_filt [v, u, angle].
    Verified mapping: u_pix = y_pix*cos(a) - x_pix*sin(a), centers at 256."""
    row = (z_mm / 51.2 + 0.5) * (NDET - 1)
    r0 = int(np.floor(row))
    f = row - r0
    if r0 >= NDET - 1:
        r0, f = NDET - 2, 1.0
    sino = ((1 - f) * torch.from_numpy(p_filt[r0]).float()
            + f * torch.from_numpy(p_filt[r0 + 1]).float()).to(device)   # [u, a]
    pix = torch.arange(NDET, device=device) - 256.0
    XPI, YPI = torch.meshgrid(pix, pix, indexing='ij')                   # rows=x, cols=y
    sino4 = sino[None, None, :, :]                                        # [1,1,u,a]
    acc = torch.zeros(NDET, NDET, device=device)
    for a_i, ang in enumerate(ANGLES):
        u = YPI * np.cos(ang) - XPI * np.sin(ang)
        gy = (u + 256.0) / (NDET - 1) * 2 - 1
        gx = torch.full_like(gy, a_i / (NANG - 1) * 2 - 1)
        g = torch.stack([gx, gy], dim=-1)[None]                           # [1,S,S,2]
        acc += torch.nn.functional.grid_sample(
            sino4, g, align_corners=True, mode='bilinear',
            padding_mode='zeros')[0, 0]
    return acc.cpu().numpy()


def recon_slice(p, z_mm):
    pf = ramp_filter(p * parker_weight()[None, :, :])
    return backproject(pf, z_mm)


def save_png(img, path):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def verify():
    proj = load_projs()
    ref_dir = os.path.join(FDK, 'FDK_raw_200Views')
    best = None
    for sign in (+1, -1):
        for idx in [15]:
            ref = np.asarray(Image.open(os.path.join(ref_dir, '%d.png' % idx)),
                             dtype=np.float64) / 255.0
            z = -16.0 + 0.5 + (idx - 1)
            rec = recon_slice(clean_proj(proj), z, sign)
            c = np.corrcoef(rec.ravel(), ref.ravel())[0, 1]
            s = (rec * ref).sum() / (rec * rec).sum()
            print('sign %+d slice %d: corr %.4f  scale-fit %.4f' % (sign, idx, c, s))
            if best is None or c > best[0]:
                best = (c, sign, s)
    print('best:', best)
    return best


def gen(I0=3e2, seed=2024, s_out=None, out_tag='3e2'):
    """Generate the harsher-flux dataset (train 122 + test 31 slices)."""
    proj = load_projs()
    # global scale: fit against clean TIGRE references on a few test slices
    if s_out is None:
        pf_c = ramp_filter(clean_proj(proj) * parker_weight()[None, :, :])
        ss = []
        for idx in [5, 15, 25]:
            ref = np.asarray(Image.open(os.path.join(
                FDK, 'FDK_raw_200Views', '%d.png' % idx)), dtype=np.float64) / 255.0
            z = -16.0 + 0.5 + (idx - 1)
            rec = backproject(pf_c, z)
            ss.append((rec * ref).sum() / (rec * rec).sum())
        s_out = float(np.median(ss))
        print('global scale =', s_out)
    pn = noise_proj(proj, I0, seed)
    pf = ramp_filter(pn * parker_weight()[None, :, :])

    out_test = os.path.join(FDK, 'FDK_raw_200Views_noise%s' % out_tag)
    out_train = os.path.join(FDK, 'FDK_raw3_noise_%s' % out_tag)
    os.makedirs(out_test, exist_ok=True)
    os.makedirs(out_train, exist_ok=True)
    dz_train = 51.0 / 128.0
    for i in range(1, 32):                       # test slices 1..31
        z = -16.0 + 0.5 + (i - 1)
        rec = backproject(pf, z) * s_out
        save_png(rec, os.path.join(out_test, '%d.png' % i))
        print('test slice', i, flush=True)
    for i in range(1, 123):                      # train slices 1..122
        z = -25.5 + (i - 0.5) * dz_train
        rec = backproject(pf, z) * s_out
        save_png(rec, os.path.join(out_train, '%d.png' % i))
        if i % 10 == 0:
            print('train slice', i, flush=True)
    print('done ->', out_test, out_train)


if __name__ == '__main__':
    if '--verify' in sys.argv:
        verify()
    if '--gen' in sys.argv:
        def _arg(flag, default):
            return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
        gen(I0=float(_arg('--i0', '3e2')), out_tag=_arg('--tag', '3e2'))
