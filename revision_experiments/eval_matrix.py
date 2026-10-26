# -*- coding: utf-8 -*-
"""Evaluate all existing checkpoints on the CLEAN held-out test set
(FDK_raw_200Views, 31 slices, disjoint from training per F9) under multiple
flux levels and multiple ROIs, to map the current state of play.

Outputs: eval_matrix/summary.csv  (model x testset x ROI -> SSIM/PSNR/RMSE mean+-SD)
         eval_matrix/per_slice/<model>__<testset>__<roi>.csv
"""
import os
import sys
import csv
import glob

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from math import sqrt
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '')  # TODO: set local path
sys.path.insert(0, '')  # TODO: set local path
from common import imread01, wavelet_branch, load_torch
from nets import (UDAN, UDANRes, UDANResNoWT, UDANNoWT, DnCNNMMD, REDCNNMMD,
                   FBPConvNetMMD)
from net.DnCNN import DnCNN
from net.RED_CNN.networks import RED_CNN
from net.FBPConvNet.model import FBPCONVNet

RUNS = ''  # TODO: set local path
OUTD = (''  # TODO: set local path
        if os.environ.get('TTA') == '1' else '')  # TODO: set local path
FDK = ''  # TODO: set local path

TESTSETS = {
    '1e4': (os.path.join(FDK, 'FDK_raw_200Views_noise1e4'), os.path.join(FDK, 'FDK_raw_200Views')),
    '1e3': (os.path.join(FDK, 'FDK_raw_200Views_noise1e3'), os.path.join(FDK, 'FDK_raw_200Views')),
    '3e2': (os.path.join(FDK, 'FDK_raw_200Views_noise3e2'), os.path.join(FDK, 'FDK_raw_200Views')),
    '3e3': (os.path.join(FDK, 'FDK_raw_200Views_noise3e3'), os.path.join(FDK, 'FDK_raw_200Views')),
    '2e3': (os.path.join(FDK, 'FDK_raw_200Views_noise2e3'), os.path.join(FDK, 'FDK_raw_200Views')),
    '9e2': (os.path.join(FDK, 'FDK_raw_200Views_noise9e2'), os.path.join(FDK, 'FDK_raw_200Views')),
    '8e2': (os.path.join(FDK, 'FDK_raw_200Views_noise8e2'), os.path.join(FDK, 'FDK_raw_200Views')),
    '7e2': (os.path.join(FDK, 'FDK_raw_200Views_noise7e2'), os.path.join(FDK, 'FDK_raw_200Views')),
    '6e2': (os.path.join(FDK, 'FDK_raw_200Views_noise6e2'), os.path.join(FDK, 'FDK_raw_200Views')),
    '5e2': (os.path.join(FDK, 'FDK_raw_200Views_noise5e2'), os.path.join(FDK, 'FDK_raw_200Views')),
}

# (display, run, kind)  kind: plain | uda(UDANRes-family via nets.py) | uda_orig(model3.UNet)
#                                            | plain_none (input baseline) | folder (pre-rendered outputs)
MODELS = [
    ('input', None, 'plain_none'),
    ('SART', '__FOLDER__SART1e3', 'folder'),
    ('DnCNN-Xray(direct)', 'DnCNN-Xray', 'plain'),
    ('REDCNN-Xray(direct)', 'REDCNN-Xray', 'plain'),
    ('FBPConvNet-Xray(direct)', 'FBPConvNet-Xray', 'plain'),
    ('DnCNN-XraySV(direct)', 'DnCNN-XraySV', 'plain'),
    ('REDCNN-XraySV(direct)', 'REDCNN-XraySV', 'plain'),
    ('FBPConvNet-XraySV(direct)', 'FBPConvNet-XraySV', 'plain'),
    ('DnCNN-MMD@1e4', 'DnCNN-MMD', 'uda'),
    ('REDCNN-MMD@1e4', 'REDCNN-MMD', 'uda'),
    ('UDAN-res@1e4', 'UDAN-residual', 'uda'),
    ('UDAN-noMMD(direct-arch)', 'UDANRes-noMMD', 'uda'),
    ('DnCNN-sup@1e4', 'DnCNN-NCT', 'plain'),
    ('REDCNN-sup@1e4', 'REDCNN-NCT', 'plain'),
    ('FBPConvNet-sup@1e4', 'FBPConvNet-NCT', 'plain'),
    ('UDAN-sup@1e4', 'UDAN-supNCT', 'uda_orig'),
    ('DnCNN-sup@1e3', 'DnCNN-NCT1e3', 'plain'),
    ('REDCNN-sup@1e3', 'REDCNN-NCT1e3', 'plain'),
    ('FBPConvNet-sup@1e3', 'FBPConvNet-NCT1e3', 'plain'),
    ('UDAN-sup@1e3', 'UDAN-supNCT1e3', 'uda_orig'),
    ('UDAN-res@1e3', 'UDAN-residual-1e3', 'uda'),
    ('UDAN-U1c@1e3', 'U1c', 'uda'),
    ('UDAN-U1b2@1e3', 'U1b2', 'uda'),
    ('UDAN-U1eEMA@1e3', 'U1e#ema', 'uda'),
    ('UDAN-U1eS1@1e3', 'U1e_s1#ema', 'uda'),
    ('UDAN-U1eL2@1e3', 'U1e_l2#ema', 'uda'),
    ('UDAN-U1eD95@1e3', 'U1e_d95#ema', 'uda'),
    ('UDAN-U1s100@1e3', 'U1s100', 'uda'),
    ('UDAN-U1l05@1e3', 'U1l05', 'uda'),
    ('UDAN-U1loc3@1e3', 'U1loc3', 'uda'),
    ('UDAN-U1eRAW@1e3', 'U1e', 'uda'),
    ('DnCNN-MMD@1e3', 'DnCNN-MMD1e3', 'uda'),
    ('REDCNN-MMD@1e3', 'REDCNN-MMD1e3', 'uda'),
    ('UDANRes-noWT@1e3', 'UDANRes-noWT1e3', 'uda'),
    ('UDAN-1e3', 'UDAN-1e3', 'uda'),
    ('FBPConvNet-MMD@1e3', 'FBPConvNet-MMD1e3', 'uda'),
    ('UDAN-supNCT3e2', 'UDAN-supNCT3e2', 'uda_orig'),
    ('DnCNN-sup@3e2', 'DnCNN-NCT3e2', 'plain'),
    ('REDCNN-sup@3e2', 'REDCNN-NCT3e2', 'plain'),
    ('FBPConvNet-sup@3e2', 'FBPConvNet-NCT3e2', 'plain'),
    ('DnCNN-MMD@3e2', 'DnCNN-MMD3e2', 'uda'),
    ('UDAN-U2eEMA@3e2', 'U2e_d95#ema', 'uda'),
    ('UDAN-supNCT3e3', 'UDAN-supNCT3e3', 'uda_orig'),
    ('DnCNN-sup@3e3', 'DnCNN-NCT3e3', 'plain'),
    ('REDCNN-sup@3e3', 'REDCNN-NCT3e3', 'plain'),
    ('FBPConvNet-sup@3e3', 'FBPConvNet-NCT3e3', 'plain'),
    ('DnCNN-MMD@3e3', 'DnCNN-MMD3e3', 'uda'),
    ('REDCNN-MMD@3e3', 'REDCNN-MMD3e3', 'uda'),
    ('FBPConvNet-MMD@3e3', 'FBPConvNet-MMD3e3', 'uda'),
    ('UDAN-U3eEMA@3e3', 'U3e_d95#ema', 'uda'),
    ('UDAN-supNCT8e2', 'UDAN-supNCT8e2', 'uda_orig'),
    ('DnCNN-sup@8e2', 'DnCNN-NCT8e2', 'plain'),
    ('REDCNN-sup@8e2', 'REDCNN-NCT8e2', 'plain'),
    ('FBPConvNet-sup@8e2', 'FBPConvNet-NCT8e2', 'plain'),
    ('DnCNN-MMD@8e2', 'DnCNN-MMD8e2', 'uda'),
    ('REDCNN-MMD@8e2', 'REDCNN-MMD8e2', 'uda'),
    ('FBPConvNet-MMD@8e2', 'FBPConvNet-MMD8e2', 'uda'),
    ('UDAN-U8e2EMA@8e2', 'U8e2_d95#ema', 'uda'),
    ('UDAN-supNCT7e2', 'UDAN-supNCT7e2', 'uda_orig'),
    ('DnCNN-sup@7e2', 'DnCNN-NCT7e2', 'plain'),
    ('REDCNN-sup@7e2', 'REDCNN-NCT7e2', 'plain'),
    ('FBPConvNet-sup@7e2', 'FBPConvNet-NCT7e2', 'plain'),
    ('DnCNN-MMD@7e2', 'DnCNN-MMD7e2', 'uda'),
    ('REDCNN-MMD@7e2', 'REDCNN-MMD7e2', 'uda'),
    ('FBPConvNet-MMD@7e2', 'FBPConvNet-MMD7e2', 'uda'),
    ('UDAN-U7e2EMA@7e2', 'U7e2_d95#ema', 'uda'),
]

_YY, _XX = np.mgrid[0:512, 0:512]
CIRC = (_YY - 256) ** 2 + (_XX - 256) ** 2 <= 240 ** 2


def object_mask(ref):
    """Binary object mask from the normal-flux reference (Otsu, largest CC, closed)."""
    v = cv2_threshold(ref)
    m = ndimage.binary_closing(v, iterations=3)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (np.argmax(sizes) + 1)
    return ndimage.binary_fill_holes(m)


def cv2_threshold(img):
    """Otsu via skimage-free implementation (histogram between-class variance)."""
    hist, _ = np.histogram(img, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    tot = hist.sum()
    w0 = np.cumsum(hist)
    w1 = tot - w0
    mu = np.cumsum(hist * np.arange(256))
    mt = mu[-1]
    valid = (w0 > 0) & (w1 > 0)
    sb = np.zeros(256)
    sb[valid] = (mt * w0[valid] - mu[valid]) ** 2 / (w0[valid] * w1[valid])
    thr = np.argmax(sb) / 255.0
    return img > thr


def main():
    os.makedirs(os.path.join(OUTD, 'per_slice'), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ts_only = set(filter(None, os.environ.get('TS_ONLY', '').split(',')))
    model_only = list(filter(None, os.environ.get('MODEL_ONLY', '').split(',')))
    rows_all = []
    for ts_name, (inp_root, ref_root) in TESTSETS.items():
        if ts_only and ts_name not in ts_only:
            continue
        refs = {os.path.basename(p): p for p in glob.glob(os.path.join(ref_root, '*.png'))}
        pairs = []
        for p in sorted(glob.glob(os.path.join(inp_root, '*.png'))):
            name = os.path.basename(p)
            if name in refs:
                pairs.append((p, refs[name], name))
        print(f'[{ts_name}] {len(pairs)} pairs')

        for disp, run, kind in MODELS:
            if model_only and not any(m in disp for m in model_only):
                continue
            net = None
            folder_dir = None
            if run is not None and run.startswith('__FOLDER__'):
                folder_dir = os.path.join(RUNS, run[len('__FOLDER__'):], 'outputs')
            elif run is not None:
                if '#' in run:
                    run_dir, suf = run.split('#')
                    ck_name = {'ema': 'ema_best.pth'}.get(suf, suf)
                else:
                    run_dir, ck_name = run, 'best.pth'
                ck_path = os.path.join(RUNS, run_dir, ck_name)
                if not os.path.exists(ck_path):
                    print(f'[{ts_name}] SKIP {disp} (no checkpoint {run})')
                    continue
                ck = load_torch(ck_path, map_location='cpu')
                if kind == 'plain':
                    if run.startswith('DnCNN'):
                        net = DnCNN(channels=1, features=64, num_of_layers=17)
                    elif run.startswith('REDCNN'):
                        net = RED_CNN(out_ch=96)
                    else:
                        net = FBPCONVNet()
                elif kind == 'uda':
                    if run.startswith('DnCNN-MMD'):
                        net = DnCNNMMD()
                    elif run.startswith('REDCNN-MMD'):
                        net = REDCNNMMD()
                    elif run.startswith('FBPConvNet-MMD'):
                        from nets import FBPConvNetMMD
                        net = FBPConvNetMMD()
                    elif run.startswith('UDANRes-noWT'):
                        net = UDANResNoWT()
                    elif run == 'UDAN-1e3':  # plain UDAN (no residual anchoring)
                        net = UDAN()
                    else:  # UDAN-residual*, U1c, U1b2, U1e#ema, UDANRes-noMMD
                        net = UDANRes()
                elif kind == 'uda_orig':
                    import model3
                    net = model3.UNet(n_channels=1, n_classes=1)
                net.load_state_dict(ck['state_dict'])
                net = net.to(device).eval()

            acc = {}
            for ip, rp, name in pairs:
                low = imread01(ip)
                ref = imread01(rp)
                if folder_dir is not None:
                    fp = os.path.join(folder_dir, name)
                    out = imread01(fp) if os.path.exists(fp) else None
                    if out is None:
                        continue
                elif net is None:
                    out = low
                else:
                    x = torch.from_numpy(low).float()[None, None].to(device)
                    with torch.no_grad():
                        def infer(arr, tens):
                            if kind in ('uda', 'uda_orig'):
                                LL2, LL3 = wavelet_branch(arr)
                                l2 = torch.from_numpy(LL2).float()[None, None].to(device)
                                l3 = torch.from_numpy(LL3).float()[None, None].to(device)
                                r = net(tens, tens, l2, l3)
                                return (r[0] if isinstance(r, tuple) else r).detach()
                            return net(tens).detach()
                        o = infer(low, x)
                        if os.environ.get('TTA') == '1':
                            xf = torch.flip(x, dims=[3])
                            lowf = np.ascontiguousarray(low[:, ::-1])
                            of = infer(lowf, xf)
                            o = 0.5 * (o + torch.flip(of, dims=[3]))
                    out = o.squeeze().cpu().numpy()
                out = np.clip(out, 0, 1)
                rr = {
                    'full': np.ones_like(ref, bool),
                    'c448': None,  # handled by slicing
                    'r240': CIRC,
                }
                for roi_name, mask in rr.items():
                    if roi_name == 'full':
                        s = float(ssim(ref, out, data_range=1))
                        a, b = ref.ravel(), out.ravel()
                    elif roi_name == 'c448':
                        rc, oc = ref[32:480, 32:480], out[32:480, 32:480]
                        s = float(ssim(rc, oc, data_range=1))
                        a, b = rc.ravel(), oc.ravel()
                    else:
                        # mask-zeroed full-frame SSIM (consistent with eval_curated protocol)
                        s = float(ssim(ref * mask, out * mask, data_range=1))
                        a, b = ref[mask], out[mask]
                    p = float(psnr(a, b, data_range=1))
                    r_ = float(sqrt(np.mean((a - b) ** 2)))
                    acc.setdefault(roi_name, []).append((name, s, p, r_))

            for roi_name, vals in acc.items():
                ss = [v[1] for v in vals]
                ps = [v[2] for v in vals]
                rs = [v[3] for v in vals]
                rows_all.append([ts_name, disp, roi_name, len(vals),
                                 np.mean(ss), np.std(ss), np.mean(ps), np.std(ps),
                                 np.mean(rs), np.std(rs)])
                print('%-5s %-26s %-5s n=%d SSIM %.4f±%.3f PSNR %.2f±%.2f RMSE %.4f' %
                      (ts_name, disp, roi_name, len(vals), np.mean(ss), np.std(ss),
                       np.mean(ps), np.std(ps), np.mean(rs)))
                with open(os.path.join(OUTD, 'per_slice', f'{disp}__{ts_name}__{roi_name}.csv'),
                          'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['slice', 'SSIM', 'PSNR_dB', 'RMSE'])
                    w.writerows(vals)

    tag = '_' + os.environ['OUT_TAG'] if os.environ.get('OUT_TAG') else ''
    out_csv = os.path.join(OUTD, f'summary{tag}.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['testset', 'model', 'roi', 'n', 'SSIM_mean', 'SSIM_sd',
                    'PSNR_mean', 'PSNR_sd', 'RMSE_mean', 'RMSE_sd'])
        w.writerows(rows_all)
    print('saved', out_csv)


if __name__ == '__main__':
    main()
