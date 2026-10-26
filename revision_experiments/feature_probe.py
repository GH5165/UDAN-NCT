# -*- coding: utf-8 -*-
"""Per-layer feature probe for the domain-gap study.

Quantifies the X-ray(source) <-> NCT(target) distribution gap at every encoder
stage of the UDAN architecture, before adaptation (X-ray-only trained, no MMD)
and after label-free MMD adaptation at each acquisition condition (flux).

Per (model, layer, condition): normalized mean-embedding distance (scale-free
linear MMD^2), raw linear MMD^2, and a linear domain-probe accuracy on a
4-component PCA subspace (5-fold CV; 0.5 = domains indistinguishable).
Also reports raw-input low-level stats per condition (normalized distance on
64x64 downsamples, Laplacian noise estimate, intensity mean/std).

Usage:
  python feature_probe.py --device cpu            # all available models
  python feature_probe.py --models noMMD,U1e_d95  # subset
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import imread01, XRAY_LOW, wavelet_branch, load_torch
from nets import UDANRes

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')

CONDITIONS = {
    '3e3': '',  # TODO: set local path
    '1e3': '',  # TODO: set local path
    '8e2': '',  # TODO: set local path
    '7e2': '',  # TODO: set local path
    '3e2': '',  # TODO: set local path
}
MODELS = {
    'noMMD': ('UDANRes-noMMD', 'best.pth'),        # X-ray only, before adaptation
    'UDAN@1e3': ('U1e_d95', 'ema_best.pth'),        # label-free adapted at 1e3
    'UDAN@3e3': ('U3e_d95', 'ema_best.pth'),        # label-free adapted at 3e3
    'UDAN@8e2': ('U8e2_d95', 'ema_best.pth'),       # label-free adapted at 8e2
    'UDAN@7e2': ('U7e2_d95', 'ema_best.pth'),       # label-free adapted at 7e2
    'UDAN@3e2': ('U2e_d95', 'ema_best.pth'),        # label-free adapted at 3e2
}
LAYERS = ['inc', 'down1', 'down2', 'down3', 'down4']
N_PER = 30
SEED = 0


def norm_dist(x, y):
    """||mu_s - mu_t||^2 / (0.5*(||mu_s||^2 + ||mu_t||^2)) — scale-free gap."""
    mu_s, mu_t = x.mean(0), y.mean(0)
    return float(((mu_s - mu_t) ** 2).sum() /
                 (0.5 * ((mu_s ** 2).sum() + (mu_t ** 2).sum()) + 1e-12))


def probe_acc(x, y, n_pca=4):
    """Linear domain probe on a tightly compressed PCA subspace (chance=0.5)."""
    z = np.concatenate([x, y])
    c = np.array([0] * len(x) + [1] * len(y))
    zp = PCA(n_components=min(n_pca, z.shape[0] - 1, z.shape[1]),
             random_state=SEED).fit_transform(z)
    return float(cross_val_score(LogisticRegression(max_iter=2000), zp, c,
                                 cv=5).mean())


def pooled_layer_features(net, imgs, device):
    """GAP-pooled features per encoder layer; src-branch hook firing only."""
    feats = {l: [] for l in LAYERS}
    base = net.base if hasattr(net, 'base') else net
    fired = {}

    def mk_hook(name):
        def hook(mod, inp, out):
            if fired.get(name, 0) == 0:  # first firing = src branch
                feats[name].append(out.mean(dim=(2, 3)).squeeze(0).cpu().numpy())
            fired[name] = fired.get(name, 0) + 1
        return hook

    handles = [getattr(base, l).register_forward_hook(mk_hook(l)) for l in LAYERS]
    net.eval()
    with torch.no_grad():
        for img in imgs:
            fired.clear()
            x = torch.from_numpy(img).float()[None, None].to(device)
            ll2, ll3 = wavelet_branch(img)
            l2 = torch.from_numpy(ll2).float()[None, None].to(device)
            l3 = torch.from_numpy(ll3).float()[None, None].to(device)
            net(x, x, l2, l3)
    for h in handles:
        h.remove()
    return {l: np.asarray(v) for l, v in feats.items()}


def lowlevel_stats(paths):
    small, lap_var, means, stds = [], [], [], []
    for p in paths:
        img = imread01(p)
        means.append(float(img.mean()))
        stds.append(float(img.std()))
        t = torch.from_numpy(img)[None, None].float()
        lap = F.conv2d(t, torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]]))
        lap_var.append(float(lap.var()))
        small.append(F.interpolate(t, size=(64, 64), mode='area')
                     .numpy().ravel())
    return np.asarray(small), float(np.mean(lap_var)), float(np.mean(means)), \
        float(np.mean(stds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--models', default=None, help='comma subset of model keys')
    args = ap.parse_args()
    device = args.device

    xpaths = sorted(glob.glob(os.path.join(XRAY_LOW, '*.png')))[:N_PER]
    x_imgs = [imread01(p) for p in xpaths]
    cond_imgs, cond_paths = {}, {}
    for c, root in CONDITIONS.items():
        ps = sorted(glob.glob(os.path.join(root, '*.png')),
                    key=lambda p: int(os.path.basename(p).split('.')[0]))[:N_PER]
        cond_paths[c] = ps
        cond_imgs[c] = [imread01(p) for p in ps]
        print('condition %s: %d imgs' % (c, len(ps)))

    # --- low-level input statistics (model independent) ---
    rows = []
    x_small, x_lap, x_mean, x_std = lowlevel_stats(xpaths)
    for c in CONDITIONS:
        s, lap, m, sd = lowlevel_stats(cond_paths[c])
        rows.append(dict(model='input', layer='pixels', condition=c,
                         norm_dist=norm_dist(x_small, s),
                         lin_mmd2=float(((x_small.mean(0) - s.mean(0)) ** 2).sum()),
                         probe_acc=probe_acc(x_small, s),
                         lap_var=lap, int_mean=m, int_std=sd))
        print('input pixels X-ray<->%s: normDist=%.4f probe=%.3f lapVar=%.5f' %
              (c, rows[-1]['norm_dist'], rows[-1]['probe_acc'], lap))
    print('X-ray input: lapVar=%.5f mean=%.4f std=%.4f' % (x_lap, x_mean, x_std))

    # --- per-layer feature gaps ---
    keys = args.models.split(',') if args.models else list(MODELS)
    for tag in keys:
        run, wname = MODELS[tag]
        pth = os.path.join(RUNS, run, wname)
        if not os.path.exists(pth):
            pth = os.path.join(RUNS, run, 'best.pth')
            if not os.path.exists(pth):
                print('SKIP %s (no weights at %s)' % (tag, pth))
                continue
        net = UDANRes()
        net.load_state_dict(load_torch(pth, map_location='cpu')['state_dict'])
        net = net.to(device)
        fx = pooled_layer_features(net, x_imgs, device)
        for c in CONDITIONS:
            fn = pooled_layer_features(net, cond_imgs[c], device)
            for l in LAYERS:
                a, b = fx[l], fn[l]
                rows.append(dict(model=tag, layer=l, condition=c,
                                 norm_dist=norm_dist(a, b),
                                 lin_mmd2=float(((a.mean(0) - b.mean(0)) ** 2).sum()),
                                 probe_acc=probe_acc(a, b),
                                 lap_var=np.nan, int_mean=np.nan, int_std=np.nan))
            print('%s %s done' % (tag, c))
        del net

    import csv
    out_csv = os.path.join(HERE, 'feature_probe.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(HERE, 'feature_probe.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    print('saved', out_csv)


if __name__ == '__main__':
    main()
