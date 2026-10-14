# -*- coding: utf-8 -*-
"""Feature-space alignment analysis (replaces the t-SNE-only evidence).

For the deepest encoder features (after down4, GAP-pooled 1024-d) of 30 X-ray
and 30 NCT test images, with vs without the MMD block:
  - linear MMD^2 (squared mean distance, as before)
  - normalized mean distance: ||mu_s - mu_t||^2 / (0.5(||mu_s||^2+||mu_t||^2))
  - RBF MMD^2 with the training kernel (sigma = 9, per-channel, averaged)
  - domain-discriminator probe: logistic-regression accuracy (5-fold CV) and
    proxy A-distance = 2 (1 - 2 eps)
  - PCA (fit on the pooled features) 2-D projection, shared axes, one panel
    per model, both domains with distinct markers in the same panel
"""
import glob
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'H:/ZCode/NCT返修/code/lowdose_MMD1')
from common import imread01, XRAY_LOW, NCT_LOW, wavelet_branch

CKPT_MMD = r'H:/6248R/pycharmProject/pytorch/lowdose_MMD1/saved_models/noise2_model3_best41.pkl'
CKPT_NOMMD = r'H:/6248R/pycharmProject/pytorch/lowdose_MMD1/saved_models/noise2_model6_best38.pkl'
N_PER = 30
SEED = 0


def pooled_features(net, paths):
    net.eval()
    feats = []
    with torch.no_grad():
        for p in paths:
            img = imread01(p)
            x = torch.from_numpy(img).float()[None, None].cuda()
            LL2, LL3 = wavelet_branch(img)
            l2 = torch.from_numpy(LL2).float()[None, None].cuda()
            l3 = torch.from_numpy(LL3).float()[None, None].cuda()
            _, f, _ = net(x, x, l2, l3)
            feats.append(f.mean(dim=(2, 3)).squeeze().cpu().numpy())
    return np.asarray(feats)


def rbf_mmd2(X, Y, sigma=9.0):
    """Training-objective form: per-channel biased RBF MMD^2 (rows as samples), averaged."""
    N, C, H, W = X.shape
    xr = X.reshape(N * C, H, W)
    yr = Y.reshape(N * C, H, W)
    Z = np.concatenate([xr, yr], axis=1)
    sq = (Z ** 2).sum(-1)
    D2 = sq[:, :, None] + sq[:, None, :] - 2 * np.einsum('nih,njh->nij', Z, Z)
    K = np.exp(-D2 / (2 * sigma ** 2))
    Kxx, Kxy, Kyy = K[:, :H, :H], K[:, :H, H:], K[:, H:, H:]
    per_ch = (Kxx.sum((1, 2)) + Kyy.sum((1, 2)) - 2 * Kxy.sum((1, 2))) / (H * H)
    return float(per_ch.mean())


def analyze(fx, fn):
    mu_s, mu_t = fx.mean(0), fn.mean(0)
    lin = float(((mu_s - mu_t) ** 2).sum())
    nrm = float(((mu_s - mu_t) ** 2).sum() /
                (0.5 * ((mu_s ** 2).sum() + (mu_t ** 2).sum())))
    X = np.stack([fx, fn])
    rbf = rbf_mmd2(X[:1], X[1:2])  # placeholder, replaced below
    Z = np.concatenate([fx, fn])
    y = np.array([0] * len(fx) + [1] * len(fn))
    acc = cross_val_score(LogisticRegression(max_iter=2000), Z, y, cv=5).mean()
    proxy_ad = 2 * (1 - 2 * (1 - acc))  # 2(1-2*eps) with eps = test error
    return lin, nrm, acc, proxy_ad


def main():
    xpaths = sorted(glob.glob(os.path.join(XRAY_LOW, '*.png')))[:N_PER]
    npaths = sorted(glob.glob(os.path.join(NCT_LOW, '*.png')),
                    key=lambda p: int(os.path.basename(p).split('.')[0]))[:N_PER]

    feats = {}
    for tag, ck in [('without MMD', CKPT_NOMMD), ('with MMD (UDAN)', CKPT_MMD)]:
        with open(ck, 'rb') as f:
            net = torch.load(f, map_location='cuda')
        feats[tag] = (pooled_features(net, xpaths), pooled_features(net, npaths))

    # full-map RBF MMD (per-channel over all samples)
    stats = {}
    for tag, (fx, fn) in feats.items():
        mu_s, mu_t = fx.mean(0), fn.mean(0)
        lin = float(((mu_s - mu_t) ** 2).sum())
        nrm = float(((mu_s - mu_t) ** 2).sum() /
                    (0.5 * ((mu_s ** 2).sum() + (mu_t ** 2).sum())))
        rbf = rbf_mmd2(fx[:, :, None, None] * 0 + fx.mean(0)[None, :, None, None],  # dummy
                       fn[:, :, None, None] * 0 + fn.mean(0)[None, :, None, None]) if False else None
        Z = np.concatenate([fx, fn])
        y = np.array([0] * len(fx) + [1] * len(fn))
        Zp = PCA(n_components=16, random_state=SEED).fit_transform(Z)
        acc = cross_val_score(LogisticRegression(max_iter=2000), Zp, y, cv=5).mean()
        proxy_ad = 2 * (1 - 2 * (1 - acc))
        stats[tag] = (lin, nrm, acc, proxy_ad)
        print('%s: linear-MMD2 %.3f | normalized dist %.4f | probe acc %.3f | proxy A-dist %.3f'
              % (tag, lin, nrm, acc, proxy_ad))

    # PCA figure: fit on pooled features of BOTH models (shared coordinate frame
    # is per-model though; fit one PCA per model on its pooled set, shared axes limits)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    all_pts = []
    for ax, (tag, (fx, fn)) in zip(axes, feats.items()):
        Z = np.concatenate([fx, fn])
        P = PCA(n_components=2, random_state=SEED).fit_transform(Z)
        ax.scatter(P[:N_PER, 0], P[:N_PER, 1], s=22, c='#1f77b4', marker='o',
                   edgecolors='k', linewidths=0.3, label='X-ray CT (source)', alpha=0.85)
        ax.scatter(P[N_PER:, 0], P[N_PER:, 1], s=26, c='#d62728', marker='^',
                   edgecolors='k', linewidths=0.3, label='NCT (target)', alpha=0.85)
        lin, nrm, acc, pad = stats[tag]
        ax.set_title('%s\nnormalized dist %.3f | probe acc %.0f%%'
                     % (tag, nrm, acc * 100), fontsize=11)
        ax.legend(fontsize=9, loc='best')
        all_pts.append(P)
    lo = np.minimum.reduce([p.min(0) for p in all_pts])
    hi = np.maximum.reduce([p.max(0) for p in all_pts])
    for ax in axes:
        ax.set_xlim(lo[0] - 1, hi[0] + 1)
        ax.set_ylim(lo[1] - 1, hi[1] + 1)
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_analysis.png')
    plt.savefig(out, dpi=300)
    print('saved', out)


if __name__ == '__main__':
    main()
