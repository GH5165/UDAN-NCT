# -*- coding: utf-8 -*-
"""Feature-space alignment analysis on the I0=1e3 low-flux regime.

Deepest encoder features (after down4, GAP-pooled 1024-d) of 30 X-ray source
images and 30 NCT 1e3 target images, for the label-free UDAN (with MMD) vs
the same architecture trained without MMD (direct transfer):
linear MMD^2, normalized mean-embedding distance, domain probe (logistic
regression, 5-fold CV, 16-d PCA subspace), and a shared-axis PCA projection.
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
from common import imread01, XRAY_LOW, wavelet_branch, load_torch
from nets import UDANRes

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
NCT_1E3 = r'H:/6248R/CodeMATLAB/result/FDK/FDK_raw_200Views_noise1e3'
UDAN_RUN = os.environ.get('UDAN_RUN', 'U1c')
NOMMD_RUN = 'UDANRes-noMMD'
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


def main():
    xpaths = sorted(glob.glob(os.path.join(XRAY_LOW, '*.png')))[:N_PER]
    npaths = sorted(glob.glob(os.path.join(NCT_1E3, '*.png')),
                    key=lambda p: int(os.path.basename(p).split('.')[0]))[:N_PER]

    feats = {}
    for tag, run in [('without MMD', NOMMD_RUN), ('with MMD (UDAN)', UDAN_RUN)]:
        net = UDANRes()
        pth = os.path.join(RUNS, run, 'ema_best.pth')
        if not (run == UDAN_RUN and os.path.exists(pth)):
            pth = os.path.join(RUNS, run, 'best.pth')
        ck = load_torch(pth, map_location='cpu')
        net.load_state_dict(ck['state_dict'])
        net = net.cuda()
        feats[tag] = (pooled_features(net, xpaths), pooled_features(net, npaths))
        print(tag, 'features done')

    stats = {}
    for tag, (fx, fn) in feats.items():
        mu_s, mu_t = fx.mean(0), fn.mean(0)
        lin = float(((mu_s - mu_t) ** 2).sum())
        nrm = float(((mu_s - mu_t) ** 2).sum() /
                    (0.5 * ((mu_s ** 2).sum() + (mu_t ** 2).sum())))
        Z = np.concatenate([fx, fn])
        y = np.array([0] * len(fx) + [1] * len(fn))
        Zp = PCA(n_components=16, random_state=SEED).fit_transform(Z)
        acc = cross_val_score(LogisticRegression(max_iter=2000), Zp, y, cv=5).mean()
        stats[tag] = (lin, nrm, acc)
        print('%s: linear-MMD2 %.3f | normalized dist %.4f | probe acc %.3f'
              % (tag, lin, nrm, acc))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    all_pts = []
    for ax, (tag, (fx, fn)) in zip(axes, feats.items()):
        Z = np.concatenate([fx, fn])
        P = PCA(n_components=2, random_state=SEED).fit_transform(Z)
        ax.scatter(P[:N_PER, 0], P[:N_PER, 1], s=22, c='#1f77b4', marker='o',
                   edgecolors='k', linewidths=0.3, label='X-ray CT (source)', alpha=0.85)
        ax.scatter(P[N_PER:, 0], P[N_PER:, 1], s=26, c='#d62728', marker='^',
                   edgecolors='k', linewidths=0.3, label='NCT (target)', alpha=0.85)
        lin, nrm, acc = stats[tag]
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
    out = os.path.join(HERE, 'feature_analysis_1e3.png')
    plt.savefig(out, dpi=300)
    print('saved', out)

    import json
    with open(os.path.join(HERE, 'feature_stats_1e3.json'), 'w') as f:
        json.dump({k: {'linear_mmd2': v[0], 'norm_dist': v[1], 'probe_acc': v[2]}
                   for k, v in stats.items()}, f, indent=1)


if __name__ == '__main__':
    main()
