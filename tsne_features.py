# -*- coding: utf-8 -*-
"""t-SNE visualization of encoder bottleneck features, with vs without MMD.

Evidence for Editor comment 3 / Reviewer 4 comment 2: MMD alignment pulls the
NCT feature distribution toward the X-ray source distribution in the deepest
encoder space. Uses the original trained checkpoints
(noise2_model3_best41 = with MMD, noise2_model6_best38 = without MMD).
"""
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'H:/ZCode/NCT返修/code/lowdose_MMD1')
from common import imread01, XRAY_LOW, NCT_LOW, wavelet_branch

CKPT_MMD = r'H:/6248R/pycharmProject/pytorch/lowdose_MMD1/saved_models/noise2_model3_best41.pkl'
CKPT_NOMMD = r'H:/6248R/pycharmProject/pytorch/lowdose_MMD1/saved_models/noise2_model6_best38.pkl'
N_PER_DOMAIN = 30


def pooled_features(net, paths, device='cuda'):
    net.eval()
    feats = []
    with torch.no_grad():
        for p in paths:
            img = imread01(p)
            x = torch.from_numpy(img).float()[None, None].to(device)
            LL2, LL3 = wavelet_branch(img)
            l2 = torch.from_numpy(LL2).float()[None, None].to(device)
            l3 = torch.from_numpy(LL3).float()[None, None].to(device)
            out, f_src, _ = net(x, x, l2, l3)
            feats.append(f_src.mean(dim=(2, 3)).squeeze().cpu().numpy())
    return np.asarray(feats)


def mmd2_linear(X, Y):
    delta = X.mean(0) - Y.mean(0)
    return float(delta @ delta)


def main():
    import glob
    xray_paths = sorted(glob.glob(os.path.join(XRAY_LOW, '*.png')))[:N_PER_DOMAIN]
    nct_paths = sorted(glob.glob(os.path.join(NCT_LOW, '*.png')),
                       key=lambda p: int(os.path.basename(p).split('.')[0]))[:N_PER_DOMAIN]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    stats = {}
    for ax, (tag, ckpt) in zip(axes, [('with MMD (UDAN)', CKPT_MMD),
                                      ('without MMD', CKPT_NOMMD)]):
        with open(ckpt, 'rb') as f:
            net = torch.load(f, map_location='cuda')
        fx = pooled_features(net, xray_paths)
        fn = pooled_features(net, nct_paths)
        d2 = mmd2_linear(fx, fn)
        stats[tag] = d2
        Z = np.concatenate([fx, fn])
        Z2 = TSNE(n_components=2, init='pca', random_state=0,
                  perplexity=15).fit_transform(Z)
        ax.scatter(Z2[:N_PER_DOMAIN, 0], Z2[:N_PER_DOMAIN, 1],
                   s=18, c='#1f77b4', label='X-ray CT (source)', alpha=0.8)
        ax.scatter(Z2[N_PER_DOMAIN:, 0], Z2[N_PER_DOMAIN:, 1],
                   s=18, c='#d62728', marker='^', label='NCT (target)', alpha=0.8)
        ax.set_title(f'{tag}\nlinear-MMD$^2$ = {d2:.1f}', fontsize=11)
        ax.legend(fontsize=9, loc='best')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'tsne_features.png')
    plt.savefig(out, dpi=300)
    print('saved', out)
    print('linear MMD^2 (pooled feature mean distance):', stats)


if __name__ == '__main__':
    main()
