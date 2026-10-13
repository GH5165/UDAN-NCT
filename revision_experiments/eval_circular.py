# -*- coding: utf-8 -*-
"""Circular-mask evaluation + UDAN error decomposition.

1) Evaluate all methods with metrics computed inside the inscribed circle of
   the reconstruction FOV (user-requested protocol; excludes FDK peripheral
   artifacts). Radii {200, 224, 240, 250} tested on all saved outputs.
2) Decompose UDAN's error: per-image global affine correction (a*out+b -> ref,
   least squares) separates photometric calibration error from structural
   error. If affine-corrected PSNR jumps, the deficit is calibration, not
   structure.
"""
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from math import sqrt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import imread01, nct_test_pairs

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')

METHOD_FOLDERS = ['DnCNN-Xray', 'FBPConvNet-Xray', 'REDCNN-Xray', 'SART',
                  'UDAN-repro', 'UDAN-noMMD', 'UDAN-noWT', 'UDAN-residual',
                  'UDAN-supNCT', 'DnCNN-MMD', 'REDCNN-MMD', 'UDAN-1e3']


def circle_mask(r):
    yy, xx = np.mgrid[0:512, 0:512]
    return (yy - 256) ** 2 + (xx - 256) ** 2 <= r * r


def eval_folder(folder, pairs, mask):
    s_l, p_l, r_l = [], [], []
    for ip, rp, name in pairs:
        fp = os.path.join(RUNS, folder, 'outputs', name)
        if not os.path.exists(fp):
            return None
        ref = np.where(mask, imread01(rp), 0.0)
        img = np.where(mask, np.clip(imread01(fp), 0, 1), 0.0)
        s_l.append(ssim(ref, img, data_range=1))
        p_l.append(psnr(ref, img, data_range=1))
        r_l.append(sqrt(mse(ref, img)))
    return np.mean(s_l), np.mean(p_l), np.mean(r_l)


def eval_input(pairs, mask):
    s_l, p_l, r_l = [], [], []
    for ip, rp, name in pairs:
        ref = np.where(mask, imread01(rp), 0.0)
        img = np.where(mask, imread01(ip), 0.0)
        s_l.append(ssim(ref, img, data_range=1))
        p_l.append(psnr(ref, img, data_range=1))
        r_l.append(sqrt(mse(ref, img)))
    return np.mean(s_l), np.mean(p_l), np.mean(r_l)


def affine_corrected(folder, pairs):
    """Per-image LS affine fit out->ref on the crop; returns corrected metrics."""
    s_l, p_l, r_l = [], [], []
    for ip, rp, name in pairs:
        fp = os.path.join(RUNS, folder, 'outputs', name)
        if not os.path.exists(fp):
            return None
        ref = imread01(rp)[32:-32, 32:-32]
        img = np.clip(imread01(fp), 0, 1)[32:-32, 32:-32]
        A = np.stack([img.flatten(), np.ones_like(img.flatten())], axis=1)
        coef, *_ = np.linalg.lstsq(A, ref.flatten(), rcond=None)
        corr = np.clip(coef[0] * img + coef[1], 0, 1)
        s_l.append(ssim(ref, corr, data_range=1))
        p_l.append(psnr(ref, corr, data_range=1))
        r_l.append(sqrt(mse(ref, corr)))
    return np.mean(s_l), np.mean(p_l), np.mean(r_l)


def main():
    pairs = nct_test_pairs()
    print('=== circular-mask evaluation (inscribed circle of FOV) ===')
    hdr = '%-18s' % 'method'
    for r in [200, 224, 240, 250]:
        hdr += ' | r=%d SSIM/PSNR ' % r
    print(hdr)
    rows = [('FDK input', None)] + [(m, m) for m in METHOD_FOLDERS]
    for label, folder in rows:
        line = '%-18s' % label
        for r in [200, 224, 240, 250]:
            mask = circle_mask(r)
            if folder is None:
                s, p, m = eval_input(pairs, mask)
            else:
                res = eval_folder(folder, pairs, mask)
                if res is None:
                    line += ' |    --/--     '
                    continue
                s, p, m = res
            line += ' | %.3f/%5.2f' % (s, p)
        print(line)

    print()
    print('=== UDAN error decomposition (per-image affine correction, crop 448) ===')
    for folder in ['UDAN-repro', 'UDAN-noMMD', 'UDAN-residual', 'DnCNN-Xray']:
        raw = eval_folder(folder, pairs, np.ones((512, 512), bool))
        aff = affine_corrected(folder, pairs)
        if raw and aff:
            print('%-16s raw SSIM %.3f PSNR %5.2f | affine-corrected SSIM %.3f PSNR %5.2f'
                  % (folder, raw[0], raw[1], aff[0], aff[1]))


if __name__ == '__main__':
    main()
