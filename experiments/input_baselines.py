# -*- coding: utf-8 -*-
"""Input-baseline SSIM/PSNR (r240) of FDK at each flux level vs the normal-flux reference."""
import csv
import glob
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import imread01

FDK = ''  #
REF = os.path.join(FDK, 'FDK_raw_200Views')
LEVELS = ['1e4', '3e3', '2e3', '1e3', '9e2', '8e2', '7e2', '6e2', '5e2', '3e2']

_YY, _XX = np.mgrid[0:512, 0:512]
CIRC = (_YY - 256) ** 2 + (_XX - 256) ** 2 <= 240 ** 2

rows = []
for lv in LEVELS:
    root = os.path.join(FDK, 'FDK_raw_200Views_noise%s' % lv)
    if not os.path.isdir(root):
        print('%-4s MISSING' % lv)
        continue
    refs = {os.path.basename(p): p for p in glob.glob(os.path.join(REF, '*.png'))}
    ss, ps = [], []
    for p in sorted(glob.glob(os.path.join(root, '*.png')),
                    key=lambda p: int(os.path.basename(p).split('.')[0])):
        n = os.path.basename(p)
        if n not in refs:
            continue
        out = imread01(p)
        ref = imread01(refs[n])
        ss.append(float(ssim(ref * CIRC, out * CIRC, data_range=1)))
        ps.append(float(psnr(ref[CIRC], out[CIRC], data_range=1)))
    rows.append((lv, len(ss), np.mean(ss), np.std(ss), np.mean(ps), np.std(ps)))
    print('%-4s n=%d SSIM %.4f±%.3f PSNR %.2f±%.2f' % rows[-1])

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'input_baselines_levels.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['level', 'n', 'SSIM_mean', 'SSIM_sd', 'PSNR_mean', 'PSNR_sd'])
    w.writerows(rows)
print('saved input_baselines_levels.csv')
