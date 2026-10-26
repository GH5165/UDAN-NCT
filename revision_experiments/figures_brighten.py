# -*- coding: utf-8 -*-
"""Brightness-normalize all manuscript figure panels (light_norm.aug recipe:
1st/99th percentile clip + min-max stretch to 25.5..229.5) and save the
subfigures to code/lowdose_MMD1/qoder_fig for inspection.
fig9 PCA scatter panels are copied unchanged (not photographic content).
"""
import os
import numpy as np
from PIL import Image

SRC = ''  # TODO: set local path
DST = ''  # TODO: set local path
os.makedirs(DST, exist_ok=True)


def aug(img):
    """Identical to light_norm.aug: percentile clip then NORM_MINMAX stretch."""
    a = np.asarray(img).astype(np.float64)
    hi, lo = np.percentile(a, 99), np.percentile(a, 1)
    a = np.clip(a, lo, hi)
    if hi > lo:
        a = (a - lo) / (hi - lo)
    a = 255 * 0.1 + a * (255 * 0.9 - 255 * 0.1)
    return a


n = 0
for fn in sorted(os.listdir(SRC)):
    if not fn.lower().endswith('.png'):
        continue
    img = Image.open(os.path.join(SRC, fn))
    if fn.startswith('fig9'):
        img.save(os.path.join(DST, fn))  # scatter plot: no brightness change
        continue
    gray = img.convert('L')
    out = Image.fromarray(np.clip(np.round(aug(gray)), 0, 255).astype(np.uint8))
    out.save(os.path.join(DST, fn))
    n += 1
print('brightness-normalized:', n, 'panels ->', DST)
