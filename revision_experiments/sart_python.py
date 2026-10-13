# -*- coding: utf-8 -*-
"""SART reconstruction of the low-flux NCT test slices (Python, scikit-image).

Revised protocol (documented in the manuscript):
  1. Re-project the low-flux FDK slice: p = radon(x, 360 views over [0,180), circle).
  2. Simulate low-count acquisition, TIGRE addCTnoise-style:
     pn = Poisson(I0 * exp(-p / p.max())), I0 = 5e4;
     b = max(-log(max(pn, 1) / I0), 0).
  3. SART: 200 iterations (iradon_sart, relaxation 0.15), output min-max
     normalized to [0, 1] for storage.

Replaces the original demo1_ART.m run, whose output writer stored the input
image instead of the reconstruction and whose sinogram normalization was
degenerate (log of unnormalized 0-255 radon values saturates the sinogram).
"""
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
from multiprocessing import Pool
from skimage.transform import radon, iradon_sart
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import NCT_LOW, imread01

VIEW = 360
THETA = np.linspace(0, 180, VIEW, endpoint=False)
I0 = 5e4
NITER = 200
RELAX = 0.15
SEED = 2024


def sart_recon(args):
    name, seed = args
    rng = np.random.RandomState(seed)
    x = (imread01(os.path.join(NCT_LOW, name)) * 255.0).astype(np.float64)
    p = radon(x, THETA, circle=True)                       # (512, 360)
    pn = rng.poisson(I0 * np.exp(-p / p.max()))
    b = np.maximum(-np.log(np.maximum(pn, 1) / I0), 0)

    recon = None
    for _ in range(NITER):
        recon = iradon_sart(b, theta=THETA, image=recon, relaxation=RELAX)
    recon = np.maximum(recon, 0)
    out = recon / recon.max() if recon.max() > 0 else recon
    return name, out


def main():
    names = sorted([f for f in os.listdir(NCT_LOW) if f.endswith('.png')],
                   key=lambda s: int(s.split('.')[0]))
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'runs', 'SART', 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    tasks = [(n, SEED + i) for i, n in enumerate(names)]
    with Pool(processes=6) as pool:
        for i, (name, out) in enumerate(pool.imap_unordered(sart_recon, tasks)):
            Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(
                os.path.join(out_dir, name))
            print(f'SART {i + 1}/{len(tasks)}: {name}', flush=True)


if __name__ == '__main__':
    main()
