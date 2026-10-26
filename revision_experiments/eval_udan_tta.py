# -*- coding: utf-8 -*-
"""Evaluate checkpoints with horizontal-flip self-ensembling (TTA) on both test
sets and APPEND rows 'UDAN-FINAL(TTA)@<ts>' / '<NAME>(TTA)@<ts>' to
eval_matrix/summary.csv for downstream table assembly.

Usage:
  python eval_udan_tta.py final <run> [--ema]          -> UDAN-FINAL(TTA)@ts
  python eval_udan_tta.py row <NAME> <run> [--ema]     -> NAME(TTA)@ts
"""
import os
import sys
import glob
import csv

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from math import sqrt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'H:/ZCode/NCT返修/code/lowdose_MMD1')
from common import imread01, wavelet_branch, load_torch
from nets import UDANRes, UDANResNoWT

RUNS = r'H:/ZCode/NCT返修/experiments/runs'
FDK = r'H:/6248R/CodeMATLAB/result/FDK'
_YY, _XX = np.mgrid[0:512, 0:512]
CIRC = (_YY - 256) ** 2 + (_XX - 256) ** 2 <= 240 ** 2


def eval_tta(run, use_ema, disp):
    ck_name = 'ema_best.pth' if use_ema else 'best.pth'
    pth = os.path.join(RUNS, run, ck_name)
    if not os.path.exists(pth):
        pth = os.path.join(RUNS, run, 'best.pth')
    net = UDANResNoWT() if run.startswith('UDANRes-noWT') else UDANRes()
    ck = load_torch(pth, map_location='cpu')
    net.load_state_dict(ck['state_dict'])
    net = net.cuda().eval()
    rows = []
    for ts, inp, ref in [('1e3', FDK + '/FDK_raw_200Views_noise1e3',
                          FDK + '/FDK_raw_200Views'),
                         ('3e2', FDK + '/FDK_raw_200Views_noise3e2',
                          FDK + '/FDK_raw_200Views'),
                         ('3e3', FDK + '/FDK_raw_200Views_noise3e3',
                          FDK + '/FDK_raw_200Views'),
                         ('8e2', FDK + '/FDK_raw_200Views_noise8e2',
                          FDK + '/FDK_raw_200Views'),
                         ('7e2', FDK + '/FDK_raw_200Views_noise7e2',
                          FDK + '/FDK_raw_200Views'),
                         ('1e4', FDK + '/FDK_raw_200Views_noise1e4',
                          FDK + '/FDK_raw_200Views')]:
        refs = {os.path.basename(p): p for p in glob.glob(os.path.join(ref, '*.png'))}
        ss_, ps_, rs_ = [], [], []
        for p in sorted(glob.glob(os.path.join(inp, '*.png'))):
            n = os.path.basename(p)
            if n not in refs:
                continue
            low = imread01(p)
            r_ = imread01(refs[n])
            x = torch.from_numpy(low).float()[None, None].cuda()
            with torch.no_grad():
                LL2, LL3 = wavelet_branch(low)
                l2 = torch.from_numpy(LL2).float()[None, None].cuda()
                l3 = torch.from_numpy(LL3).float()[None, None].cuda()
                o1 = net(x, x, l2, l3)[0].detach()
                lowf = np.ascontiguousarray(low[:, ::-1])
                xf = torch.flip(x, dims=[3])
                L2f, L3f = wavelet_branch(lowf)
                l2f = torch.from_numpy(L2f).float()[None, None].cuda()
                l3f = torch.from_numpy(L3f).float()[None, None].cuda()
                o2 = torch.flip(net(xf, xf, l2f, l3f)[0].detach(), dims=[3])
            out = np.clip((0.5 * (o1 + o2)).squeeze().cpu().numpy(), 0, 1)
            a, b = r_[CIRC], out[CIRC]
            ss_.append(float(ssim(r_ * CIRC, out * CIRC, data_range=1)))
            ps_.append(float(psnr(a, b, data_range=1)))
            rs_.append(float(sqrt(np.mean((a - b) ** 2))))
        rows.append([ts, '%s(TTA)@%s' % (disp, ts), 'r240', len(ss_),
                     np.mean(ss_), np.std(ss_), np.mean(ps_), np.std(ps_),
                     np.mean(rs_), np.std(rs_)])
        print('%s %s(TTA) r240 SSIM %.4f±%.4f PSNR %.2f (n=%d, ckpt %s ep%s)'
              % (ts, disp, np.mean(ss_), np.std(ss_), np.mean(ps_), len(ss_),
                 run, ck.get('epoch')))
    return rows


def append_rows(rows):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'eval_matrix', 'summary.csv')
    keep = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for r in csv.reader(f):
                if len(r) > 1 and any(r[1] == nr[1] for nr in rows):
                    continue
                keep.append(r)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerows(keep)
        w.writerows(rows)
    print('updated', path)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'final':
        rows = eval_tta(sys.argv[2], '--ema' in sys.argv, 'UDAN-FINAL')
    else:
        rows = eval_tta(sys.argv[3], '--ema' in sys.argv, sys.argv[2])
    append_rows(rows)
