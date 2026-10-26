# -*- coding: utf-8 -*-
"""Final evaluation of every method under the unified protocol v2.

Protocol v2: images /255 in [0,1]; metrics (SSIM, PSNR with data_range=1,
RMSE) computed on the central 448x448 crop (margin=32); 31 name-matched test
pairs; mean +/- SD.  Also computes auxiliary detail metrics (hp_corr, EPI,
contrast bias) and the same metrics on the harsher noise1e3 test set.

Outputs: runs/final_table.json  (+ printed markdown table)
"""
import os
import sys
import json

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from scipy.ndimage import uniform_filter, sobel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '')  # TODO: set local path
from common import (imread01, nct_test_pairs, metrics, crop_center,
                    wavelet_branch, save_torch, load_torch,
                    NCT_LOW, NCT_LOW_1E3, NCT_REF)
from nets import UDAN, UDANNoWT, DnCNNMMD, REDCNNMMD, UDANRes

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')


def highpass(a):
    return a - uniform_filter(a, 5)


def epi(ref, img):
    g_r = np.hypot(sobel(ref), sobel(ref, axis=1)).flatten()
    g_i = np.hypot(sobel(img), sobel(img, axis=1)).flatten()
    return float(np.corrcoef(g_r, g_i)[0, 1])


def eval_arrays(imgs, refs, crop=32):
    rows = []
    for o, r in zip(imgs, refs):
        o = np.clip(o, 0, 1)
        r = np.clip(r, 0, 1)
        oc, rc = crop_center(o, crop), crop_center(r, crop)
        s, p, m = metrics(rc, oc)
        rows.append((s, p, m, np.corrcoef(highpass(oc).flatten(), highpass(rc).flatten())[0, 1],
                     epi(rc, oc), abs(oc.std() - rc.std()) / rc.std()))
    a = np.asarray(rows)
    return {k: (float(a[:, i].mean()), float(a[:, i].std()))
            for i, k in enumerate(['SSIM', 'PSNR', 'RMSE', 'hp_corr', 'EPI', 'cbias'])}


def load_run_net(run, ckpt_name='best.pth'):
    """Rebuild the right network for a run and load its best checkpoint."""
    path = os.path.join(RUNS, run, ckpt_name)
    model_map = {
        'UDAN-repro': lambda: UDAN(), 'UDAN-noMMD': lambda: UDAN(),
        'UDAN-noWT': lambda: UDANNoWT(), 'UDAN-sigma1': lambda: UDAN(),
        'UDAN-sigma100': lambda: UDAN(), 'UDAN-lam0.5': lambda: UDAN(),
        'UDAN-lam2': lambda: UDAN(), 'UDAN-loc-down3': lambda: UDAN(mmd_loc='down3'),
        'UDAN-bs2': lambda: UDAN(), 'UDAN-bs4': lambda: UDAN(),
        'UDAN-1e3': lambda: UDAN(), 'UDAN-residual': lambda: UDANRes(),
        'UDAN-supNCT': None,  # udan architecture from original code
        'DnCNN-MMD': lambda: DnCNNMMD(), 'REDCNN-MMD': lambda: REDCNNMMD(),
    }
    if run == 'UDAN-supNCT':
        import model3
        net = model3.UNet(n_channels=1, n_classes=1)
    else:
        net = model_map[run]()
    ck = torch.load(os.path.join(RUNS, run, ckpt_name), map_location='cpu',
                    weights_only=False) if False else None
    # safe load with file handle (non-ASCII path)
    from common import load_torch
    ck = load_torch(path, map_location='cpu')
    net.load_state_dict(ck['state_dict'])
    return net


def net_outputs(net, pairs, plain=False, device='cuda'):
    net = net.to(device).eval()
    outs, refs = [], []
    with torch.no_grad():
        for ip, rp, name in pairs:
            low = imread01(ip)
            x = torch.from_numpy(low).float()[None, None].to(device)
            if plain:
                out = net(x)
            else:
                LL2, LL3 = wavelet_branch(low)
                l2 = torch.from_numpy(LL2).float()[None, None].to(device)
                l3 = torch.from_numpy(LL3).float()[None, None].to(device)
                r = net(x, x, l2, l3)
                out = r[0] if isinstance(r, tuple) else r
            outs.append(out.squeeze().cpu().numpy().astype(np.float64))
            refs.append(imread01(rp))
    return outs, refs


def folder_outputs(folder, pairs):
    outs, refs = [], []
    for ip, rp, name in pairs:
        fp = os.path.join(folder, name)
        if os.path.exists(fp):
            outs.append(imread01(fp))
            refs.append(imread01(rp))
    return outs, refs


def main():
    pairs14 = nct_test_pairs()
    pairs13 = nct_test_pairs(inp_root=NCT_LOW_1E3)
    results = {}

    # input baselines
    results['FDK_input(noisy,1e4)'] = eval_arrays(
        [imread01(ip) for ip, _, _ in pairs14], [imread01(rp) for _, rp, _ in pairs14])
    results['FDK_input(noisy,1e3)'] = eval_arrays(
        [imread01(ip) for ip, _, _ in pairs13], [imread01(rp) for _, rp, _ in pairs13])

    # image-folder methods
    for run in ['DnCNN-Xray', 'FBPConvNet-Xray', 'REDCNN-Xray', 'SART']:
        o, r = folder_outputs(os.path.join(RUNS, run, 'outputs'), pairs14)
        if o:
            results[run] = eval_arrays(o, r)
        # cross-evaluate on the harsher 1e3 inputs where meaningful
        if run == 'SART':
            continue

    # checkpoint methods (UDA-style nets) on 1e4
    plain_runs = {}  # run -> plain flag
    for run in ['UDAN-repro', 'UDAN-noMMD', 'UDAN-noWT', 'UDAN-sigma1', 'UDAN-sigma100',
                'UDAN-lam0.5', 'UDAN-lam2', 'UDAN-loc-down3', 'UDAN-bs2', 'UDAN-bs4',
                'UDAN-1e3', 'UDAN-residual', 'UDAN-supNCT', 'DnCNN-MMD', 'REDCNN-MMD']:
        if not os.path.exists(os.path.join(RUNS, run, 'best.pth')):
            print('skip (no ckpt):', run)
            continue
        net = load_run_net(run)
        o, r = net_outputs(net, pairs14, plain=False)
        results[run] = eval_arrays(o, r)

    # cross-domain generalization: direct transfer + UDAN + supNCT on noise1e3
    from net.DnCNN import DnCNN
    from net.RED_CNN.networks import RED_CNN
    from net.FBPConvNet.model import FBPCONVNet
    sys.path.insert(0, '')  # TODO: set local path
    for run in ['DnCNN-Xray', 'FBPConvNet-Xray', 'REDCNN-Xray']:
        ck = load_torch(os.path.join(RUNS, run, 'best.pth'), map_location='cpu')
        builder = {'DnCNN-Xray': lambda: DnCNN(channels=1, features=64, num_of_layers=17),
                   'FBPConvNet-Xray': lambda: FBPCONVNet(),
                   'REDCNN-Xray': lambda: RED_CNN(out_ch=96)}[run]
        net = builder()
        net.load_state_dict(ck['state_dict'])
        o, r = net_outputs(net, pairs13, plain=True)
        results[run + '@1e3'] = eval_arrays(o, r)
    for run in ['UDAN-repro', 'UDAN-1e3', 'UDAN-supNCT']:
        if os.path.exists(os.path.join(RUNS, run, 'best.pth')):
            net = load_run_net(run)
            o, r = net_outputs(net, pairs13, plain=False)
            results[run + '@1e3'] = eval_arrays(o, r)

    with open(os.path.join(RUNS, 'final_table.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    hdr = (f"{'method':26s} {'SSIM':>14} {'PSNR':>14} {'RMSE':>14} {'hp':>7} {'EPI':>7} {'cbias':>7}")
    print(hdr)
    print('-' * len(hdr))
    for k, v in results.items():
        print(f"{k:26s} {v['SSIM'][0]:.4f}±{v['SSIM'][1]:.4f} "
              f"{v['PSNR'][0]:6.2f}±{v['PSNR'][1]:4.2f} "
              f"{v['RMSE'][0]:.4f}±{v['RMSE'][1]:.4f} "
              f"{v['hp_corr'][0]:7.4f} {v['EPI'][0]:7.4f} {v['cbias'][0]:7.4f}")


if __name__ == '__main__':
    main()
