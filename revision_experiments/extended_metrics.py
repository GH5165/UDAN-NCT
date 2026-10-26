# -*- coding: utf-8 -*-
"""Extended metric analysis: unified protocol + detail-preservation metrics.

For every method folder (or the original UDAN checkpoint) compute over the 31
NCT test pairs:
  SSIM / PSNR / RMSE (data_range=1, [0,1] floats)      -- unified protocol
  hp_corr  : correlation of high-pass residual with reference high-pass
             (detail/texture preservation; 1 = perfect)
  EPI      : edge preservation index (gradient correlation)
  delta_std: |std(img) - std(ref)| / std(ref)          (contrast bias)
"""
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from scipy.ndimage import uniform_filter, sobel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '')  #   #
from common import imread01, nct_test_pairs, metrics, NCT_REF, NCT_LOW, wavelet_branch


def highpass(a):
    return a - uniform_filter(a, 5)


def epi(ref, img):
    gx_r, gy_r = sobel(ref), sobel(ref, axis=1)
    gx_i, gy_i = sobel(img), sobel(img, axis=1)
    g_r = np.hypot(gx_r, gy_r).flatten()
    g_i = np.hypot(gx_i, gy_i).flatten()
    return float(np.corrcoef(g_r, g_i)[0, 1])


def analyze_arrays(get_img, pairs, tag, results):
    s_l, p_l, r_l, hp_l, e_l, d_l = [], [], [], [], [], []
    for ip, rp, name in pairs:
        ref = imread01(rp)
        img = np.clip(get_img(ip, name), 0, 1)
        s, p, r = metrics(ref, img)
        s_l.append(s); p_l.append(p); r_l.append(r)
        hp_l.append(np.corrcoef(highpass(img).flatten(),
                                highpass(ref).flatten())[0, 1])
        e_l.append(epi(ref, img))
        d_l.append(abs(img.std() - ref.std()) / ref.std())
    results[tag] = {
        'SSIM': np.mean(s_l), 'PSNR': np.mean(p_l), 'RMSE': np.mean(r_l),
        'hp_corr': np.mean(hp_l), 'EPI': np.mean(e_l),
        'contrast_bias': np.mean(d_l),
        'SSIM_std': np.std(s_l), 'PSNR_std': np.std(p_l), 'RMSE_std': np.std(r_l),
    }


def main():
    pairs = nct_test_pairs()
    results = {}

    analyze_arrays(lambda ip, name: imread01(ip), pairs, 'FDK_input(noisy)', results)

    for run in ['DnCNN-Xray', 'FBPConvNet-Xray', 'REDCNN-Xray']:
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'runs', run, 'outputs')
        if os.path.isdir(folder):
            analyze_arrays(lambda ip, name, f=folder: imread01(os.path.join(f, name)),
                           pairs, run, results)

    # original UDAN best41 checkpoint (unified protocol inference)
    ckpt_path = ''  #
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            net = torch.load(f, map_location='cuda')
        net.eval()

        def udan_img(ip, name):
            low = imread01(ip)
            x = torch.from_numpy(low).float()[None, None].cuda()
            LL2, LL3 = wavelet_branch(low)
            l2 = torch.from_numpy(LL2).float()[None, None].cuda()
            l3 = torch.from_numpy(LL3).float()[None, None].cuda()
            with torch.no_grad():
                out, _, _ = net(x, x, l2, l3)
            return out.squeeze().cpu().numpy()

        analyze_arrays(udan_img, pairs, 'UDAN-orig-best41', results)

    # per-image min-max normalized variant (this work's display protocol)
    def norm_img(a):
        a = a.astype(float)
        return (a - a.min()) / (a.max() - a.min() + 1e-12)

    for run in ['DnCNN-Xray', 'FBPConvNet-Xray', 'REDCNN-Xray']:
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'runs', run, 'outputs')
        if os.path.isdir(folder):
            analyze_arrays(lambda ip, name, f=folder: norm_img(imread01(os.path.join(f, name))),
                           pairs, run + '(norm)', results)

    hdr = f"{'method':22s} {'SSIM':>6} {'PSNR':>7} {'RMSE':>7} {'hp_corr':>8} {'EPI':>6} {'c_bias':>7}"
    print(hdr)
    print('-' * len(hdr))
    for k, v in results.items():
        print(f"{k:22s} {v['SSIM']:6.4f} {v['PSNR']:7.2f} {v['RMSE']:7.4f} "
              f"{v['hp_corr']:8.4f} {v['EPI']:6.4f} {v['contrast_bias']:7.4f}")

    import json
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'extended_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
