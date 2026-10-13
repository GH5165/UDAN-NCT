# -*- coding: utf-8 -*-
"""UDA training (UDAN and MMD-transplanted variants) for the revision experiments.

Faithful re-implementation of the original lowdose_MMD1/main.py training loop:
  - source: X-ray pairs (low-dose input -> normal-dose label), supervised MSE
  - target: unlabeled NCT images (FDK_raw3_noise_1e4 by default)
  - MMD on encoder features: per-sample, per-channel biased MMD^2 with a single
    RBF kernel (rows of each channel map used as samples, exactly as the
    original mix_rbf_mmd2 call chain), summed over channels and batch and
    normalized by (batch * n_channels/4) — reproducing the original
    `loss_mmd / (batch_size * 256)` for 1024 channels.
  - total loss = MSE + lambda_mmd * MMD   (lambda 1.0 reproduces the paper)

Configurable: --model {udan, udan_nowt, dncnn_mmd, redcnn_mmd},
--mmd {on, off}, --mmd_loc {down4, down3}, --sigma, --lam, --bs,
--target (target image folder), --eval_input/--eval_ref (epoch-wise eval set).

Usage examples:
  python train_udan.py --model udan --name UDAN
  python train_udan.py --model udan --mmd off --name UDAN-noMMD
  python train_udan.py --model udan --sigma 1 --name UDAN-sigma1
"""
import argparse
import csv
import os
import sys
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (PairDataset, TargetDataset, XRAY_FULL, XRAY_LOW,
                    NCT_TRAIN_LOW, NCT_REF, NCT_LOW, evaluate_model,
                    append_json, save_torch, load_torch, wavelet_branch)
from nets import UDAN, UDANNoWT, DnCNNMMD, REDCNNMMD, UDANRes


def mmd2_rbf_rows(X, Y, sigma):
    """Vectorized biased MMD^2, single RBF kernel, rows-as-samples per channel.

    X, Y: [N, C, H, W]. For each sample n and channel c the H row-vectors of
    X[n, c] and Y[n, c] are the two empirical sets (matching the original
    per-channel mmd_loss(s_coder[j, k, :, :], t_coder[j, k, :, :])).
    Returns per-(n, c) MMD^2 tensor of shape [N, C].
    """
    N, C, H, W = X.shape
    Xr = X.reshape(N * C, H, W)
    Yr = Y.reshape(N * C, H, W)
    Z = torch.cat([Xr, Yr], dim=1)                    # [N*C, 2H, W]
    sq = (Z * Z).sum(-1)                              # [N*C, 2H]
    D2 = sq[:, :, None] + sq[:, None, :] - 2 * Z @ Z.transpose(1, 2)
    K = torch.exp(-D2 / (2 * sigma ** 2))             # [N*C, 2H, 2H]
    Kxx = K[:, :H, :H]
    Kxy = K[:, :H, H:]
    Kyy = K[:, H:, H:]
    mmd2 = (Kxx.sum(dim=(1, 2)) + Kyy.sum(dim=(1, 2)) - 2 * Kxy.sum(dim=(1, 2))) / (H * H)
    return mmd2.reshape(N, C)


def build_model(name, mmd_loc):
    if name == 'udan':
        return UDAN(mmd_loc=mmd_loc)
    if name == 'udan_nowt':
        return UDANNoWT()
    if name == 'dncnn_mmd':
        return DnCNNMMD()
    if name == 'redcnn_mmd':
        return REDCNNMMD()
    if name == 'udan_res':
        return UDANRes()
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='udan',
                    choices=['udan', 'udan_nowt', 'dncnn_mmd', 'redcnn_mmd', 'udan_res'])
    ap.add_argument('--name', default=None)
    ap.add_argument('--mmd', default='on', choices=['on', 'off'])
    ap.add_argument('--mmd_loc', default='down4', choices=['down4', 'down3'])
    ap.add_argument('--sigma', type=float, default=9.0,
                    help='RBF kernel sigma (baseline 9 = original GAMMA=10^3 bitwise XOR)')
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--bs', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--target', default=NCT_TRAIN_LOW)
    ap.add_argument('--eval_input', default=NCT_LOW)
    ap.add_argument('--eval_ref', default=NCT_REF)
    ap.add_argument('--seed', type=int, default=2024)
    args = ap.parse_args()

    run_name = args.name or args.model
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', run_name)
    os.makedirs(out_root, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = build_model(args.model, args.mmd_loc).to(device)
    needs_wt = args.model in ('udan', 'udan_nowt')

    source_loader = DataLoader(PairDataset(XRAY_LOW, XRAY_FULL),
                               batch_size=args.bs, shuffle=True,
                               drop_last=True, num_workers=0)
    target_loader = DataLoader(TargetDataset(args.target),
                               batch_size=args.bs, shuffle=True,
                               drop_last=True, num_workers=0)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999), eps=1e-8)

    from common import nct_test_pairs, imread01
    eval_pairs = nct_test_pairs(ref_root=args.eval_ref, inp_root=args.eval_input)

    log_path = os.path.join(out_root, 'epoch_log.csv')
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['epoch', 'mse', 'mmd', 'SSIM', 'PSNR', 'RMSE',
                                'seconds'])

    best = {'SSIM': -1.0}
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        stats_mse, stats_mmd, iters = 0.0, 0.0, 0
        for (x, y, _), (t, _n) in zip(source_loader, target_loader):
            x = x[:, None].to(device)
            y = y[:, None].to(device)
            t = t[:, None].to(device)
            if needs_wt:
                l2 = torch.stack([torch.from_numpy(wavelet_branch(xi)[0]).float()
                                  for xi in x.squeeze(1).cpu().numpy()]).unsqueeze(1).to(device)
                l3 = torch.stack([torch.from_numpy(wavelet_branch(xi)[1]).float()
                                  for xi in x.squeeze(1).cpu().numpy()]).unsqueeze(1).to(device)
            else:
                l2 = l3 = x
            optimizer.zero_grad()
            out, f_src, f_tar = net(x, t, l2, l3)
            mse = loss_fn(out, y)
            if args.mmd == 'on':
                per_ch = mmd2_rbf_rows(f_src, f_tar, args.sigma)  # [B, C]
                # original normalization: sum over 1024 channels / (B * 256)
                # generalized: sum over C channels / (B * C / 4)  ==  4 * mean_ch / B
                mmd = per_ch.sum() / (x.size(0) * per_ch.shape[1] / 4.0)
            else:
                mmd = torch.zeros((), device=device)
            loss = mse + args.lam * mmd
            loss.backward()
            optimizer.step()
            stats_mse += mse.item()
            stats_mmd += float(mmd)
            iters += 1

        m = evaluate_model(net, device, plain_input=False, pairs=eval_pairs)
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch + 1, stats_mse / max(iters, 1),
                                    stats_mmd / max(iters, 1),
                                    m['SSIM_mean'], m['PSNR_mean'], m['RMSE_mean'],
                                    round(time.time() - t0, 1)])
        print(f"[{run_name}] epoch {epoch + 1}/{args.epochs} mse={stats_mse / max(iters, 1):.5f} "
              f"mmd={stats_mmd / max(iters, 1):.6f} SSIM={m['SSIM_mean']:.4f} "
              f"PSNR={m['PSNR_mean']:.2f} RMSE={m['RMSE_mean']:.4f}", flush=True)
        if m['SSIM_mean'] > best['SSIM']:
            best = {'SSIM': m['SSIM_mean'], 'epoch': epoch + 1}
            save_torch({'state_dict': net.state_dict(), 'epoch': epoch + 1,
                        'metrics': m, 'args': vars(args)},
                       os.path.join(out_root, 'best.pth'))

    ckpt = load_torch(os.path.join(out_root, 'best.pth'), map_location=device)
    net.load_state_dict(ckpt['state_dict'])
    final = evaluate_model(net, device, plain_input=False, pairs=eval_pairs,
                           save_dir=os.path.join(out_root, 'outputs'))
    record = {'run': run_name, 'best_epoch': ckpt['epoch'], 'final': final,
              'args': vars(args)}
    append_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
                             'summary_uda.json'), record)
    print(f"[{run_name}] BEST epoch {ckpt['epoch']}: "
          f"SSIM {final['SSIM_mean']:.4f}±{final['SSIM_std']:.4f} "
          f"PSNR {final['PSNR_mean']:.2f}±{final['PSNR_std']:.2f} "
          f"RMSE {final['RMSE_mean']:.4f}±{final['RMSE_std']:.4f}", flush=True)


if __name__ == '__main__':
    main()
