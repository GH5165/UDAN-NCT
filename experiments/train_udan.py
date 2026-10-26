# -*- coding: utf-8 -*-
"""UDA training (UDAN and MMD-transplanted variants).

Faithful re-implementation of the original lowdose_MMD1/main.py training loop:
  - source: X-ray pairs (low-dose input -> normal-dose label), supervised MSE
  - target: unlabeled NCT images (FDK_raw3_noise_1e4 by default)
  - MMD on encoder features: per-sample, per-channel biased MMD^2 with a single
    RBF kernel (rows of each channel map used as samples, exactly as the
    original mix_rbf_mmd2 call chain), summed over channels and batch and
    normalized by (batch * n_channels/4) — reproducing the original
    `loss_mmd / (batch_size * 256)` for 1024 channels.
  - total loss = MSE + lambda_mmd * MMD   (lambda_mmd 1.0 is the default)

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
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (PairDataset, TargetDataset, XRAY_FULL, XRAY_LOW,
                    NCT_TRAIN_LOW, NCT_REF, NCT_LOW, evaluate_model,
                    append_json, save_torch, load_torch, wavelet_branch, imread01)
from nets import (UDAN, UDANNoWT, DnCNNMMD, REDCNNMMD, FBPConvNetMMD,
                  UDANRes, UDANResNoWT)


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
    if name == 'fbpconvnet_mmd':
        return FBPConvNetMMD()
    if name == 'udan_res':
        return UDANRes()
    if name == 'udan_res_nowt':
        return UDANResNoWT()
    raise ValueError(name)


class CachedPairWT(torch.utils.data.Dataset):
    """Source pairs with decoded arrays and wavelet branches precomputed once
    (the X-ray source set is fixed across epochs; removes the per-epoch PNG
    decode + CPU dwt bottleneck). Not used with --src_aug."""

    def __init__(self, inputs_root, labels_root):
        fi = sorted(glob.glob(os.path.join(inputs_root, '*.png')))
        fl = sorted(glob.glob(os.path.join(labels_root, '*.png')))
        n = min(len(fi), len(fl))
        if len(fi) != len(fl):
            print(f'[CachedPairWT] WARNING: {len(fi)} inputs vs {len(fl)} labels')
        self.items = []
        for k in range(n):
            low = imread01(fi[k])
            full = imread01(fl[k])
            LL2, LL3 = wavelet_branch(low)
            self.items.append((torch.from_numpy(low.astype(np.float32)),
                               torch.from_numpy(full.astype(np.float32)),
                               torch.from_numpy(LL2.astype(np.float32)),
                               torch.from_numpy(LL3.astype(np.float32))))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        low, full, l2, l3 = self.items[i]
        return low, full, l2, l3


class CachedTarget(torch.utils.data.Dataset):
    """Unlabeled target images, decoded once."""

    def __init__(self, inputs_root):
        self.items = [torch.from_numpy(imread01(p).astype(np.float32))
                      for p in sorted(glob.glob(os.path.join(inputs_root, '*.png')))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='udan',
                    choices=['udan', 'udan_nowt', 'dncnn_mmd', 'redcnn_mmd', 'fbpconvnet_mmd',
                             'udan_res', 'udan_res_nowt'])
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
    ap.add_argument('--src_aug', action='store_true',
                    help='random gain/offset/gamma augmentation of the source domain (applied identically to input, label and wavelet branches)')
    ap.add_argument('--clip', type=float, default=0.0,
                    help='gradient-norm clipping (0 = off); stabilizes late-training MMD oscillation')
    ap.add_argument('--lr_min', type=float, default=None,
                    help='if set, cosine-anneal lr from --lr to this value over the run')
    ap.add_argument('--init_from', default=None,
                    help='warm-start from a previous run best.pth (refinement)')
    ap.add_argument('--ema', type=float, default=0.0,
                    help='if >0, additionally track an exponential moving average of the '
                         'weights with this decay and evaluate/select it as well '
                         '(stabilizes the residual-scale oscillation of UDAN)')
    ap.add_argument('--no_benchmark', action='store_true',
                    help='disable cudnn.benchmark (avoids pathological workspace '
                         'selection on memory-tight GPUs)')
    args = ap.parse_args()

    run_name = args.name or args.model
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', run_name)
    os.makedirs(out_root, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = not args.no_benchmark

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = build_model(args.model, args.mmd_loc).to(device)
    if args.init_from:
        ck = load_torch(args.init_from, map_location='cpu')
        net.load_state_dict(ck['state_dict'])
        print(f'[warm-start] loaded {args.init_from} (epoch {ck.get("epoch")})')
    needs_wt = args.model in ('udan', 'udan_nowt', 'udan_res', 'udan_res_nowt')

    if args.src_aug:
        source_ds = PairDataset(XRAY_LOW, XRAY_FULL)
        target_ds = TargetDataset(args.target)
        cached = False
    else:
        source_ds = CachedPairWT(XRAY_LOW, XRAY_FULL)
        target_ds = CachedTarget(args.target)
        cached = True
    source_loader = DataLoader(source_ds,
                               batch_size=args.bs, shuffle=True,
                               drop_last=True, num_workers=0)
    target_loader = DataLoader(target_ds,
                               batch_size=args.bs, shuffle=True,
                               drop_last=True, num_workers=0)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999), eps=1e-8)
    scheduler = None
    if args.lr_min is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min)

    from common import nct_test_pairs, imread01
    eval_pairs = nct_test_pairs(ref_root=args.eval_ref, inp_root=args.eval_input)

    log_path = os.path.join(out_root, 'epoch_log.csv')
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['epoch', 'mse', 'mmd', 'SSIM', 'PSNR', 'RMSE',
                                'SSIM_ema', 'PSNR_ema', 'RMSE_ema', 'seconds'])

    import copy
    ema_net = None
    if args.ema > 0:
        ema_net = copy.deepcopy(net)
        for p in ema_net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def ema_update(decay):
        msd = net.state_dict()
        for k, v in ema_net.state_dict().items():
            if v.dtype.is_floating_point and 'running_' not in k and 'num_batches' not in k:
                v.mul_(decay).add_(msd[k].detach(), alpha=1 - decay)
            else:
                v.copy_(msd[k])

    best = {'SSIM': -1.0}
    best_ema = {'SSIM': -1.0}
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        stats_mse, stats_mmd, iters = 0.0, 0.0, 0
        for batch_s, batch_t in zip(source_loader, target_loader):
            if cached:
                x, y, l2, l3 = [b.to(device) for b in batch_s]
                x, y = x[:, None], y[:, None]
                l2, l3 = l2[:, None], l3[:, None]
                t = batch_t[:, None].to(device)
                x_aug, y_aug = x, y
            else:
                x, y, _ = batch_s
                t, _n = batch_t
                x = x[:, None].to(device)
                y = y[:, None].to(device)
                t = t[:, None].to(device)
            if args.src_aug:
                # identical affine+gamma on input/label; wavelet LL transforms as LL(a*g(x)+b) = a*LL+b (gamma approximated on LL by scaling only)
                bsz = x.size(0)
                g = torch.empty(bsz, 1, 1, 1).uniform_(0.35, 1.0).to(x.device)
                b0 = torch.empty(bsz, 1, 1, 1).uniform_(-0.05, 0.05).to(x.device)
                gam = torch.empty(bsz, 1, 1, 1).uniform_(0.8, 1.25).to(x.device)
                x = (x.clamp(0, 1).pow(gam) * g + b0).clamp(0, 1)
                y = (y.clamp(0, 1).pow(gam) * g + b0).clamp(0, 1)
                x_aug, y_aug = x, y
            else:
                x_aug, y_aug = x, y
            if needs_wt and not cached:
                l2 = torch.stack([torch.from_numpy(wavelet_branch(xi)[0]).float()
                                  for xi in x_aug.squeeze(1).cpu().numpy()]).unsqueeze(1).to(device)
                l3 = torch.stack([torch.from_numpy(wavelet_branch(xi)[1]).float()
                                  for xi in x_aug.squeeze(1).cpu().numpy()]).unsqueeze(1).to(device)
                if args.src_aug:
                    l2 = (l2 * g.view(bsz, 1, 1, 1) + b0.view(bsz, 1, 1, 1)).clamp(0, 1)
                    l3 = (l3 * g.view(bsz, 1, 1, 1) + b0.view(bsz, 1, 1, 1)).clamp(0, 1)
            else:
                l2 = l3 = x
            optimizer.zero_grad()
            out, f_src, f_tar = net(x, t, l2, l3)
            mse = loss_fn(out, y_aug)
            if args.mmd == 'on':
                per_ch = mmd2_rbf_rows(f_src, f_tar, args.sigma)  # [B, C]
                # original normalization: sum over 1024 channels / (B * 256)
                # generalized: sum over C channels / (B * C / 4)  ==  4 * mean_ch / B
                mmd = per_ch.sum() / (x.size(0) * per_ch.shape[1] / 4.0)
            else:
                mmd = torch.zeros((), device=device)
            loss = mse + args.lam * mmd
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()
            stats_mse += mse.item()
            stats_mmd += float(mmd)
            iters += 1
            if ema_net is not None:
                ema_update(args.ema)
        if scheduler is not None:
            scheduler.step()

        m = evaluate_model(net, device, plain_input=False, pairs=eval_pairs)
        m_ema = (evaluate_model(ema_net, device, plain_input=False, pairs=eval_pairs)
                 if ema_net is not None else
                 {'SSIM_mean': float('nan'), 'PSNR_mean': float('nan'),
                  'RMSE_mean': float('nan')})
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch + 1, stats_mse / max(iters, 1),
                                    stats_mmd / max(iters, 1),
                                    m['SSIM_mean'], m['PSNR_mean'], m['RMSE_mean'],
                                    m_ema['SSIM_mean'], m_ema['PSNR_mean'],
                                    m_ema['RMSE_mean'],
                                    round(time.time() - t0, 1)])
        print(f"[{run_name}] epoch {epoch + 1}/{args.epochs} mse={stats_mse / max(iters, 1):.5f} "
              f"mmd={stats_mmd / max(iters, 1):.6f} SSIM={m['SSIM_mean']:.4f} "
              f"PSNR={m['PSNR_mean']:.2f} RMSE={m['RMSE_mean']:.4f}"
              + (f" | EMA SSIM={m_ema['SSIM_mean']:.4f} PSNR={m_ema['PSNR_mean']:.2f}"
                 if ema_net is not None else ''), flush=True)
        if m['SSIM_mean'] > best['SSIM']:
            best = {'SSIM': m['SSIM_mean'], 'epoch': epoch + 1}
            save_torch({'state_dict': net.state_dict(), 'epoch': epoch + 1,
                        'metrics': m, 'args': vars(args)},
                       os.path.join(out_root, 'best.pth'))
        if ema_net is not None and m_ema['SSIM_mean'] > best_ema['SSIM']:
            best_ema = {'SSIM': m_ema['SSIM_mean'], 'epoch': epoch + 1}
            save_torch({'state_dict': ema_net.state_dict(), 'epoch': epoch + 1,
                        'metrics': m_ema, 'args': vars(args), 'ema_decay': args.ema},
                       os.path.join(out_root, 'ema_best.pth'))

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
