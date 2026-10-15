# -*- coding: utf-8 -*-
"""Supervised training for the comparison networks (direct-transfer experiments).

Trains DnCNN / REDCNN / FBPConvNet on paired X-ray CT data (low-dose input ->
normal-dose label, WT_SR full_dose/low_dose test1, 122 pairs), exactly following
the original protocol (Adam, lr=1e-5, betas 0.9/0.999, batch 2, MSE, 100 epochs),
then applies the trained network directly to NCT (direct transfer baseline).

After every epoch the network is evaluated on the NCT test set under the unified
protocol; the best checkpoint by SSIM is kept (same selection rule as the
original scripts).

Usage:
  python train_supervised.py --net dncnn --name DnCNN-Xray
  python train_supervised.py --net redcnn --name REDCNN-Xray
  python train_supervised.py --net fbpconvnet --name FBPConvNet-Xray
  python train_supervised.py --net udan_sup --train_inputs <NCT noisy> --train_labels <NCT clean>
"""
import argparse
import csv
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'H:/ZCode/NCT返修/code/Compare_lowdose')
sys.path.insert(0, r'H:/ZCode/NCT返修/code/lowdose_MMD1')

from common import (PairDataset, XRAY_FULL, XRAY_LOW, NCT_LOW, NCT_REF,
                    evaluate_model, append_json, wavelet_branch, save_torch, load_torch,
                    nct_test_pairs)
from net.DnCNN import DnCNN
from net.RED_CNN.networks import RED_CNN
from net.FBPConvNet.model import FBPCONVNet
import model3  # UDAN architecture (lowdose_MMD1)


def build_net(name):
    if name == 'dncnn':
        return DnCNN(channels=1, features=64, num_of_layers=17)
    if name == 'redcnn':
        return RED_CNN(out_ch=96)
    if name == 'fbpconvnet':
        return FBPCONVNet()
    if name == 'udan_sup':
        return model3.UNet(n_channels=1, n_classes=1)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True,
                    choices=['dncnn', 'redcnn', 'fbpconvnet', 'udan_sup'])
    ap.add_argument('--name', default=None, help='run name (default: NET-Xray)')
    ap.add_argument('--train_inputs', default=XRAY_LOW)
    ap.add_argument('--train_labels', default=XRAY_FULL)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--eval_input', default=NCT_LOW)
    ap.add_argument('--eval_ref', default=NCT_REF)
    ap.add_argument('--seed', type=int, default=2024)
    args = ap.parse_args()

    run_name = args.name or f'{args.net}-Xray'
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', run_name)
    os.makedirs(out_root, exist_ok=True)

    torch.manual_seed(args.seed)
    np_seed = args.seed
    import numpy as np
    np.random.seed(np_seed)
    torch.backends.cudnn.benchmark = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = build_net(args.net).to(device)
    plain_input = args.net != 'udan_sup'

    train_loader = DataLoader(
        PairDataset(args.train_inputs, args.train_labels),
        batch_size=args.bs, shuffle=True, drop_last=True, num_workers=0)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999), eps=1e-8)

    log_path = os.path.join(out_root, 'epoch_log.csv')
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(['epoch', 'train_mse', 'SSIM', 'PSNR', 'RMSE', 'seconds'])

    best = {'SSIM': -1.0}
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        total, count = 0.0, 0
        for x, y, _ in train_loader:
            x = x[:, None].to(device)
            y = y[:, None].to(device)
            optimizer.zero_grad()
            if plain_input:
                out = net(x)
            else:
                l2 = torch.stack([torch.from_numpy(wavelet_branch(xi)[0]).float()
                                  for xi in x.squeeze(1).cpu().numpy()]).unsqueeze(1).to(device)
                l3 = torch.stack([torch.from_numpy(wavelet_branch(xi)[1]).float()
                                  for xi in x.squeeze(1).cpu().numpy()]).unsqueeze(1).to(device)
                out = net(x, x, l2, l3)[0]
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
            count += 1
        eval_pairs = nct_test_pairs(ref_root=args.eval_ref, inp_root=args.eval_input)
        m = evaluate_model(net, device, plain_input=plain_input, pairs=eval_pairs)
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch + 1, total / max(count, 1),
                                    m['SSIM_mean'], m['PSNR_mean'], m['RMSE_mean'],
                                    round(time.time() - t0, 1)])
        print(f"[{run_name}] epoch {epoch + 1}/{args.epochs} mse={total / max(count, 1):.5f} "
              f"SSIM={m['SSIM_mean']:.4f} PSNR={m['PSNR_mean']:.2f} RMSE={m['RMSE_mean']:.4f}",
              flush=True)
        if m['SSIM_mean'] > best['SSIM']:
            best = {'SSIM': m['SSIM_mean'], 'epoch': epoch + 1, **m}
            save_torch({'state_dict': net.state_dict(), 'epoch': epoch + 1,
                       'metrics': m, 'args': vars(args)},
                      os.path.join(out_root, 'best.pth'))

    # Final evaluation of the best checkpoint + save output images
    ckpt = load_torch(os.path.join(out_root, 'best.pth'), map_location=device)
    net.load_state_dict(ckpt['state_dict'])
    final = evaluate_model(net, device, plain_input=plain_input, pairs=eval_pairs,
                           save_dir=os.path.join(out_root, 'outputs'))
    record = {'run': run_name, 'best_epoch': ckpt['epoch'], 'final': final}
    append_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
                             'summary_supervised.json'), record)
    print(f"[{run_name}] BEST epoch {ckpt['epoch']}: "
          f"SSIM {final['SSIM_mean']:.4f}±{final['SSIM_std']:.4f} "
          f"PSNR {final['PSNR_mean']:.2f}±{final['PSNR_std']:.2f} "
          f"RMSE {final['RMSE_mean']:.4f}±{final['RMSE_std']:.4f}", flush=True)


if __name__ == '__main__':
    main()
