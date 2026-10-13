# -*- coding: utf-8 -*-
"""Unified data loading & evaluation protocol for the NCT revision experiments.

Protocol (documented in the revised manuscript):
  - All images loaded as grayscale float in [0, 1] (PNG/255).
  - Reference: H:/6248R/CodeMATLAB/result/FDK/FDK_raw_200Views (normal-flux FDK, 31 slices)
  - Input:     H:/6248R/CodeMATLAB/result/FDK/FDK_raw_200Views_noise1e4 (low-flux, 32 slices)
  - Pairs matched by filename; slices without a reference are skipped.
  - Metrics: SSIM(data_range=1), PSNR(data_range=1), RMSE = sqrt(MSE), mean +/- SD over slices.
"""
import glob
import json
import os

import numpy as np
import pywt
import torch
from PIL import Image
from math import sqrt
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset

WAVENAME = 'bior1.3'

import os
XRAY_FULL = os.environ.get("UDAN_XRAY_FULL", r"H:/6248R/pycharmProject/pytorch/WT_SR/data/full_dose/test1")
XRAY_LOW = os.environ.get("UDAN_XRAY_LOW", r"H:/6248R/pycharmProject/pytorch/WT_SR/data/low_dose/test1")
NCT_REF = os.environ.get("UDAN_NCT_REF", r"H:/6248R/CodeMATLAB/result/FDK/FDK_raw_200Views")
NCT_LOW = os.environ.get("UDAN_NCT_LOW", r"H:/6248R/CodeMATLAB/result/FDK/FDK_raw_200Views_noise1e4")
NCT_TRAIN_LOW = os.environ.get("UDAN_NCT_TRAIN_LOW", r"H:/6248R/CodeMATLAB/result/FDK/FDK_raw3_noise_1e4")
NCT_TRAIN_CLEAN = os.environ.get("UDAN_NCT_TRAIN_CLEAN", r"H:/6248R/CodeMATLAB/result/FDK/FDK_raw3")
NCT_LOW_1E3 = os.environ.get("UDAN_NCT_LOW_1E3", r"H:/6248R/CodeMATLAB/result/FDK/FDK_raw_200Views_noise1e3")


def imread01(path):
    return np.asarray(Image.open(path).convert('L'), dtype=np.float64) / 255.0


def wavelet_branch(img):
    """LL2/LL3 branches exactly as in the original SourceImageDataset (3-level dwt, cropped)."""
    LL1, _ = pywt.dwt2(img, WAVENAME)
    LL2, _ = pywt.dwt2(LL1, WAVENAME)
    LL3, _ = pywt.dwt2(LL2, WAVENAME)
    return LL2[0:128, 0:128], LL3[0:64, 0:64]


class PairDataset(Dataset):
    """Supervised pair dataset: (input, label) float [0,1] tensors + filename."""

    def __init__(self, inputs_root, labels_root):
        self.files_input = sorted(glob.glob(os.path.join(inputs_root, '*.png')))
        self.files_label = sorted(glob.glob(os.path.join(labels_root, '*.png')))
        if len(self.files_input) != len(self.files_label):
            print(f"[PairDataset] WARNING: {len(self.files_input)} inputs vs "
                  f"{len(self.files_label)} labels in {inputs_root}")

    def __len__(self):
        return min(len(self.files_input), len(self.files_label))

    def __getitem__(self, index):
        low = imread01(self.files_input[index % len(self.files_input)])
        full = imread01(self.files_label[index % len(self.files_label)])
        name = os.path.basename(self.files_input[index % len(self.files_input)])
        return torch.from_numpy(low).float(), torch.from_numpy(full).float(), name


class TargetDataset(Dataset):
    """Unlabeled target-domain images (for MMD adaptation)."""

    def __init__(self, inputs_root):
        self.files = sorted(glob.glob(os.path.join(inputs_root, '*.png')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = imread01(self.files[index % len(self.files)])
        name = os.path.basename(self.files[index % len(self.files)])
        return torch.from_numpy(img).float(), name


def nct_test_pairs(ref_root=NCT_REF, inp_root=NCT_LOW):
    """Return list of (input_path, reference_path, name) matched by filename."""
    refs = {os.path.basename(p): p for p in glob.glob(os.path.join(ref_root, '*.png'))}
    pairs = []
    for p in sorted(glob.glob(os.path.join(inp_root, '*.png'))):
        name = os.path.basename(p)
        if name in refs:
            pairs.append((p, refs[name], name))
    return pairs


def metrics(ref, img):
    """Unified metrics on [0,1] float arrays."""
    return (float(ssim(ref, img, data_range=1)),
            float(psnr(ref, img, data_range=1)),
            float(sqrt(mse(ref, img))))


CROP_MARGIN = 32  # unified protocol v2: central 448x448 crop (removes FDK periphery errors)


def crop_center(a, margin=CROP_MARGIN):
    return a[margin:-margin, margin:-margin] if margin > 0 else a


def summarize(vals):
    a = np.asarray(vals)
    return float(a.mean()), float(a.std())


def evaluate_model(net, device='cuda', plain_input=True, pairs=None, save_dir=None,
                   crop=0):
    """Evaluate a trained network on the NCT test set.

    plain_input=True  -> net(x)
    plain_input=False -> net(x, x, LL2, LL3)  (UDAN-style, returns (out, s_feat, t_feat))
    crop              >0: metrics on central crop only (protocol v2, final tables);
                         crop=0: full frame (used uniformly for checkpoint selection).
    """
    was_training = net.training
    net.eval()
    if pairs is None:
        pairs = nct_test_pairs()
    s_l, p_l, r_l = [], [], []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for ip, rp, name in pairs:
            low = imread01(ip)
            x = torch.from_numpy(low).float()[None, None].to(device)
            if plain_input:
                out = net(x)
            else:
                LL2, LL3 = wavelet_branch(low)
                l2 = torch.from_numpy(LL2).float()[None, None].to(device)
                l3 = torch.from_numpy(LL3).float()[None, None].to(device)
                out, _, _ = net(x, x, l2, l3)
            o = out.squeeze().cpu().numpy().astype(np.float64)
            o = np.clip(o, 0.0, 1.0)
            ref = imread01(rp)
            s, p, r = metrics(crop_center(ref, crop), crop_center(o, crop))
            s_l.append(s)
            p_l.append(p)
            r_l.append(r)
            if save_dir:
                Image.fromarray((o * 255).round().astype(np.uint8)).save(
                    os.path.join(save_dir, name))
    if was_training:
        net.train()
    return {
        'n': len(s_l),
        'SSIM_mean': summarize(s_l)[0], 'SSIM_std': summarize(s_l)[1],
        'PSNR_mean': summarize(p_l)[0], 'PSNR_std': summarize(p_l)[1],
        'RMSE_mean': summarize(r_l)[0], 'RMSE_std': summarize(r_l)[1],
    }


def append_json(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    records = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []
    records.append(record)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def save_torch(obj, path):
    """torch.save via file handle: PyTorch's native writer rejects non-ASCII paths on Windows."""
    with open(path, 'wb') as f:
        torch.save(obj, f)


def load_torch(path, map_location='cpu'):
    with open(path, 'rb') as f:
        return torch.load(f, map_location=map_location)
