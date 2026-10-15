# -*- coding: utf-8 -*-
"""Network variants for the revision experiments.

Faithful re-implementations of the original project models
(H:/ZCode/NCT返修/code/lowdose_MMD1) with configurable MMD attachment point:

  UDAN       : model3.UNet  (WT/DnCNN branches + MMD after down4)   [paper's UDAN]
  UDAN NoWT  : model5 intent (all skip connections, no WT/DnCNN)    [ablation]
  UDAN noMMD : same as UDAN but loss ignores MMD                    [ablation, model6]
  DnCNN_MMD  : DnCNN backbone + MMD on mid-level features (pooled to 32x32)
  REDCNN_MMD : RED-CNN backbone + MMD on bottleneck features (pooled to 32x32)

All MMD-capable nets expose forward(x_src, x_tar, LL2=None, LL3=None)
-> (output, feat_src, feat_tar). Denoising paths keep full resolution;
only the MMD branch is pooled.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, r'H:/ZCode/NCT返修/code/lowdose_MMD1')
from unet_parts import DoubleConv, Down, Up, OutConv  # noqa: E402
from unet_parts import DnCNN as WT_DnCNN  # 3-layer, non-residual (WT branch)  # noqa: E402

sys.path.insert(0, r'H:/ZCode/NCT返修/code/Compare_lowdose')
from net.RED_CNN.networks import RED_CNN  # noqa: E402
from net.DnCNN import DnCNN as PlainDnCNN  # 17-layer residual denoiser  # noqa: E402


class UDAN(nn.Module):
    """model3.UNet with configurable MMD feature location ('down4' or 'down3')."""

    def __init__(self, n_channels=1, n_classes=1, mmd_loc='down4'):
        super().__init__()
        assert mmd_loc in ('down4', 'down3')
        self.mmd_loc = mmd_loc
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear=False)
        self.up2 = Up(512, 256, bilinear=False)
        self.up3 = Up(256, 128, bilinear=False)
        self.up4 = Up(128, 64, bilinear=False)
        self.outc = OutConv(64, n_classes)
        self.tranf4 = WT_DnCNN(1, 512, 3)  # WT branch, level-3 LL (64x64 -> 512ch)
        self.tranf3 = WT_DnCNN(1, 256, 3)  # WT branch, level-2 LL (128x128 -> 256ch)

    def forward(self, x_src, x_tar, LL2=None, LL3=None):
        x1_src = self.inc(x_src)
        x2_src = self.down1(x1_src)
        x3 = self.down2(x2_src)
        x4 = self.down3(x3)
        x5_src = self.down4(x4)

        x1_tar = self.inc(x_tar)
        x2_tar = self.down1(x1_tar)
        x3_tar = self.down2(x2_tar)
        x4_tar = self.down3(x3_tar)
        x5_tar = self.down4(x4_tar)

        if self.mmd_loc == 'down4':
            feat_src, feat_tar = x5_src, x5_tar
        else:
            feat_src, feat_tar = x4, x4_tar

        x5_transf = self.tranf4(LL3)
        x4_transf = self.tranf3(LL2)
        d = self.up1(x5_src, x5_transf)
        d = self.up2(d, x4_transf)
        d = self.up3(d, x2_src)
        d = self.up4(d, x1_src)
        return self.outc(d), feat_src, feat_tar


class UDANNoWT(nn.Module):
    """model5 intent: standard skip connections at all levels, no WT/DnCNN branch."""

    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear=False)
        self.up2 = Up(512, 256, bilinear=False)
        self.up3 = Up(256, 128, bilinear=False)
        self.up4 = Up(128, 64, bilinear=False)
        self.outc = OutConv(64, n_classes)

    def forward(self, x_src, x_tar, LL2=None, LL3=None):
        x1_src = self.inc(x_src)
        x2_src = self.down1(x1_src)
        x3_src = self.down2(x2_src)
        x4_src = self.down3(x3_src)
        x5_src = self.down4(x4_src)

        x1_tar = self.inc(x_tar)
        x2_tar = self.down1(x1_tar)
        x3_tar = self.down2(x2_tar)
        x4_tar = self.down3(x3_tar)
        x5_tar = self.down4(x4_tar)

        d = self.up1(x5_src, x4_src)
        d = self.up2(d, x3_src)
        d = self.up3(d, x2_src)
        d = self.up4(d, x1_src)
        return self.outc(d), x5_src, x5_tar


class DnCNNMMD(nn.Module):
    """17-layer DnCNN (residual) + MMD on mid-level features, pooled to 32x32."""

    def __init__(self, features=64, num_of_layers=17, split_conv=9):
        super().__init__()
        full = PlainDnCNN(1, features, num_of_layers).dncnn  # Sequential
        # layers: [conv, relu] + (num-2)*[conv, bn, relu] + [conv]
        head_idx = 2 + 3 * (split_conv - 1) + 1  # include relu of last head conv
        self.head = nn.Sequential(*list(full.children())[:head_idx])
        self.tail = nn.Sequential(*list(full.children())[head_idx:])
        self.pool = nn.AdaptiveAvgPool2d((32, 32))

    def forward(self, x_src, x_tar, LL2=None, LL3=None):
        h_src = self.head(x_src)          # [B, 64, H, W] full resolution
        out = x_src - self.tail(h_src)    # residual denoising, full resolution
        h_tar = self.head(x_tar)
        return out, self.pool(h_src), self.pool(h_tar)


class REDCNNMMD(nn.Module):
    """RED-CNN backbone + MMD on bottleneck features, pooled to 32x32."""

    def __init__(self, out_ch=96):
        super().__init__()
        self.base = RED_CNN(out_ch=out_ch)
        self.pool = nn.AdaptiveAvgPool2d((32, 32))

    def _encode(self, x):
        b = self.base
        out = b.relu(b.conv1(x))
        r2 = b.relu(b.conv2(out))
        out = b.relu(b.conv3(r2))
        out = b.relu(b.conv4(out))
        bottleneck = b.relu(b.conv5(out))
        return r2, bottleneck

    def _decode(self, bottleneck, r2, x):
        b = self.base
        out = b.tconv1(bottleneck)
        out = b.tconv2(b.relu(out))
        out = b.tconv3(b.relu(out))
        out = out + r2
        out = b.tconv4(b.relu(out))
        out = b.tconv5(b.relu(out))
        out = out + x
        return b.relu(out)

    def forward(self, x_src, x_tar, LL2=None, LL3=None):
        r2_src, b_src = self._encode(x_src)
        out_src = self._decode(b_src, r2_src, x_src)
        _, b_tar = self._encode(x_tar)
        return out_src, self.pool(b_src), self.pool(b_tar)


class UDANRes(nn.Module):
    """UDAN with residual output anchoring: output = x_src + decoder(x_src).

    Same encoder/WT branches as UDAN; only the output head adds the input, so
    the reconstruction inherits the target-domain photometric scale at test
    time (addresses the intensity-calibration drift of the direct decoder).
    """

    def __init__(self, n_channels=1, n_classes=1, mmd_loc='down4'):
        super().__init__()
        self.base = UDAN(n_channels, n_classes, mmd_loc)

    def forward(self, x_src, x_tar, LL2=None, LL3=None):
        out, f_src, f_tar = self.base(x_src, x_tar, LL2, LL3)
        return x_src + out, f_src, f_tar


class UDANResNoWT(nn.Module):
    """NoWT variant with residual output anchoring (for the revised ablation table)."""

    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.base = UDANNoWT(n_channels, n_classes)

    def forward(self, x_src, x_tar, LL2=None, LL3=None):
        out, f_src, f_tar = self.base(x_src, x_tar, LL2, LL3)
        return x_src + out, f_src, f_tar
