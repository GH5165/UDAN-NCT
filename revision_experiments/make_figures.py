# -*- coding: utf-8 -*-
"""Generate revised qualitative figures (Fig 5-8 equivalents) from the new outputs."""
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
NCT = r'H:/6248R/CodeMATLAB/result/FDK/FDK_raw_200Views_noise1e4'


def load(path):
    return np.asarray(Image.open(path).convert('L'), dtype=float) / 255.0


def panel6(names_paths, out_png, labels):
    fig, axes = plt.subplots(2, 3, figsize=(9, 9.6))
    for ax, (lab, p) in zip(axes.flat, zip(labels, names_paths)):
        ax.imshow(load(p), cmap='gray', vmin=0, vmax=1)
        ax.set_title(lab, fontsize=13)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def roi_zoom(names_paths, out_png, labels, box=(192, 320, 192, 320)):
    fig, axes = plt.subplots(2, 3, figsize=(9, 9.6))
    for ax, (lab, p) in zip(axes.flat, zip(labels, names_paths)):
        img = load(p)
        y0, y1, x0, x1 = box
        ax.imshow(img[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1)
        ax.set_title(lab, fontsize=13)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def overview_with_box(slice_png, out_png, box=(192, 320, 192, 320)):
    img = (load(slice_png) * 255).astype(np.uint8)
    im = Image.fromarray(img).convert('L')
    d = ImageDraw.Draw(im)
    y0, y1, x0, x1 = box
    d.rectangle([x0, y0, x1, y1], outline=255, width=3)
    im.save(out_png)


def main():
    os.makedirs(os.path.join(HERE, 'new_figs'), exist_ok=True)
    out = os.path.join(HERE, 'new_figs')

    def run_img(run, name):
        return os.path.join(RUNS, run, 'outputs', name)

    for zname, f5, f6 in [('7.png', 'fig5_new.png', 'fig6_new.png')]:
        paths = [os.path.join(NCT, zname),
                 run_img('SART', zname),
                 run_img('UDAN-residual', zname),
                 run_img('DnCNN-Xray', zname),
                 run_img('FBPConvNet-Xray', zname),
                 run_img('REDCNN-Xray', zname)]
        labels = ['(a) FDK (input)', '(b) SART', '(c) UDAN (proposed)',
                  '(d) DnCNN', '(e) FBPConvNet', '(f) REDCNN']
        panel6(paths, os.path.join(out, f5), labels)

    zname = '10.png'
    box = (176, 304, 216, 344)
    paths = [os.path.join(NCT, zname),
             run_img('SART', zname),
             run_img('UDAN-residual', zname),
             run_img('DnCNN-Xray', zname),
             run_img('FBPConvNet-Xray', zname),
             run_img('REDCNN-Xray', zname)]
    labels = ['(a) FDK (input)', '(b) SART', '(c) UDAN (proposed)',
              '(d) DnCNN', '(e) FBPConvNet', '(f) REDCNN']
    panel6(paths, os.path.join(out, 'fig6_new.png'), labels)
    roi_zoom(paths, os.path.join(out, 'fig7_new.png'), labels, box=box)
    overview_with_box(os.path.join(NCT, zname), os.path.join(out, 'fig6_roi_marked.png'), box=box)

    ab_paths = [run_img('UDAN-noMMD', zname),
                run_img('UDAN-noWT', zname),
                run_img('UDAN-repro', zname)]
    # 3-panel ablation
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for ax, (lab, p) in zip(axes.flat, zip(['(a) UDAN without MMD', '(b) UDAN without WT',
                                            '(c) UDAN (full)'], ab_paths)):
        ax.imshow(load(p), cmap='gray', vmin=0, vmax=1)
        ax.set_title(lab, fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'fig8_new.png'), dpi=200)
    plt.close()
    print('new figures written to', out)


if __name__ == '__main__':
    main()
