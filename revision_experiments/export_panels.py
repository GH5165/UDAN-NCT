# -*- coding: utf-8 -*-
"""Export every individual figure panel (single images, no composition) to
H:/ZCode/NCT返修/figures for the author to assemble.

Naming: <fig>_<panel>_<method>_<slice>.png  (+ _roi versions for Figure 7)
"""
import os
import glob
import shutil

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
FDK = 'H:/6248R/CodeMATLAB/result/FDK'
OUT = r'H:/ZCode/NCT返修/figures'
SLICES = {'simple': '15.png', 'complex': '25.png'}
ROI = (176, 336, 176, 336)  # y0,y1,x0,x1

# (figure tag, method tag, source kind, path template)
SOURCES = [
    ('fig3', 'reference', 'file', FDK + '/FDK_raw_200Views/{slice}'),
    ('fig3', 'input1e3', 'file', FDK + '/FDK_raw_200Views_noise1e3/{slice}'),
    ('fig3', 'input3e2', 'file', FDK + '/FDK_raw_200Views_noise3e2/{slice}'),
    ('fig5', 'reference', 'file', FDK + '/FDK_raw_200Views/{slice}'),
    ('fig5', 'input', 'file', FDK + '/FDK_raw_200Views_noise1e3/{slice}'),
    ('fig5', 'SART', 'run_outputs', 'SART1e3'),
    ('fig5', 'FBPConvNet_direct', 'run_outputs_1e3', 'FBPConvNet-Xray'),
    ('fig5', 'FBPConvNet_MMD', 'run_outputs_1e3', 'FBPConvNet-MMD1e3'),
    ('fig5', 'DnCNN_MMD', 'run_outputs_1e3', 'DnCNN-MMD1e3'),
    ('fig5', 'REDCNN_MMD', 'run_outputs_1e3', 'REDCNN-MMD1e3'),
    ('fig5', 'UDAN', 'run_outputs_1e3', 'U1e_d95'),
    ('fig5', 'UDAN_sup', 'run_outputs_1e3', 'UDAN-supNCT1e3'),
    ('fig5', 'FBPConvNet_sup', 'run_outputs_1e3', 'FBPConvNet-NCT1e3'),
    ('fig5', 'DnCNN_sup', 'run_outputs_1e3', 'DnCNN-NCT1e3'),
    ('fig5', 'REDCNN_sup', 'run_outputs_1e3', 'REDCNN-NCT1e3'),
    ('fig8', 'noMMD', 'run_outputs_1e3', 'UDANRes-noMMD'),
    ('fig8', 'noWT', 'run_outputs_1e3', 'UDANRes-noWT1e3'),
]


def src_path(kind, arg, sl):
    if kind == 'file':
        return arg.replace('{slice}', sl)
    d = 'outputs' if kind == 'run_outputs' else 'outputs_1e3'
    return os.path.join(RUNS, arg, d, sl)


def main():
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for tag, meth, kind, arg in SOURCES:
        for sl_tag, sl in SLICES.items():
            p = src_path(kind, arg, sl)
            if not os.path.exists(p):
                print('missing:', p)
                continue
            dst = os.path.join(OUT, '%s_%s_%s.png' % (tag, meth, sl_tag))
            shutil.copy(p, dst)
            n += 1
            if tag == 'fig5':  # ROI crops for Figure 7
                img = Image.open(p)
                y0, y1, x0, x1 = ROI
                img.crop((x0, y0, x1, y1)).save(
                    os.path.join(OUT, 'fig7_%s_%s.png' % (meth, sl_tag)))
                n += 1
    # Figure 9 panels: split the PCA figure into left/right halves
    f9 = os.path.join(HERE, 'feature_analysis_1e3.png')
    if os.path.exists(f9):
        img = Image.open(f9)
        w, h = img.size
        img.crop((0, 0, w // 2, h)).save(os.path.join(OUT, 'fig9_withoutMMD.png'))
        img.crop((w // 2, 0, w, h)).save(os.path.join(OUT, 'fig9_withMMD_UDAN.png'))
        n += 2
    print('exported', n, 'panels ->', OUT)


if __name__ == '__main__':
    main()
