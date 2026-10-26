# -*- coding: utf-8 -*-
"""Redraw manuscript figures with simple=20.png / complex=31.png.
- brightness normalization (light_norm recipe) for every panel
- subfigures saved to <hidden-local-path>
- composites (with (a),(b),... labels) saved to qoder_fig/composites
- Figure 6 panels carry the red ROI box from pic_deal.py (i=220,j=150,h=120,w=220);
  Figure 7 = zoomed crops of the same ROI.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
FDK = ''  # TODO: set local path
OUT = ''  # TODO: set local path
COMP = os.path.join(OUT, 'composites')
os.makedirs(COMP, exist_ok=True)
SLICES = {'simple': '20.png', 'complex': '31.png'}
ROI = (150, 220, 370, 340)  # left, upper, right, lower (pic_deal i=220,j=150,h=120,w=220)

METHODS8 = [
    ('reference', 'file', FDK + '/FDK_raw_200Views/{s}'),
    ('input', 'file', FDK + '/FDK_raw_200Views_noise1e3/{s}'),
    ('SART', 'outputs', 'SART1e3'),
    ('FBPConvNet_direct', 'outputs_1e3', 'FBPConvNet-Xray'),
    ('DnCNN_MMD', 'outputs_1e3', 'DnCNN-MMD1e3'),
    ('UDAN', 'outputs_1e3', 'U1e_d95'),
    ('UDAN_sup', 'outputs_1e3', 'UDAN-supNCT1e3'),
    ('FBPConvNet_sup', 'outputs_1e3', 'FBPConvNet-NCT1e3'),
]
FIG8 = [
    ('input', 'file', FDK + '/FDK_raw_200Views_noise1e3/{s}'),
    ('noMMD', 'outputs_1e3', 'UDANRes-noMMD'),
    ('noWT', 'outputs_1e3', 'UDANRes-noWT1e3'),
    ('UDAN', 'outputs_1e3', 'U1e_d95'),
]


def src(kind, arg, sl):
    if kind == 'file':
        return arg.replace('{s}', sl)
    return os.path.join(RUNS, arg, kind, sl)


def aug(a):
    a = a.astype(np.float64)
    hi, lo = np.percentile(a, 99), np.percentile(a, 1)
    a = np.clip(a, lo, hi)
    if hi > lo:
        a = (a - lo) / (hi - lo)
    return 255 * 0.1 + a * 255 * 0.8


def load_panel(kind, arg, sl):
    img = Image.open(src(kind, arg, sl)).convert('L')
    return Image.fromarray(np.clip(np.round(aug(np.asarray(img))), 0, 255).astype(np.uint8))


def font(sz=30):
    try:
        return ImageFont.truetype('arial.ttf', sz)
    except OSError:
        return ImageFont.load_default()


def compose(panels, ncol, name, box=False, zoom=False):
    if zoom:
        panels = [p.crop(ROI).resize((660, 360), Image.LANCZOS) for p in panels]
    if box:
        panels = [p.convert('RGB') for p in panels]
        for p in panels:
            d = ImageDraw.Draw(p)
            d.rectangle([ROI[0], ROI[1], ROI[2], ROI[3]], outline=(255, 0, 0), width=3)
    w, h = panels[0].size
    lab_h = 40
    nrow = (len(panels) + ncol - 1) // ncol
    mode = 'RGB' if box else 'L'
    canvas = Image.new(mode, (ncol * w, nrow * (h + lab_h)), (255, 255, 255) if box else 255)
    d = ImageDraw.Draw(canvas)
    for k, p in enumerate(panels):
        x, y = (k % ncol) * w, (k // ncol) * (h + lab_h)
        canvas.paste(p, (x, y + lab_h))
        d.text((x + 8, y + 4), '(%s)' % chr(97 + k), fill=0, font=font())
    canvas.save(os.path.join(COMP, name))
    print('composite', name, canvas.size)


def main():
    # Figure 3: low-flux input, simple + complex
    p3 = []
    for tag, sl in SLICES.items():
        p = load_panel('file', FDK + '/FDK_raw_200Views_noise1e3/{s}', sl)
        p.save(os.path.join(OUT, 'fig3_input1e3_%s.png' % tag))
        p3.append(p)
    compose(p3, 2, 'fig3.png')

    # Figures 5/6: eight methods, simple / complex
    for tag, sl in SLICES.items():
        panels = []
        for meth, kind, arg in METHODS8:
            p = load_panel(kind, arg, sl)
            p.save(os.path.join(OUT, 'fig5_%s_%s.png' % (meth, tag)))
            panels.append(p)
        if tag == 'simple':
            compose(panels, 4, 'fig5.png')
        else:
            compose(panels, 4, 'fig6.png', box=True)
            # Figure 7: ROI zooms of the same eight methods
            for meth, p in zip([m[0] for m in METHODS8], panels):
                p.crop(ROI).resize((660, 360), Image.LANCZOS).save(
                    os.path.join(OUT, 'fig7_%s_complex.png' % meth))
            compose(panels, 4, 'fig7.png', zoom=True)

    # Figure 8: ablation, complex
    panels = []
    for meth, kind, arg in FIG8:
        p = load_panel(kind, arg, SLICES['complex'])
        p.save(os.path.join(OUT, 'fig8_%s_complex.png' % meth))
        panels.append(p)
    compose(panels, 4, 'fig8.png')

    # Figure 9: PCA panels unchanged
    f9 = Image.open(os.path.join(HERE, 'feature_analysis_1e3.png')).convert('L')
    w, h = f9.size
    f9.crop((0, 0, w // 2, h)).save(os.path.join(OUT, 'fig9_withoutMMD.png'))
    f9.crop((w // 2, 0, w, h)).save(os.path.join(OUT, 'fig9_withMMD_UDAN.png'))
    compose([f9.crop((0, 0, w // 2, h)), f9.crop((w // 2, 0, w, h))], 2, 'fig9.png')


if __name__ == '__main__':
    main()
