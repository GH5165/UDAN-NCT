# -*- coding: utf-8 -*-
"""Replace the supervised-reference panel in Figures 6 and 7 with REDCNN-sup.
Subpanels saved to qoder_fig; composites rebuilt; manuscript updated (v15->v16)
with captions changed accordingly.
"""
import os
import sys
import docx
from docx.enum.text import WD_COLOR_INDEX
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures_final import load_panel, compose, OUT, SLICES

ROOT = ''  # TODO: set local path
METHODS = [
    ('reference', 'file', ''),  # TODO: set local path
    ('input', 'file', ''),  # TODO: set local path
    ('SART', 'outputs', 'SART1e3'),
    ('FBPConvNet_direct', 'outputs_1e3', 'FBPConvNet-Xray'),
    ('DnCNN_MMD', 'outputs_1e3', 'DnCNN-MMD1e3'),
    ('UDAN', 'outputs_1e3', 'U1e_d95'),
    ('REDCNN_sup', 'outputs_1e3', 'REDCNN-NCT1e3'),
    ('FBPConvNet_sup', 'outputs_1e3', 'FBPConvNet-NCT1e3'),
]

panels = []
for meth, kind, arg in METHODS:
    p = load_panel(kind, arg, SLICES['complex'])
    if meth == 'REDCNN_sup':
        p.save(os.path.join(OUT, 'fig5_REDCNN_sup_complex.png'))
        p.crop((150, 220, 370, 340)).resize((660, 360), Image.LANCZOS).save(
            os.path.join(OUT, 'fig7_REDCNN_sup_complex.png'))
    panels.append(p)
compose(panels, 4, 'fig6.png', box=True)
compose(panels, 4, 'fig7.png', zoom=True)

doc = docx.Document(os.path.join(ROOT, 'SR_UDAN_final_v15.docx'))
for idx, fn in [(5, 'fig6.png'), (6, 'fig7.png')]:
    shape = doc.inline_shapes[idx]
    rid = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
    part = doc.part.related_parts[rid]
    with open(os.path.join(ROOT, 'qoder_fig', 'composites', fn), 'rb') as f:
        part._blob = f.read()
    with Image.open(os.path.join(ROOT, 'qoder_fig', 'composites', fn)) as im:
        w, h = im.size
    shape.height = int(shape.width * h / w)
    print('replaced', fn)

for p in doc.paragraphs:
    if p.text.strip().startswith(('Figure 6:', 'Figure 7:')):
        for r in p.runs:
            if 'UDAN supervised upper bound' in r.text:
                r.text = r.text.replace('UDAN supervised upper bound',
                                        'REDCNN supervised on NCT pairs')
                r.font.highlight_color = WD_COLOR_INDEX.BRIGHT_GREEN
                print('caption updated:', p.text[:12])

doc.save(os.path.join(ROOT, 'SR_UDAN_final_v16.docx'))
print('saved SR_UDAN_final_v16.docx')
