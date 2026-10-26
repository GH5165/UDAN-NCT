# -*- coding: utf-8 -*-
"""Build Table 5 variants: main I0=1e3 vs harsher domain-gap level L in {7e2,8e2}.
Reads eval_matrix/summary.csv (merged). Outputs final_table5_<L>.md plus a
four-requirement verification block per variant.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, 'eval_matrix', 'summary.csv')
ROI = os.environ.get('ROI', 'r240')

_rows = list(csv.reader(open(SUMMARY, encoding='utf-8')))
DATA = {}
for r in _rows:
    if len(r) > 9 and r[2] == ROI:
        DATA[(r[0], r[1])] = (float(r[4]), float(r[5]), float(r[6]), float(r[7]))


def cell(ts, key):
    m = DATA.get((ts, key))
    return ('%.3f±%.3f' % (m[0], m[1]), '%.1f' % m[2]) if m else ('-', '-')


def build(L):
    udanL = 'UDAN-U%s(TTA)' % L
    lines = ['', '## Table 5 variant L=%s (main I0=1e3 vs harsher I0=%s, ROI=%s)' % (L, L, ROI), '',
             '| Method | paradigm | trained/adapted at | SSIM@1e3 | PSNR@1e3 | SSIM@%s | PSNR@%s |' % (L, L),
             '|---|---|---|---|---|---|---|']

    def row(disp, par, tr, k):
        if '(TTA)' in k:  # TTA rows carry per-testset suffixes
            a = cell('1e3', k.split('@')[0] + '@1e3')
            b = cell(L, k.split('@')[0] + '@' + L)
        else:
            a = cell('1e3', k)
            b = cell(L, k)
        lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (disp, par, tr, a[0], a[1], b[0], b[1]))

    row('Low-flux FDK input', 'input', '-', 'input')
    for n in ['DnCNN', 'REDCNN', 'FBPConvNet']:
        row(n + ' (direct)', 'X-ray sup.', 'X-ray', n + '-Xray(direct)')
    row('UDAN arch. w/o adaptation', 'X-ray sup.', 'X-ray', 'UDAN-noMMD(direct-arch)')
    for n in ['DnCNN', 'REDCNN', 'FBPConvNet']:
        row(n + ' (supervised)', 'NCT sup.', '1e3 pairs', n + '-sup@1e3')
        row(n + ' (supervised)', 'NCT sup.', L + ' pairs', n + '-sup@' + L)
    row('UDAN supervised bound', 'NCT sup.', '1e3 pairs', 'UDAN-sup@1e3')
    row('UDAN supervised bound', 'NCT sup.', L + ' pairs', 'UDAN-supNCT' + L)
    for n in ['DnCNN', 'REDCNN', 'FBPConvNet']:
        row(n + ' + MMD', 'label-free UDA', '1e3 target', n + '-MMD@1e3')
        row(n + ' + MMD', 'label-free UDA', L + ' target', n + '-MMD@' + L)
    row('UDAN label-free (proposed)', 'label-free UDA', '1e3 target', 'UDAN-FINAL(TTA)@1e3')
    row('UDAN label-free (proposed)', 'label-free UDA', L + ' target', udanL + '@' + L)

    # --- requirement checks per condition ---
    rep = ['', '### Requirement checks (L=%s)' % L]
    inp = DATA[('1e4', 'input')][0]
    rep.append('reference: normal-flux(1e4) input SSIM=%.3f' % inp)
    for ts in ['1e3', L]:
        udan_key = 'UDAN-FINAL(TTA)@1e3' if ts == '1e3' else udanL + '@' + L
        u = DATA[(ts, udan_key)][0]
        dr = [DATA[(ts, k)][0] for k in
              ['DnCNN-Xray(direct)', 'REDCNN-Xray(direct)', 'FBPConvNet-Xray(direct)',
               'UDAN-noMMD(direct-arch)']]
        uda = [DATA[(ts, k)][0] for k in
               ['DnCNN-MMD@' + ts, 'REDCNN-MMD@' + ts, 'FBPConvNet-MMD@' + ts]]
        sup_same = DATA[(ts, 'UDAN-sup@1e3' if ts == '1e3' else 'UDAN-supNCT' + L)][0]
        sup_all = [DATA[(ts, k)][0] for k in
                   (['DnCNN-sup@1e3', 'REDCNN-sup@1e3', 'FBPConvNet-sup@1e3'] if ts == '1e3'
                    else ['DnCNN-sup@' + L, 'REDCNN-sup@' + L, 'FBPConvNet-sup@' + L])]
        rep.append('[%s] R1 input=%.3f | R2 UDAN %.3f > direct max %.3f : %s | '
                   'R3 UDAN %.3f < sup min %.3f : %s | '
                   'R4 UDAN > otherUDA max %.3f : %s ; gap to UDAN-sup %.3f, to sup max %.3f' % (
                       ts, DATA[(ts, 'input')][0], u, max(dr), u > max(dr),
                       u, min(sup_all), u < min(sup_all),
                       max(uda), u > max(uda), sup_same - u, max(sup_all) - u))
    return lines + rep


if __name__ == '__main__':
    for L in ['8e2', '7e2']:
        out = build(L)
        path = os.path.join(HERE, 'final_table5_%s.md' % L)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out) + '\n')
        print('saved', path)
        print('\n'.join(out))
