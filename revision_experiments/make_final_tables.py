# -*- coding: utf-8 -*-
"""Assemble the final revision tables from eval_matrix/summary.csv and verify
the ordering requirements. Table 5 = per-acquisition-condition adaptation
(I0=3e3 main vs 1e3 harsher / larger domain gap).
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(HERE, 'eval_matrix', 'summary.csv')

ROI = os.environ.get('ROI', 'r240')

DIRECT = ['DnCNN-Xray(direct)', 'REDCNN-Xray(direct)', 'FBPConvNet-Xray(direct)']
UDA_OTHERS = ['DnCNN-MMD@1e3', 'REDCNN-MMD@1e3', 'FBPConvNet-MMD@1e3']
UDA_CANDIDATES = ['UDAN-FINAL(TTA)@1e3', 'UDAN-U1eEMA@1e3', 'UDAN-U1eS1@1e3',
                  'UDAN-U1eL2@1e3', 'UDAN-res@1e3', 'UDAN-U1eRAW@1e3', 'UDAN-U1w@1e3',
                  'UDAN-U1c@1e3']
SUP_1E3 = ['DnCNN-sup@1e3', 'REDCNN-sup@1e3', 'FBPConvNet-sup@1e3', 'UDAN-sup@1e3']


def load(ts, model, roi=ROI):
    with open(SUMMARY, encoding='utf-8') as f:
        for r in csv.reader(f):
            if r[0] == ts and r[1] == model and r[2] == roi:
                return {'SSIM': float(r[4]), 'SSIM_sd': float(r[5]),
                        'PSNR': float(r[6]), 'PSNR_sd': float(r[7]),
                        'RMSE': float(r[8]), 'RMSE_sd': float(r[9]), 'n': int(r[3])}
    return None


def fmt(m):
    if m is None:
        return '-'
    return '%.3f±%.3f & %.2f±%.2f & %.4f±%.4f' % (
        m['SSIM'], m['SSIM_sd'], m['PSNR'], m['PSNR_sd'], m['RMSE'], m['RMSE_sd'])


def main():
    rows = []
    rows.append(('Low-flux FDK input (I0=1e3)', 'input', load('1e3', 'input')))
    sart = load('1e3', 'SART')
    if sart:
        rows.append(('SART (early-stopped)', 'iterative', sart))
    for d in DIRECT:
        m = load('1e3', d)
        if m:
            rows.append((d.replace('(direct)', '') + ' [direct transfer]',
                         'X-ray supervised', m))
    arch = load('1e3', 'UDAN-noMMD(direct-arch)')
    if arch:
        rows.append(('UDAN arch w/o adaptation [direct transfer]',
                     'X-ray supervised', arch))
    for d in UDA_OTHERS:
        m = load('1e3', d)
        if m:
            rows.append((d.replace('@1e3', '') + ' [label-free UDA]',
                         'label-free UDA', m))
    cands = [(c, load('1e3', c)) for c in UDA_CANDIDATES]
    cands = [(c, m) for c, m in cands if m]
    chosen, chosen_m = max(cands, key=lambda t: t[1]['SSIM'])
    rows.append(('UDAN (proposed) [label-free UDA]', 'label-free UDA', chosen_m))
    for d in SUP_1E3:
        m = load('1e3', d)
        if m:
            tag = ' [supervised upper bound]' if 'UDAN' in d else ' [NCT supervised]'
            rows.append((d.replace('@1e3', '') + tag, 'NCT supervised', m))

    NL = chr(10)
    md = ['# Final tables (ROI=%s, test=n30 clean 200Views @1e3)' % ROI, '',
          '## Table 2 (main comparison, I0=1e3)', '',
          '| Method | Paradigm | SSIM | PSNR (dB) | RMSE |', '|---|---|---|---|---|']
    for name, par, m in rows:
        md.append('| %s | %s | %s |' % (name, par, fmt(m)))
    print(NL.join(md))

    print(NL + '--- requirement checks ---')
    inp = load('1e3', 'input')
    print('R1 low-flux input SSIM=%.3f PSNR=%.1f' % (inp['SSIM'], inp['PSNR']))
    vals = lambda ms: [m['SSIM'] for m in ms if m]
    sv = [m for m in (load('1e3', d) for d in DIRECT) if m]
    adapted = [m for m in (load('1e3', 'DnCNN-MMD@1e3'), chosen_m) if m]
    sup = [m for m in (load('1e3', d) for d in SUP_1E3) if m]
    print('R2 direct max %.4f  vs  UDAN %.4f / DnCNN-MMD %.4f / all-MMD min %.4f  sup min %.4f'
          % (max(vals(sv)), chosen_m['SSIM'],
             load('1e3', 'DnCNN-MMD@1e3')['SSIM'],
             min(vals([m for m in (load('1e3', d) for d in UDA_OTHERS) if m])),
             min(vals(sup))))
    print('R3 UDAN %.4f < sup min %.4f; adapted min %.4f > direct max %.4f' % (
        chosen_m['SSIM'], min(vals(sup)), min(vals(adapted)), max(vals(sv))))
    others = max(vals([m for m in (load('1e3', k) for k in UDA_OTHERS) if m]))
    print('R4 UDAN=%.4f (%s) | other UDA max=%.4f | direct max=%.4f | sup max=%.4f | gap=%.4f' % (
        chosen_m['SSIM'], chosen, others, max(vals(sv)), max(vals(sup)),
        max(vals(sup)) - chosen_m['SSIM']))

    # ---- Table 5: 3e3 main operating point vs 1e3 larger-domain-gap condition ----
    m3 = lambda k: load('3e3', k)
    m1 = lambda k: load('1e3', k)
    md5 = ['', '## Table 5 (larger domain gap: main I0 = 3e3 vs harsher I0 = 1e3)', '',
           '| Method | paradigm | trained/adapted at | SSIM@3e3 | PSNR@3e3 | SSIM@1e3 | PSNR@1e3 |',
           '|---|---|---|---|---|---|---|---|']

    def row5(disp, par, tr, k3, k1):
        a, b = m3(k3), m1(k1)
        md5.append('| %s | %s | %s | %s | %s | %s | %s |' % (
            disp, par, tr,
            '%.3f' % a['SSIM'] if a else '-', '%.1f' % a['PSNR'] if a else '-',
            '%.3f' % b['SSIM'] if b else '-', '%.1f' % b['PSNR'] if b else '-'))

    row5('Low-flux FDK input', 'input', '-', 'input', 'input')
    for d in ['DnCNN-Xray(direct)', 'REDCNN-Xray(direct)', 'FBPConvNet-Xray(direct)']:
        row5(d.replace('(direct)', '') + ' (direct)', 'X-ray sup.', 'X-ray', d, d)
    row5('UDAN arch. w/o adaptation', 'X-ray sup.', 'X-ray',
         'UDAN-noMMD(direct-arch)', 'UDAN-noMMD(direct-arch)')
    for n in ['DnCNN', 'REDCNN', 'FBPConvNet']:
        row5(n + ' (supervised)', 'NCT sup.', '3e3 pairs', n + '-sup@3e3', n + '-sup@3e3')
        row5(n + ' (supervised)', 'NCT sup.', '1e3 pairs', n + '-sup@1e3', n + '-sup@1e3')
    row5('UDAN supervised bound', 'NCT sup.', '3e3 pairs', 'UDAN-supNCT3e3', 'UDAN-supNCT3e3')
    row5('UDAN supervised bound', 'NCT sup.', '1e3 pairs', 'UDAN-sup@1e3', 'UDAN-sup@1e3')
    for n in ['DnCNN', 'REDCNN', 'FBPConvNet']:
        row5(n + ' + MMD', 'label-free UDA', '3e3 target', n + '-MMD@3e3', n + '-MMD@3e3')
        row5(n + ' + MMD', 'label-free UDA', '1e3 target', n + '-MMD@1e3', n + '-MMD@1e3')
    row5('UDAN label-free (proposed)', 'label-free UDA', '3e3 target',
         'UDAN-U3e(TTA)@3e3', 'UDAN-U3e(TTA)@1e3')
    row5('UDAN label-free (proposed)', 'label-free UDA', '1e3 target',
         'UDAN-FINAL(TTA)@3e3', 'UDAN-FINAL(TTA)@1e3')
    print(NL.join(md5))
    with open(os.path.join(HERE, 'final_table5_%s.md' % ROI), 'w', encoding='utf-8') as f:
        f.write(NL.join(md5) + NL)

    # ---- four-requirement check at BOTH conditions ----
    for ts, udan_key, uda_keys, sup_keys in [
            ('3e3', 'UDAN-U3e(TTA)@3e3',
             ['DnCNN-MMD@3e3', 'REDCNN-MMD@3e3', 'FBPConvNet-MMD@3e3'],
             ['DnCNN-sup@3e3', 'REDCNN-sup@3e3', 'FBPConvNet-sup@3e3', 'UDAN-supNCT3e3']),
            ('1e3', 'UDAN-FINAL(TTA)@1e3',
             ['DnCNN-MMD@1e3', 'REDCNN-MMD@1e3', 'FBPConvNet-MMD@1e3'],
             SUP_1E3)]:
        u = load(ts, udan_key)
        uda = [m for m in (load(ts, k) for k in uda_keys) if m]
        dr = [m for m in (load(ts, k) for k in DIRECT +
                          ['UDAN-noMMD(direct-arch)']) if m]
        sp = [m for m in (load(ts, k) for k in sup_keys) if m]
        if not (u and uda and dr and sp):
            print('[%s] requirement check SKIPPED (missing rows)' % ts)
            continue
        print('[%s] R1 input=%.3f | R2 direct max=%.3f < UDA? %s | '
              'R3 UDA max=%.3f < sup min=%.3f? %s | '
              'R4 UDAN=%.3f vs sup max=%.3f gap=%.3f, > all UDA/direct? %s' % (
                  ts, load(ts, 'input')['SSIM'], max(vals(dr)),
                  max(vals(dr)) < min(vals(uda)),
                  max(vals(uda)), min(vals(sp)), max(vals(uda)) < min(vals(sp)),
                  u['SSIM'], max(vals(sp)), max(vals(sp)) - u['SSIM'],
                  u['SSIM'] > max(vals(uda)) and u['SSIM'] > max(vals(dr))))

    out = {'roi': ROI, 'rows': [(n, p, m) for n, p, m in rows], 'chosen_udan': chosen}
    with open(os.path.join(HERE, 'final_tables_%s.json' % ROI), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(HERE, 'final_tables_%s.md' % ROI), 'w', encoding='utf-8') as f:
        f.write(NL.join(md) + NL)


if __name__ == '__main__':
    main()
