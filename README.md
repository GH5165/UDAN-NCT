# UDAN-NCT: Unsupervised Domain Adaptation Network for Low-Flux Neutron CT Image Denoising

Complete code release for the revised manuscript. The repository contains the
original research code and the corrected, unified-protocol implementations used
to produce every quantitative table and figure in the revision.

```
udan_original/         Original UDAN training code (PyTorch)
comparison_original/   Original DnCNN / REDCNN / FBPConvNet experiments
matlab/                FRM-II projection -> FDK/SART pipeline (TIGRE)
revision_experiments/  Unified-protocol training, evaluation, analysis and figures
results/               Final result tables and feature-analysis statistics
```

## Revised experimental protocol

The comparison-network training loaders in `comparison_original/` use paired
**NCT low-flux inputs and normal-flux labels**. All local absolute paths in
every script have been replaced by empty strings with `TODO: set local path`
comments; set these paths to your own data layout before running.

- Main operating point: projection-domain Poisson + electronic noise at
  **I0 = 10³** counts per detector element, followed by FDK reconstruction.
- Larger-domain-gap study: two additional harsher conditions,
  **I0 = 8×10²** and **I0 = 7×10²**, generated with the identical pipeline.
- Test data: 122 training slices and 30 held-out test slices at the main
  operating point (31 for the additional conditions), with disjoint positions.
- Metrics: grayscale float [0,1], SSIM/PSNR with `data_range=1`, RMSE, computed
  inside the inscribed reconstruction-FOV circle (radius 240 px); mean ± SD.
- Checkpoint selection: highest evaluation SSIM, identical rule for all methods.
- UDAN deployment: EMA weights (decay 0.9995) and horizontal-flip self-ensembling;
  comparison methods use native single-pass inference.

## `revision_experiments/` contents

| File | Purpose |
|---|---|
| `train_udan.py` | UDAN, no-MMD/no-WT/no-residual ablations, MMD transplants (DnCNN/REDCNN/FBPConvNet), sensitivity (σ, λ, depth, EMA), domain-gap adaptation |
| `train_supervised.py` | X-ray direct-transfer baselines and per-condition NCT-supervised references, including the UDAN supervised upper bound |
| `nets.py` | All network variants, including residual output anchoring |
| `tigre_repro.py` | Python replication of the TIGRE noise+FDK pipeline; `--gen --i0 8e2 --tag 8e2` generates train/test slices |
| `eval_matrix.py` | Unified multi-condition evaluation (`TS_ONLY`, `MODEL_ONLY`, `OUT_TAG` filters) |
| `eval_udan_tta.py` | UDAN EMA + flip self-ensemble evaluation |
| `make_final_tables.py` | Main-table assembly and requirement checks |
| `make_table5_variants.py` | Domain-gap tables for the 8e2 and 7e2 conditions |
| `feature_analysis_1e3.py` | Figure 9 PCA feature-alignment analysis |
| `feature_probe.py` | Per-layer normalized feature-domain distances across acquisition conditions |
| `input_baselines.py` | FDK input baselines across flux levels |
| `sart_python.py` | Re-projection SART baseline with early stopping |
| `make_figures_final.py` | Brightness-normalized subfigures and composites (simple slice 20, complex slice 31) |
| `make_fig67_redcnn.py` | Figure 6/7 composites with the REDCNN supervised reference |
| `figures_brighten.py` | light_norm-style display normalization (visualization only) |
| `export_panels.py` | Individual panel export |

Update the path constants in `revision_experiments/common.py` and the data-path
constants at the top of the evaluation/generation scripts to match your local
layout. No NCT data are redistributed here; FRM-II projections are available
under the license described in the manuscript.

## Loss-weight convention

The implementation optimizes `MSE + c × MMD`. Equation (16) in the manuscript
uses the convex combination `(1−λ) MSE + λ MMD`; therefore `c = λ/(1−λ)`.
The adopted configuration is `c = 1`, i.e. **λ = 1/2** in Eq. (16).
The sensitivity rows λ = 1/3, 1/2 and 2/3 correspond to code ratios 0.5, 1 and 2.

## Main results (r240, SSIM)

| Method / condition | I0=10³ | I0=8×10² | I0=7×10² |
|---|---:|---:|---:|
| FDK input | 0.624 | 0.602 | 0.580 |
| Best direct transfer (FBPConvNet) | 0.876 | 0.863 | 0.852 |
| Best transplanted MMD (DnCNN+MMD) | 0.915 | 0.871 | 0.864 |
| **UDAN (label-free)** | **0.926** | **0.898** | **0.888** |
| UDAN supervised upper bound | 0.964 | 0.948 | 0.943 |

Complete mean ± SD tables (SSIM/PSNR/RMSE) are in `results/final_tables_r240.md`,
`results/final_table5_8e2.md` and `results/final_table5_7e2.md`.
The full machine-readable evaluation is in `results/summary.csv`.

## Feature-space evidence

`results/feature_probe.csv` reports per-layer normalized mean-embedding
distances. At the deepest encoder stage the X-ray↔NCT distance is 1.81 before
adaptation and 0.043–0.045 after label-free MMD adaptation at all three
acquisition conditions, including conditions not used during adaptation.

## Environment

Python 3.8–3.10, PyTorch ≥ 1.11 (CUDA), scikit-image, PyWavelets,
scikit-learn, OpenCV, Pillow; MATLAB with TIGRE for the original projection
pipeline. The Python `tigre_repro.py` generator does not require MATLAB.

## License and citation

MIT license. GitHub: https://github.com/GH5165/UDAN-NCT
Archived release: https://doi.org/10.5281/zenodo.23328862
