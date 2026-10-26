# UDAN-NCT: Unsupervised Domain Adaptation Network for Low-Flux Neutron CT Image Denoising

Source code for data generation, training and evaluation of the UDAN denoising
framework and all comparison methods.

```
udan_original/         Original UDAN training code (PyTorch)
comparison_original/   Original DnCNN / REDCNN / FBPConvNet training code
matlab/                FRM-II projection -> FDK/SART pipeline (TIGRE)
experiments/           Unified-protocol training and evaluation scripts
results/               Evaluation outputs (per-condition metrics, feature statistics)
```

## Experimental protocol

The comparison-network training loaders in `comparison_original/` use paired
**NCT low-flux inputs and normal-flux labels**. All local absolute paths in
every script have been replaced by empty strings with `TODO: set local path`
comments; set these paths to your own data layout before running.

- Main operating point: projection-domain Poisson + electronic noise at
  **I0 = 10³** counts per detector element, followed by FDK reconstruction.
- Domain-gap conditions: two harsher flux levels,
  **I0 = 8×10²** and **I0 = 7×10²**, generated with the identical pipeline.
- Data: 122 training slices and 30 held-out test slices at the main operating
  point (31 for the additional conditions), with disjoint positions.
- Metrics: grayscale float [0,1], SSIM/PSNR with `data_range=1`, RMSE, computed
  inside the inscribed reconstruction-FOV circle (radius 240 px); mean ± SD.
- Checkpoint selection: highest evaluation SSIM, identical rule for all methods.
- UDAN deployment: EMA weights (decay 0.9995) and horizontal-flip self-ensembling;
  comparison methods use native single-pass inference.

## `experiments/` contents

| File | Purpose |
|---|---|
| `train_udan.py` | UDAN training, ablations (no-MMD / no-WT / no-residual), MMD transplants (DnCNN/REDCNN/FBPConvNet), sensitivity (σ, λ, depth, EMA), adaptation at other flux levels |
| `train_supervised.py` | X-ray direct-transfer baselines and per-flux NCT-supervised references, including the UDAN supervised upper bound |
| `nets.py` | Network variants, including residual output anchoring |
| `tigre_repro.py` | Python replication of the TIGRE noise+FDK pipeline; `--gen --i0 8e2 --tag 8e2` generates train/test slices |
| `eval_matrix.py` | Multi-condition evaluation of all checkpoints (`TS_ONLY`, `MODEL_ONLY`, `OUT_TAG` filters) |
| `eval_udan_tta.py` | UDAN EMA + flip self-ensemble evaluation |
| `eval_circular.py` | Circular-ROI evaluation and error decomposition |
| `final_eval.py` / `extended_metrics.py` | Unified final evaluation and extended metrics |
| `feature_analysis.py` / `feature_analysis_1e3.py` | PCA feature-alignment analysis |
| `feature_probe.py` | Per-layer normalized feature-domain distances across flux levels |
| `input_baselines.py` | FDK input baselines across flux levels |
| `sart_python.py` | Re-projection SART baseline with early stopping |
| `common.py` | Shared data loading and metric utilities |

Update the path constants in `experiments/common.py` and the data-path
constants at the top of the evaluation/generation scripts to match your local
layout. No NCT data are redistributed here; FRM-II projections are available
under the license described in the paper's Data Availability statement.

## Loss-weight convention

The implementation optimizes `MSE + c × MMD`. A convex-combination form
`(1−λ) MSE + λ MMD` corresponds to `c = λ/(1−λ)`. The default configuration is
`c = 1`, i.e. **λ = 1/2**; the sensitivity values λ = 1/3, 1/2 and 2/3
correspond to code ratios 0.5, 1 and 2.

## Results

Summary metrics for every model and flux level are in `results/summary.csv`;
per-layer feature statistics are in `results/feature_probe.csv`; FDK input
baselines across ten flux levels are in `results/input_baselines_levels.csv`.

## Environment

Python 3.8–3.10, PyTorch ≥ 1.11 (CUDA), scikit-image, PyWavelets,
scikit-learn, OpenCV, Pillow; MATLAB with TIGRE for the original projection
pipeline. The Python `tigre_repro.py` generator does not require MATLAB.

## License and citation

MIT license. GitHub: https://github.com/GH5165/UDAN-NCT
Archived release: https://doi.org/10.5281/zenodo.23328862
