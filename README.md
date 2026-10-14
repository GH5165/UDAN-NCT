# UDAN-NCT: Unsupervised Domain Adaptation Network for Low-Flux Neutron CT Image Denoising

Complete code release for the manuscript. The repository is organized in three
layers that together cover every experiment in the paper and its revision:

```
udan_original/         Original UDAN training code (PyTorch)
                       - main.py            : UDAN training (source-supervised + MMD target alignment)
                       - model3.py          : UDAN architecture (WT/DnCNN branch, MMD after down4)  [paper model]
                       - model5.py          : ablation - skip connections instead of WT branch
                       - model6.py          : ablation - without MMD
                       - datasets.py, mmd.py, unet_parts.py, test_chest.py, ...
comparison_original/   Original comparison-network experiments (PyTorch)
                       - run/run_DnCNN.py, run_REDCNN.py, run_FBPConvNet.py : supervised training on
                         the X-ray source pairs + direct transfer to NCT
                       - run/compare_SSIM.py : unified metric re-evaluation
                       - net/  : DnCNN / REDCNN / FBPConvNet definitions
                       - dataset/, myTool/  : data loading and normalization utilities
matlab/                MATLAB data pipeline (FRM II projections -> FDK/SART)
                       - readRaw.m                : uint16 .raw projection -> tif conversion
                       - tigre_work/raw_FDK.m     : clean FDK reconstruction (reference images)
                       - tigre_work/raw_noise_FDK.m : TIGRE addCTnoise Poisson simulation (I0) + FDK
                         (low-flux inputs; requires the TIGRE toolbox: https://github.com/CERN/TIGRE,
                          Utilities/addCTnoise.m)
                       - tigre_work/raw_SART.m, raw_OSART.m, raw_SART_TV.m : SART reconstructions
                       - ART/demo1_ART.m, ART.m   : ART-based baseline
                       - noise/addNoise.m         : radon-domain multiplicative Gaussian noise (X-ray)
revision_experiments/  Unified-protocol experiments added during revision (PyTorch)
                       - train_supervised.py : direct-transfer baselines + supervised UDAN upper bound
                       - train_udan.py       : UDAN / ablations / MMD-transplant variants / MMD
                                               sensitivity (sigma, lambda, depth, batch) / domain-gap
                       - nets.py             : all network variants incl. residual output anchoring
                       - final_eval.py       : unified evaluation (inscribed-FOV-circle protocol)
                       - feature_analysis.py : feature alignment analysis - PCA projection, normalized MMD,
                                               domain probe (Figure 9)
                       - sart_python.py      : re-projection SART baseline with early stopping
results/final_table.json   All quantitative results under the unified protocol
```

Note: the scripts under `udan_original/`, `comparison_original/` and `matlab/`
are the original research code and reference the original data paths; the
scripts under `revision_experiments/` are the corrected, self-contained
re-implementations used to produce every table of the revised manuscript under
one documented protocol.

## Method summary

UDAN denoises low-flux NCT without paired NCT labels:
- Source domain (supervised): paired low-/normal-flux X-ray CT (MSE loss).
- Target domain (unsupervised): unlabeled low-flux NCT; encoder features of
  both domains aligned with an MMD loss (single Gaussian RBF kernel, sigma = 9,
  per-channel computation at the deepest encoder stage, batch 1).
- Architecture: U-Net encoder-decoder; the two deepest skip connections are
  replaced by a wavelet-transform branch (bior1.3 low-pass subbands) followed
  by compact DnCNN blocks.

## Data layout (revision experiments)

Update the path constants in `revision_experiments/common.py`:

| Constant | Content |
|---|---|
| XRAY_FULL / XRAY_LOW | paired normal-/low-flux X-ray CT slices (512x512 PNG) |
| NCT_TRAIN_LOW | unlabeled low-flux NCT slices (Poisson simulation, I0 = 1e4) |
| NCT_REF / NCT_LOW | NCT test set: normal-flux FDK reference + low-flux input |
| NCT_LOW_1E3 | harsher low-flux test set (I0 = 1e3) for the domain-gap experiment |

Low-flux NCT generation (MATLAB/TIGRE, `matlab/tigre_work/raw_noise_FDK.m`):
```matlab
Im = I0 * exp(-p / max(p(:)));            % p: measured intensity projections
Im = poissrnd(Im) + randn(size(Im))*0.5;  % photon + electronic noise
p_noisy = log(I0 ./ max(Im,1e-6)) * max(p(:));
b = -log(p_noisy / (max(p_noisy(:))+1));  % line integrals -> FDK
```

## Unified evaluation protocol (revision tables)

- Images loaded as grayscale, /255 (float [0,1]).
- SSIM / PSNR with data_range=1, RMSE = sqrt(MSE), computed inside the
  inscribed circle of the reconstruction FOV (radius 240 px); 31 name-matched
  test pairs; mean +/- SD.
- Model selection: highest full-frame SSIM on the evaluation set, identical
  rule for every method.

## Environment

Python 3.8-3.10, PyTorch >= 1.11 (CUDA), scikit-image, PyWavelets,
scikit-learn; MATLAB with the TIGRE toolbox for the projection pipeline.

## Citation / license

MIT license. NCT projection data (FRM II, glued-box sample) available from the
Forschungs-Neutronenquelle Heinz Maier-Leibnitz under license; see the
manuscript's Data Availability statement.
