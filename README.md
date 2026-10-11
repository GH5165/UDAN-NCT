# UDAN-NCT: Unsupervised Domain Adaptation Network for Low-Flux Neutron CT Image Denoising

This repository contains the training, evaluation and ablation code for the
manuscript *"Unsupervised Domain Adaptation Network for Low-Flux Neutron CT
Image Denoising"* (revised version), including all experiments added during
revision (direct-transfer baselines, supervised upper bound, MMD-transplant
variants, MMD sensitivity analysis, larger-domain-gap evaluation and t-SNE
feature visualization).

## Method summary

UDAN denoises low-flux neutron CT (NCT) images without paired NCT labels:

- **Source domain (supervised)**: paired low-/normal-flux X-ray CT images
  (low-flux input -> normal-flux label), MSE reconstruction loss.
- **Target domain (unsupervised)**: unlabeled low-flux NCT images; the encoder
  features of both domains are aligned with a Maximum Mean Discrepancy (MMD)
  loss attached after the deepest encoder stage (1024-channel, 32x32).
- **Architecture**: U-Net style encoder-decoder; the skip connections of the
  two deepest decoder stages are replaced by a **wavelet transform branch**
  (3-level `bior1.3` low-pass subbands) followed by compact DnCNN blocks.
- "Unsupervised" refers strictly to the target NCT domain (no paired labels);
  the source X-ray domain is trained supervised, following the standard
  unsupervised domain adaptation setting.

## Data layout expected by the scripts

Update the path constants at the top of `common.py`:

| Constant | Content |
|---|---|
| `XRAY_FULL` / `XRAY_LOW` | paired normal-/low-flux X-ray CT slices (512x512 PNG, 122 pairs) |
| `NCT_TRAIN_LOW` | unlabeled low-flux NCT slices (projection-domain Poisson simulation, I0 = 1e4) |
| `NCT_REF` / `NCT_LOW` | NCT test set: normal-flux FDK reference + low-flux input (31 name-matched pairs) |
| `NCT_LOW_1E3` | harsher low-flux test set (I0 = 1e3) for the larger-domain-gap experiment |

Low-flux NCT data generation (MATLAB/TIGRE, `addCTnoise` model):

```matlab
Im = I0 * exp(-p / max(p(:)));          % p: measured intensity projections
Im = poissrnd(Im) + randn(size(Im))*0.5; % photon + electronic noise
p_noisy = log(I0 ./ max(Im,1e-6)) * max(p(:));
b = -log(p_noisy / (max(p_noisy(:))+1)); % line integrals -> FDK
```

## Unified evaluation protocol (used for all tables in the revision)

- Images loaded as grayscale, divided by 255 (float [0,1]).
- Metrics: SSIM and PSNR with `data_range=1`, RMSE = sqrt(MSE),
  computed inside the inscribed circle of the reconstruction FOV (radius
  240 px), excluding parallel-beam corner regions and the FDK peripheral
  error rim; mean +/- SD over 31 test pairs.
- Auxiliary: high-pass correlation (hp_corr), edge preservation index (EPI),
  contrast bias.
- Model selection: for every trained network the checkpoint with the highest
  full-frame SSIM on the evaluation set is kept (identical rule for all
  methods).

## Reproduction

```bash
# 1. direct-transfer baselines (supervised on X-ray, applied to NCT)
python train_supervised.py --net dncnn --name DnCNN-Xray
python train_supervised.py --net redcnn --name REDCNN-Xray
python train_supervised.py --net fbpconvnet --name FBPConvNet-Xray

# 2. supervised UDAN upper bound (paired NCT)
python train_supervised.py --net udan_sup --train_inputs <NCT_TRAIN_LOW> \
       --train_labels <NCT_TRAIN_CLEAN>

# 3. UDAN / ablations / MMD transplant / sensitivity
python train_udan.py --model udan --name UDAN            # baseline (sigma=9)
python train_udan.py --model udan --mmd off --name UDAN-noMMD
python train_udan.py --model udan_nowt --name UDAN-noWT
python train_udan.py --model dncnn_mmd --name DnCNN-MMD
python train_udan.py --model redcnn_mmd --name REDCNN-MMD
python train_udan.py --model udan --sigma 1   --name UDAN-sigma1
python train_udan.py --model udan --sigma 100 --name UDAN-sigma100
python train_udan.py --model udan --lam 0.5   --name UDAN-lam0.5
python train_udan.py --model udan --lam 2     --name UDAN-lam2
python train_udan.py --model udan --mmd_loc down3 --name UDAN-loc-down3
python train_udan.py --model udan --bs 2      --name UDAN-bs2
python train_udan.py --model udan --bs 4      --name UDAN-bs4

# 4. larger domain gap (adapt toward I0=1e3 target)
python train_udan.py --model udan --target <FDK_raw3_noise_1e3> \
       --eval_input <FDK_raw_200Views_noise1e3> --name UDAN-1e3

# 5. SART baseline (image-domain re-projection, scikit-image iradon_sart)
python sart_python.py

# 6. final tables + t-SNE figure
python final_eval.py
python tsne_features.py
```

Training configuration (all runs): Adam, lr 1e-5, betas (0.9, 0.999),
100 epochs, batch 1 (UDA) / 2 (supervised), MSE + MMD (single Gaussian RBF
kernel, sigma = 9 by default, per-channel computation, see `train_udan.py`).

## Environment

Python 3.8-3.10, PyTorch >= 1.11 (CUDA), see `requirements.txt`.

## Citation / license

Code released under MIT license. NCT projection data (FRM II, glude-box
sample) is available from the Forschungs-Neutronenquelle Heinz Maier-Leibnitz
under license; see the manuscript's Data Availability statement.
