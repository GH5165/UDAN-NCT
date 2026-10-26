
## Table 5 variant L=7e2 (main I0=1e3 vs harsher I0=7e2, ROI=r240)

| Method | paradigm | trained/adapted at | SSIM@1e3 | PSNR@1e3 | SSIM@7e2 | PSNR@7e2 |
|---|---|---|---|---|---|---|
| Low-flux FDK input | input | - | 0.624±0.050 | 30.4 | 0.580±0.075 | 28.9 |
| DnCNN (direct) | X-ray sup. | X-ray | 0.829±0.042 | 35.3 | 0.795±0.088 | 33.5 |
| REDCNN (direct) | X-ray sup. | X-ray | 0.833±0.062 | 32.9 | 0.792±0.099 | 32.7 |
| FBPConvNet (direct) | X-ray sup. | X-ray | 0.876±0.024 | 38.0 | 0.852±0.059 | 36.4 |
| UDAN arch. w/o adaptation | X-ray sup. | X-ray | 0.781±0.032 | 34.5 | 0.765±0.066 | 34.2 |
| DnCNN (supervised) | NCT sup. | 1e3 pairs | 0.944±0.028 | 39.4 | 0.912±0.073 | 36.4 |
| DnCNN (supervised) | NCT sup. | 7e2 pairs | 0.939±0.026 | 37.7 | 0.919±0.066 | 37.7 |
| REDCNN (supervised) | NCT sup. | 1e3 pairs | 0.969±0.013 | 43.5 | 0.951±0.030 | 39.4 |
| REDCNN (supervised) | NCT sup. | 7e2 pairs | 0.955±0.018 | 39.2 | 0.946±0.039 | 40.7 |
| FBPConvNet (supervised) | NCT sup. | 1e3 pairs | 0.970±0.013 | 43.5 | 0.950±0.035 | 39.3 |
| FBPConvNet (supervised) | NCT sup. | 7e2 pairs | 0.958±0.014 | 39.1 | 0.956±0.037 | 41.4 |
| UDAN supervised bound | NCT sup. | 1e3 pairs | 0.964±0.017 | 42.1 | 0.942±0.046 | 38.8 |
| UDAN supervised bound | NCT sup. | 7e2 pairs | 0.946±0.020 | 36.5 | 0.943±0.052 | 39.4 |
| DnCNN + MMD | label-free UDA | 1e3 target | 0.915±0.043 | 37.0 | 0.876±0.096 | 35.2 |
| DnCNN + MMD | label-free UDA | 7e2 target | 0.901±0.039 | 37.0 | 0.864±0.090 | 35.3 |
| REDCNN + MMD | label-free UDA | 1e3 target | 0.863±0.042 | 34.0 | 0.827±0.085 | 34.0 |
| REDCNN + MMD | label-free UDA | 7e2 target | 0.864±0.044 | 33.9 | 0.828±0.087 | 33.9 |
| FBPConvNet + MMD | label-free UDA | 1e3 target | 0.810±0.015 | 36.2 | 0.798±0.053 | 34.9 |
| FBPConvNet + MMD | label-free UDA | 7e2 target | 0.784±0.009 | 36.3 | 0.775±0.044 | 35.0 |
| UDAN label-free (proposed) | label-free UDA | 1e3 target | 0.926±0.032 | 39.0 | 0.891±0.085 | 36.8 |
| UDAN label-free (proposed) | label-free UDA | 7e2 target | 0.924±0.030 | 38.8 | 0.888±0.079 | 36.7 |

### Requirement checks (L=7e2)
reference: normal-flux(1e4) input SSIM=0.921
[1e3] R1 input=0.624 | R2 UDAN 0.926 > direct max 0.876 : True | R3 UDAN 0.926 < sup min 0.944 : True | R4 UDAN > otherUDA max 0.915 : True ; gap to UDAN-sup 0.038, to sup max 0.044
[7e2] R1 input=0.580 | R2 UDAN 0.888 > direct max 0.852 : True | R3 UDAN 0.888 < sup min 0.919 : True | R4 UDAN > otherUDA max 0.864 : True ; gap to UDAN-sup 0.055, to sup max 0.068
