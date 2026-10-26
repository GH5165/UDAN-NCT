
## Table 5 variant L=8e2 (main I0=1e3 vs harsher I0=8e2, ROI=r240)

| Method | paradigm | trained/adapted at | SSIM@1e3 | PSNR@1e3 | SSIM@8e2 | PSNR@8e2 |
|---|---|---|---|---|---|---|
| Low-flux FDK input | input | - | 0.624±0.050 | 30.4 | 0.602±0.078 | 29.5 |
| DnCNN (direct) | X-ray sup. | X-ray | 0.829±0.042 | 35.3 | 0.812±0.087 | 33.9 |
| REDCNN (direct) | X-ray sup. | X-ray | 0.833±0.062 | 32.9 | 0.811±0.097 | 33.1 |
| FBPConvNet (direct) | X-ray sup. | X-ray | 0.876±0.024 | 38.0 | 0.863±0.058 | 36.7 |
| UDAN arch. w/o adaptation | X-ray sup. | X-ray | 0.781±0.032 | 34.5 | 0.782±0.066 | 34.5 |
| DnCNN (supervised) | NCT sup. | 1e3 pairs | 0.944±0.028 | 39.4 | 0.917±0.069 | 36.5 |
| DnCNN (supervised) | NCT sup. | 8e2 pairs | 0.937±0.026 | 37.5 | 0.924±0.063 | 37.7 |
| REDCNN (supervised) | NCT sup. | 1e3 pairs | 0.969±0.013 | 43.5 | 0.952±0.029 | 39.5 |
| REDCNN (supervised) | NCT sup. | 8e2 pairs | 0.952±0.017 | 38.9 | 0.949±0.044 | 40.8 |
| FBPConvNet (supervised) | NCT sup. | 1e3 pairs | 0.970±0.013 | 43.5 | 0.951±0.033 | 39.3 |
| FBPConvNet (supervised) | NCT sup. | 8e2 pairs | 0.956±0.014 | 39.1 | 0.958±0.036 | 41.6 |
| UDAN supervised bound | NCT sup. | 1e3 pairs | 0.964±0.017 | 42.1 | 0.944±0.044 | 38.9 |
| UDAN supervised bound | NCT sup. | 8e2 pairs | 0.949±0.017 | 38.4 | 0.948±0.044 | 40.7 |
| DnCNN + MMD | label-free UDA | 1e3 target | 0.915±0.043 | 37.0 | 0.882±0.091 | 35.5 |
| DnCNN + MMD | label-free UDA | 8e2 target | 0.905±0.042 | 37.0 | 0.871±0.089 | 35.7 |
| REDCNN + MMD | label-free UDA | 1e3 target | 0.863±0.042 | 34.0 | 0.833±0.080 | 34.3 |
| REDCNN + MMD | label-free UDA | 8e2 target | 0.863±0.043 | 33.9 | 0.833±0.081 | 34.2 |
| FBPConvNet + MMD | label-free UDA | 1e3 target | 0.810±0.015 | 36.2 | 0.805±0.051 | 35.2 |
| FBPConvNet + MMD | label-free UDA | 8e2 target | 0.804±0.009 | 36.0 | 0.803±0.038 | 35.3 |
| UDAN label-free (proposed) | label-free UDA | 1e3 target | 0.926±0.032 | 39.0 | 0.897±0.081 | 37.1 |
| UDAN label-free (proposed) | label-free UDA | 8e2 target | 0.925±0.033 | 38.6 | 0.898±0.083 | 37.1 |

### Requirement checks (L=8e2)
reference: normal-flux(1e4) input SSIM=0.921
[1e3] R1 input=0.624 | R2 UDAN 0.926 > direct max 0.876 : True | R3 UDAN 0.926 < sup min 0.944 : True | R4 UDAN > otherUDA max 0.915 : True ; gap to UDAN-sup 0.038, to sup max 0.044
[8e2] R1 input=0.602 | R2 UDAN 0.898 > direct max 0.863 : True | R3 UDAN 0.898 < sup min 0.924 : True | R4 UDAN > otherUDA max 0.871 : True ; gap to UDAN-sup 0.050, to sup max 0.060
