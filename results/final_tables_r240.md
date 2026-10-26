# Final tables (ROI=r240, test=n30 clean 200Views @1e3)

## Table 2 (main comparison, I0=1e3)

| Method | Paradigm | SSIM | PSNR (dB) | RMSE |
|---|---|---|---|---|
| Low-flux FDK input (I0=1e3) | input | 0.624±0.050 & 30.44±2.22 & 0.0312±0.0099 |
| SART (early-stopped) | iterative | 0.790±0.024 & 31.50±1.23 & 0.0269±0.0044 |
| DnCNN-Xray [direct transfer] | X-ray supervised | 0.829±0.042 & 35.26±2.04 & 0.0178±0.0052 |
| REDCNN-Xray [direct transfer] | X-ray supervised | 0.833±0.062 & 32.91±1.66 & 0.0231±0.0052 |
| FBPConvNet-Xray [direct transfer] | X-ray supervised | 0.876±0.024 & 37.97±1.78 & 0.0129±0.0032 |
| UDAN arch w/o adaptation [direct transfer] | X-ray supervised | 0.781±0.032 & 34.54±1.38 & 0.0190±0.0035 |
| DnCNN-MMD [label-free UDA] | label-free UDA | 0.915±0.043 & 37.03±2.50 & 0.0148±0.0058 |
| REDCNN-MMD [label-free UDA] | label-free UDA | 0.863±0.042 & 33.99±1.72 & 0.0204±0.0048 |
| FBPConvNet-MMD [label-free UDA] | label-free UDA | 0.810±0.015 & 36.24±1.55 & 0.0157±0.0034 |
| UDAN (proposed) [label-free UDA] | label-free UDA | 0.926±0.032 & 38.96±2.34 & 0.0118±0.0042 |
| DnCNN-sup [NCT supervised] | NCT supervised | 0.944±0.028 & 39.39±2.06 & 0.0111±0.0033 |
| REDCNN-sup [NCT supervised] | NCT supervised | 0.969±0.013 & 43.52±1.93 & 0.0069±0.0019 |
| FBPConvNet-sup [NCT supervised] | NCT supervised | 0.970±0.013 & 43.50±1.97 & 0.0069±0.0019 |
| UDAN-sup [supervised upper bound] | NCT supervised | 0.964±0.017 & 42.15±2.20 & 0.0081±0.0027 |
