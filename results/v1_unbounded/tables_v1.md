## NeRF-Synthetic (per scene, mean ± std over seeds)

**PSNR** ↑

| Method | Lego | Chair | Drums | Ficus | Hotdog | Materials | Mic | Ship | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Scaffold-GS | 35.45 | 34.92 | 26.25 | 34.87 | 37.74 | 30.48 | 36.42 | 31.32 | 33.43 ± 0.02 |
| SCGC (ours) | 35.37 | 34.89 | 26.27 | 34.90 | 37.69 | 30.43 | 36.36 | 31.27 | 33.40 ± 0.04 |

**SSIM** ↑

| Method | Lego | Chair | Drums | Ficus | Hotdog | Materials | Mic | Ship | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Scaffold-GS | 0.980 | 0.985 | 0.949 | 0.985 | 0.984 | 0.959 | 0.991 | 0.901 | 0.967 ± 0.000 |
| SCGC (ours) | 0.980 | 0.985 | 0.949 | 0.985 | 0.984 | 0.959 | 0.991 | 0.899 | 0.966 ± 0.000 |

**LPIPS** ↓

| Method | Lego | Chair | Drums | Ficus | Hotdog | Materials | Mic | Ship | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Scaffold-GS | 0.018 | 0.013 | 0.046 | 0.014 | 0.023 | 0.041 | 0.008 | 0.114 | 0.035 ± 0.000 |
| SCGC (ours) | 0.019 | 0.013 | 0.047 | 0.014 | 0.022 | 0.042 | 0.009 | 0.113 | 0.035 ± 0.001 |

### SCGC − Scaffold-GS, paired by seed

| Scene | ΔPSNR | ΔSSIM | ΔLPIPS | seeds where SCGC PSNR is higher |
|---|---|---|---|---|
| Lego | -0.08 ± 0.06 | -0.000 ± 0.000 | +0.000 ± 0.000 | 0/3 |
| Chair | -0.03 ± 0.08 | -0.000 ± 0.000 | +0.000 ± 0.000 | 1/3 |
| Drums | +0.01 ± 0.08 | -0.000 ± 0.000 | +0.001 ± 0.001 | 1/3 |
| Ficus | +0.02 ± 0.17 | +0.000 ± 0.000 | -0.000 ± 0.000 | 2/3 |
| Hotdog | -0.04 ± 0.07 | +0.000 ± 0.000 | -0.001 ± 0.000 | 1/3 |
| Materials | -0.05 ± 0.05 | -0.000 ± 0.000 | +0.001 ± 0.001 | 0/3 |
| Mic | -0.06 ± 0.07 | -0.000 ± 0.000 | +0.000 ± 0.001 | 0/3 |
| Ship | -0.05 ± 0.10 | -0.002 ± 0.001 | -0.001 ± 0.001 | 1/3 |
| **Average** | -0.03 ± 0.04 | -0.000 ± 0.000 | +0.000 ± 0.000 | 0/3 |

## Mip-NeRF 360

| Method | bicycle | garden | stump | flowers | treehill | room | counter | kitchen | bonsai | Avg PSNR | Avg SSIM | Avg LPIPS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Scaffold-GS | 25.20 | 27.48 | 26.61 | 21.44 | 23.09 | 32.02 | 29.51 | 31.60 | 32.58 | 27.73 | 0.815 | 0.220 |
| SCGC (ours) | 21.14 | 27.40 | 26.66 | 21.25 | 23.03 | 32.08 | 29.47 | 31.39 | 32.65 | 27.23 | 0.782 | 0.243 |

(per-scene cells are PSNR; one seed)

## Tanks & Temples

| Method | truck | train | Avg PSNR | Avg SSIM | Avg LPIPS |
|---|---|---|---|---|---|
| Scaffold-GS | 25.78 | 22.40 | 24.09 | 0.854 | 0.174 |
| SCGC (ours) | 25.75 | 12.44 | 19.10 | 0.658 | 0.390 |

(per-scene cells are PSNR; one seed)

## Deep Blending

| Method | playroom | drjohnson | Avg PSNR | Avg SSIM | Avg LPIPS |
|---|---|---|---|---|---|
| Scaffold-GS | 30.86 | 29.72 | 30.29 | 0.909 | 0.253 |
| SCGC (ours) | 30.74 | 29.66 | 30.20 | 0.908 | 0.253 |

(per-scene cells are PSNR; one seed)

# Ablations (NeRF-Synthetic, 8-scene average, seed 0)

## Progressive components (paper Table 5)

| Config | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Scaffold-GS | 33.41 | 0.967 | 0.035 |
| + Control Points | 33.39 | 0.967 | 0.035 |
| + Transformer | 33.41 | 0.967 | 0.035 |
| + Control Contrast | 33.42 | 0.967 | 0.035 |
| SCGC (ours) | 33.39 | 0.967 | 0.034 |

## Number of control points (paper Table 2)

| Config | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 256 pts | 33.41 | 0.967 | 0.034 |
| SCGC (ours) | 33.39 | 0.967 | 0.034 |
| 1024 pts | 33.38 | 0.967 | 0.034 |

## Transformer size (paper Table 3)

| Config | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 1L-2H | 33.38 | 0.967 | 0.034 |
| SCGC (ours) | 33.39 | 0.967 | 0.034 |
| 4L-4H | 33.39 | 0.967 | 0.034 |
| 2L-8H | 33.41 | 0.967 | 0.034 |

## Loss weights λ1 / λ2 / λ3 (paper Table 4; paper default 0.10 / 0.05 / 0.02, ours 0.10 / 0.05 / 0.002)

| Config | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 0.01 / 0.01 / 0.01 | 32.88 | 0.965 | 0.035 |
| SCGC, paper λ3 = 0.02 | 31.38 | 0.962 | 0.045 |
| 0.15 / 0.03 / 0.01 | 32.89 | 0.965 | 0.035 |
| 0.30 / 0.10 / 0.05 | 29.04 | 0.954 | 0.065 |
| SCGC (ours) | 33.39 | 0.967 | 0.034 |
