# Resolution information-loss audit

## Frozen protocol

For every one of the 138 frozen training images, OpenCV `INTER_AREA` downsamples 4096×1024 to 1536×384 and `INTER_LINEAR` restores it to 4096×1024. Integer pixel-center behavior is OpenCV's fixed implementation; there is no PyTorch `align_corners` option. GT ROIs are centered at each GT, three times its width/height with a 32×32 minimum. A same-size, deterministic, non-overlapping offset ROI is the background control.

| Subset | Defect PSNR | Background PSNR | Defect SSIM | Background SSIM |
|---|---:|---:|---:|---:|
| All 174 GT | 36.28 | 37.80 | 0.868 | 0.866 |
| 52 tiny GT | 35.98 | 37.43 | 0.863 | 0.859 |
| Flash | 34.91 | 36.09 | 0.837 | 0.831 |
| Black | 37.48 | 39.30 | 0.896 | 0.896 |

The reconstructed defect ROIs have lower PSNR than background, so their pixels change more in an MSE sense. However, response attenuation does not show selective destruction of defect structure:

- Mean gradient attenuation: defect 36.33%, background 42.93%.
- Peak-gradient attenuation: defect 28.01%, background 42.90%.
- Absolute Laplacian attenuation: defect 78.94%, background 81.05%.
- Tiny-GT gradient attenuation: defect 37.95%, background 43.91%.
- Edge-peak displacement: defect ROI 3.24 px, background ROI 28.93 px on average.
- GT-boundary-neighborhood reconstruction MAE: 4.11 intensity levels.

The predeclared response-contrast differences are negative on average (`ΔC_gradient=-0.193`, `ΔC_laplacian=-0.112`): restoration often smooths normal texture more than the defect ROI and therefore increases, rather than decreases, the normalized defect/background response difference. PSNR/SSIM and high-frequency loss prove image alteration, not lost detection information.

Conclusion: measurable high-frequency and pixel information is lost, especially around Flash ROIs, but matched-background controls do **not** show that defect-discriminative information is preferentially lost.
