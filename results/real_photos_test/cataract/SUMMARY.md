# Cataract Photos Test Report (`data/real_photos_test/cataract`)

**Total Samples:** 8 confirmed Cataract eyes  
**Date:** 2026-11-17 09:34:09 UTC  

## 1. Sensitivity / Recall on Cataract Photos

| Model / Pipeline | Total Cataracts | Correctly Detected (TP) | Missed (FN) | Sensitivity (Recall) |
| :--- | :---: | :---: | :---: | :---: |
| **Stage 1 (Luminance + SVM)** | 8 | 7 | 1 | **87.5%** |
| **Two-Stage Pipeline** | 8 | 7 | 1 | **87.5%** |
| **EfficientNet-B0 Binary CNN** | 8 | 1 | 7 | **12.5%** |

## 2. Per-Image Breakdown

| Image | Mean Lum. | Stage 1 SVM | Two-Stage Pipeline Output | Stage 2 Severity | B0 Binary CNN |
| :--- | :---: | :---: | :--- | :--- | :--- |
| `cataracts.jpg` | 15.99 | **PASS (Cataract)** | Cataract Detected (Immature) | Immature (56.0%) | Normal (51.9%) |
| `DS00050-IM01228-vi7-cataractthu-en.avif` | 18.67 | **PASS (Cataract)** | Cataract Detected (Mature) | Mature (51.6%) | Normal (62.1%) |
| `images (10).jpg` | 15.51 | **PASS (Cataract)** | Cataract Detected (Immature) | Immature (52.1%) | Normal (61.4%) |
| `images (7).jpg` | 14.49 | **FAIL (Missed Normal)** | No Cataract Detected (Normal) | Mature (50.9%) | Normal (53.3%) |
| `images (8).jpg` | 15.51 | **PASS (Cataract)** | Cataract Detected (Immature) | Immature (51.0%) | Normal (59.4%) |
| `images (9).jpg` | 15.61 | **PASS (Cataract)** | Cataract Detected (Immature) | Immature (56.0%) | Cataract (52.2%) |
| `overview-of-cataract-top-image.avif` | 15.55 | **PASS (Cataract)** | Cataract Detected (Immature) | Immature (56.5%) | Normal (55.8%) |
| `shutterstock_673305820-352x235.jpg` | 15.41 | **PASS (Cataract)** | Cataract Detected (Immature) | Immature (50.3%) | Normal (52.3%) |
