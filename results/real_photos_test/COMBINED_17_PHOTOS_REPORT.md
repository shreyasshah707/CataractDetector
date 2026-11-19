# EDD v2 — Full Benchmark Report on Real Eye Photos

**Total Real Photos Evaluated:** 17 (9 Normal + 8 Cataract)
**Report Generated:** 2026-11-17 09:34:09 UTC

## 1. Overall Performance Comparison

| Architecture | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1 (Luminance + SVM)** | **88.2%** (15/17) | **87.5%** (7/8) | **88.9%** (8/9) | **87.5%** | **87.5%** |
| **Two-Stage Pipeline** | **88.2%** (15/17) | **87.5%** (7/8) | **88.9%** (8/9) | **87.5%** | **87.5%** |
| **EfficientNet-B0 Binary CNN** | **41.2%** (7/17) | **12.5%** (1/8) | **66.7%** (6/9) | **25.0%** | **16.7%** |

## 2. Confusion Matrices

### Stage 1 (Luminance + SVM):
- **True Negatives (Normal correctly identified):** 8 / 9
- **False Positives (Normal falsely flagged as Cataract):** 1 / 9
- **True Positives (Cataract correctly identified):** 7 / 8
- **False Negatives (Cataract missed):** 1 / 8

### EfficientNet-B0 Binary CNN:
- **True Negatives (Normal correctly identified):** 6 / 9
- **False Positives (Normal falsely flagged as Cataract):** 3 / 9
- **True Positives (Cataract correctly identified):** 1 / 8
- **False Negatives (Cataract missed):** 7 / 8

