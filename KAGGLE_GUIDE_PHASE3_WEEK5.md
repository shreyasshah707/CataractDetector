# Kaggle Guide: Phase 3 Week 5 (Data Augmentation & Regularization)

## Overview
In **Phase 3 Week 5**, we upgrade the multi-disease classifier with:
1. **Medical-tuned Albumentations** (CLAHE vessel contrast enhancement, sensor shift, and occlusion cutout).
2. **Mixup & CutMix** batch-level image blending to resolve boundary confusion between Normal, DR, and Other.
3. **Model Regularization**: Classifier head dropout (`0.3`), DropPath/Stochastic Depth (`0.2`), and weight decay (`1e-4`).
4. **Weighted Soft Cross-Entropy**: Supporting soft probability targets during training with inverse-frequency class weights.
5. **Automated Benchmark Comparison**: Comparing directly against the Phase 2 Week 4 baseline (62.9% Bal.Acc, 0.537 Macro F1) and generating delta improvement charts.

---

## Step 1: Upload Notebook to Kaggle

1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. In the top menu, click **File → Import Notebook**.
3. Upload **`edd_phase3_week5.ipynb`** from your computer (`d:\Projects\EDD\edd_phase3_week5.ipynb`).

---

## Step 2: Attach Datasets & Enable GPU

1. In the right-hand panel under **Input**:
   - Click **+ Add Input**.
   - Search and add:
     - `andrewmvd/ocular-disease-recognition-odir5k`
     - `andrewmvd/retinal-disease-classification`
2. Under **Notebook settings** (right panel):
   - **Accelerator**: Select **GPU T4 x2** (or GPU P100).
   - **Internet**: Toggle **On** (required for `pip install` and `timm` weights download).
   - **Persistence**: Select **Files only** (preserves generated files).

---

## Step 3: Run the Notebook

1. **Cell 2 (Kaggle API Key)**: *(Optional)* Paste your `KAGGLE_USERNAME` and `KAGGLE_KEY` if you want automatic checkpoint uploads to your Kaggle Datasets.
2. Run **Cells 1 through 8** to initialize imports, configuration, data loaders, Albumentations transforms, and the regularized EfficientNet-B3 model.
3. **Cell 9 (`train()`)**:
   - Starts training for up to 30 epochs with early stopping (patience = 8).
   - Displays training/validation loss, balanced accuracy, and Macro F1 for every epoch.
   - Saves `checkpoint_best.pth` and `checkpoint_latest.pth` automatically after each epoch.
4. **Test Set Evaluation & Comparison**:
   - Automatically runs after training completes.
   - Displays the confusion matrix, classification report, and the **Baseline vs Week 5 Comparison Table & Bar Chart**.

---

## Step 4: Pausing & Resuming

- **To Pause**: Click the **Stop (Interrupt)** button on the running training cell.
- **To Resume in the Same Session**: Simply run the `train()` cell again. The script auto-detects `/kaggle/working/checkpoints/checkpoint_latest.pth` and resumes from the exact epoch.
- **To Resume in a New Session**:
  1. Add your checkpoint dataset or upload `checkpoint_latest.pth`.
  2. In **Cell 3 (Configuration)**, set:
     ```python
     CFG["resume_from"] = "/kaggle/input/your-checkpoint-slug/checkpoint_latest.pth"
     ```
  3. Run the notebook.

---

## Outputs Generated in `/kaggle/working/results/`

| File | Description |
| :--- | :--- |
| `summary_metrics.json` | Final Accuracy, Balanced Accuracy, and Macro F1 on the test set. |
| `classification_report.csv` | Precision, Recall, and F1-score for each of the 6 disease classes. |
| `baseline_vs_week5_comparison.csv` | Side-by-side metric comparison and deltas vs Week 4 baseline. |
| `confusion_matrix.png` | 6x6 test confusion matrix plot. |
| `augmented_vs_baseline_f1.png` | Grouped bar chart comparing per-class F1 scores (Baseline vs Augmented). |
| `training_curves.png` | Loss, Balanced Accuracy, and F1 curves over all epochs. |
| `test_predictions.csv` | Per-image true labels, predicted labels, and softmax probabilities. |
