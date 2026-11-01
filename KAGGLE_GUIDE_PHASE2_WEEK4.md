# EDD – Phase 2, Week 4: Kaggle Setup Guide
## Multi-Disease Eye Classifier (ODIR-2019 + RFMiD 2.0)

---

## Context

| Phase | Week | Task | Status |
|-------|------|------|--------|
| Phase 2 | Week 3 | DR Baseline (APTOS, ResNet-50) | Done — 79.78% acc, 0.63 F1 |
| Phase 2 | **Week 4** | **Multi-Disease (ODIR+RFMiD, EfficientNet-B3)** | **← You are here** |
| Phase 3 | Week 5 | Augmentation & Regularization | Upcoming |

**Target**: >80% Balanced Accuracy on 6-class problem (Normal, DR, Glaucoma, Cataract, AMD, Other)

---

## Step 1 — Add Datasets to Kaggle

Open your notebook and click **"+ Add Data"** for each:

| Dataset | Kaggle Slug | Notes |
|---------|------------|-------|
| ODIR-2019 | `andrewmvd/ocular-disease-recognition-odir5k` | ~8,000 fundus images |
| RFMiD 2.0 | `andrewmvd/retinal-disease-classification` | ~5,639 images |

After adding, your input paths will be:
```
/kaggle/input/ocular-disease-recognition-odir5k/
/kaggle/input/retinal-disease-classification/
```

**Update CFG in the script accordingly:**
```python
CFG = {
    "odir_root":  "/kaggle/input/ocular-disease-recognition-odir5k",
    "rfmid_root": "/kaggle/input/retinal-disease-classification",
    ...
}
```

---

## Step 2 — Upload the Script

Option A — Paste into a notebook cell (recommended):
1. Create a new Kaggle notebook
2. Paste the entire contents of `kaggle_phase2_week4.py` into a code cell
3. Set **Accelerator → GPU T4 x2** (or P100)
4. Set **Persistence → Files only** (critical for checkpoints)

Option B — Upload as a dataset:
1. Upload `kaggle_phase2_week4.py` as a Kaggle dataset
2. Add it as input and run: `exec(open("/kaggle/input/your-script/kaggle_phase2_week4.py").read())`

---

## Step 3 — First Run (Fresh Start)

Make sure `CFG["resume_from"] = None` (default).

The script will auto-detect no prior checkpoint and train from epoch 1.

Checkpoint files are saved to `/kaggle/working/checkpoints/`:
```
checkpoints/
  checkpoint_latest.pth       <- always updated (safe resume point)
  checkpoint_best.pth         <- best val balanced accuracy so far
  checkpoint_epoch_005.pth    <- every 5 epochs (extra safety)
  checkpoint_epoch_010.pth
  training_state.json         <- human-readable: last epoch, best metric
```

---

## Step 4 — Resuming After a Kaggle Session Limit

When Kaggle's 12-hour GPU session cuts you off:

### 4a. Save the checkpoint output

After the session ends, go to **Output → Files** and download:
- `checkpoints/checkpoint_latest.pth` (or `checkpoint_best.pth`)

### 4b. Upload checkpoint as a dataset

1. Go to **Datasets → New Dataset**
2. Upload the `.pth` file
3. Name it `edd-phase2-ckpt`
4. Make it private

### 4c. Resume in the new session

Add `edd-phase2-ckpt` as input to your new notebook, then change:
```python
CFG["resume_from"] = "/kaggle/input/edd-phase2-ckpt/checkpoint_latest.pth"
```

The script will:
- Load all model weights
- Restore optimizer & scheduler states
- Continue from the exact epoch where it stopped
- Preserve the full training history for plotting

**Alternatively**, if you saved the checkpoint to `/kaggle/working/` in the same
session and are rerunning the same notebook, the script auto-detects it —
no config change needed.

---

## Step 5 — Monitor Training

Each epoch writes to `/kaggle/working/logs/epoch_NNN.json`:
```json
{
  "epoch": 7,
  "train": {"loss": 0.82, "acc": 0.74, "bal_acc": 0.71, "f1": 0.69},
  "val":   {"loss": 0.91, "acc": 0.73, "bal_acc": 0.70, "f1": 0.68},
  "lr": 9.5e-05,
  "elapsed_s": 312
}
```

Quick check from a notebook cell:
```python
import json, glob
logs = sorted(glob.glob("/kaggle/working/logs/*.json"))
for f in logs[-5:]:
    d = json.load(open(f))
    print(f"Ep{d['epoch']:>3}  val_bal_acc={d['val']['bal_acc']:.4f}  f1={d['val']['f1']:.4f}")
```

---

## Step 6 — Results

After training completes, outputs are in `/kaggle/working/results/`:

| File | Description |
|------|------------|
| `summary_metrics.json` | Balanced accuracy + macro F1 on test set |
| `classification_report.csv` | Per-class precision/recall/F1 |
| `test_predictions.csv` | Per-image predictions + class probabilities |
| `confusion_matrix.png` | 6x6 confusion matrix heatmap |
| `per_class_f1.png` | Per-class F1 bar chart (green=good, orange=ok, red=poor) |
| `training_curves.png` | Loss / Balanced Acc / F1 curves |

---

## Expected Performance

| Metric | Target | Baseline (Week 3 ResNet-50 DR only) |
|--------|--------|--------------------------------------|
| Balanced Accuracy | >80% | 79.78% (5-class DR) |
| Macro F1 | >0.80 | 0.63 |
| Hardest class | Cataract/AMD | Grade_1 DR |

EfficientNet-B3 + weighted loss + CLAHE augmentation should comfortably
exceed the 80% balanced accuracy target on this 6-class problem.

---

## Troubleshooting

**"Cannot find ODIR label CSV"**
→ Check exact folder structure after adding dataset. Update `load_odir_dataframe()`
  `csv_candidates` list with the correct path.

**"Cannot locate ODIR image directory"**
→ Add your image folder path to the `img_dirs` list in `load_odir_dataframe()`.

**Out of memory on GPU**
→ Reduce `CFG["batch_size"]` to 16 or `CFG["img_size"]` to 260.

**Training too slow**
→ Increase `CFG["num_workers"]` to 4 and ensure GPU accelerator is enabled.

**Import errors**
→ The script auto-installs `timm` and `albumentations`. If still failing,
  add a cell before: `!pip install timm albumentations -q`
