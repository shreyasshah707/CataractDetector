"""
===============================================================================
EDD Phase 3 - Week 5: Data Augmentation & Regularization
===============================================================================
Project   : Eye Disease Detection (EDD)
Task      : Phase 3 Week 5 - Multi-Disease Classifier with Advanced Augmentation
Backbone  : EfficientNet-B3 (with Classifier Dropout & DropPath Regularization)
Techniques:
  1. Medical-tuned Albumentations (CLAHE, sensor shift, occlusion dropout)
  2. Batch-level Mixup & CutMix with soft target smoothing
  3. Classifier dropout (0.3) + DropPath / Stochastic Depth (0.2) + Weight Decay (1e-4)
  4. Weighted Cross-Entropy supporting soft probability targets
  5. Automated comparison against Phase 2 Week 4 baseline (62.9% Bal.Acc, 0.537 F1)
  6. PyTorch 2.6+ safe checkpoint loading (weights_only=False)
===============================================================================
"""

import os
import sys
import json
import time
import random
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import timm
from timm.data import Mixup

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score, accuracy_score
)

warnings.filterwarnings("ignore")

# --- 1. Configuration --------------------------------------------------------

CFG = {
    # Dataset input paths
    "odir_root":   "/kaggle/input/datasets/andrewmvd/ocular-disease-recognition-odir5k",
    "rfmid_root":  "/kaggle/input/datasets/andrewmvd/retinal-disease-classification",

    # Output directories
    "output_dir":  "/kaggle/working",
    "log_dir":     "/kaggle/working/logs",
    "ckpt_dir":    "/kaggle/working/checkpoints",
    "result_dir":  "/kaggle/working/results",

    # Resume checkpoint path (None for fresh training or auto-detect latest)
    "resume_from": None,

    # Model architecture & regularization
    "model_name":     "efficientnet_b3",
    "num_classes":    6,
    "img_size":       300,
    "pretrained":     True,
    "drop_rate":      0.3,   # Classifier head dropout
    "drop_path_rate": 0.2,   # Stochastic depth / drop path

    # Data Augmentation & Mixup / CutMix
    "use_mixup":       True,
    "mixup_alpha":     0.8,   # Beta distribution param for Mixup
    "cutmix_alpha":    1.0,   # Beta distribution param for CutMix
    "mixup_prob":      0.6,   # Probability of applying mixup or cutmix per batch
    "switch_prob":     0.5,   # Probability of switching between mixup and cutmix
    "label_smoothing": 0.1,

    # Training
    "epochs":              30,
    "batch_size":          32,
    "num_workers":         2,
    "lr":                  1e-4,
    "weight_decay":        1e-4,
    "scheduler":           "cosine",
    "T_0":                 10,
    "step_size":           10,
    "gamma":               0.1,
    "early_stop_patience": 8,
    "grad_clip":           1.0,
    "mixed_precision":     True,
    "seed":                42,

    # Phase 2 Week 4 Baseline Reference Metrics (for benchmark comparison)
    "baseline_bal_acc":  0.6290,
    "baseline_macro_f1": 0.5369,
    "baseline_acc":      0.5701,
    "baseline_per_class_f1": {
        "Normal":   0.533,
        "DR":       0.675,
        "Glaucoma": 0.496,
        "Cataract": 0.432,
        "AMD":      0.512,
        "Other":    0.573,
    },
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Normal", "DR", "Glaucoma", "Cataract", "AMD", "Other"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG["seed"])

for d in [CFG["log_dir"], CFG["ckpt_dir"], CFG["result_dir"]]:
    Path(d).mkdir(parents=True, exist_ok=True)

print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"timm   : {timm.__version__}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# --- 2. Dataset Preparation (ODIR-2019 + RFMiD 2.0) ---------------------------

def load_odir_dataframe(odir_root):
    """
    Parse ODIR-2019 into a flat DataFrame with columns: filepath, label.
    Supports nested Kaggle directory structures, binary columns, and text diagnoses.
    """
    root = Path(odir_root)

    csv_candidates = [
        root / "ODIR-5K" / "ODIR-5K" / "data.xlsx",
        root / "ODIR-5K" / "ODIR-5K" / "ODIR-5K_Training_Annotations(Updated)_V2.xlsx",
        root / "full_df.csv",
        root / "data.xlsx",
        root / "ODIR-5K_Training_Annotations(Updated)_V2.xlsx",
        root / "training.csv",
        root / "data" / "training.csv",
        root / "ODIR-5K" / "training.csv",
        root / "ODIR-5K" / "data.xlsx",
    ]

    label_path = None
    for c in csv_candidates:
        if c.exists():
            label_path = c
            break

    if label_path is None and root.exists():
        all_tables = []
        for ext in ["*.csv", "*.xlsx", "*.xls"]:
            all_tables.extend(list(root.rglob(ext)))
        if all_tables:
            relevant = [f for f in all_tables if any(k in f.name.lower() for k in ["data", "train", "annot", "full", "odir"])]
            label_path = relevant[0] if relevant else all_tables[0]

    if label_path is None or not label_path.exists():
        raise FileNotFoundError(
            f"Could not find ODIR label file in {odir_root}. "
            "Check your Kaggle dataset slug and file layout."
        )

    if label_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(label_path)
    else:
        df = pd.read_csv(label_path)
    print(f"[ODIR] Loaded labels from {label_path}")

    img_dir_candidates = [
        root / "ODIR-5K" / "ODIR-5K" / "Training Images",
        root / "ODIR-5K" / "Training Images",
        root / "preprocessed_images",
        root / "images",
        root / "ODIR-5K_Training_Dataset",
        root / "train_images",
        root / "Training Images",
    ]

    img_dir = next((d for d in img_dir_candidates if d.exists() and any(d.iterdir())), None)

    if img_dir is None and root.exists():
        for p in root.rglob("*"):
            if p.is_dir():
                has_imgs = any(f.suffix.lower() in [".jpg", ".jpeg", ".png"] for f in p.glob("*") if f.is_file())
                if has_imgs:
                    img_dir = p
                    break

    if img_dir is None:
        raise FileNotFoundError(f"Cannot locate ODIR image directory under {odir_root}")
    print(f"[ODIR] Found images directory at {img_dir}")

    left_col = next((c for c in df.columns if "left" in c.lower() and ("fundus" in c.lower() or "filename" in c.lower() or "image" in c.lower())), None)
    right_col = next((c for c in df.columns if "right" in c.lower() and ("fundus" in c.lower() or "filename" in c.lower() or "image" in c.lower())), None)
    single_col = next((c for c in df.columns if c.lower() in ["filename", "filepath", "file", "image", "img_path"]), None) if not (left_col or right_col) else None

    diag_left_col = next((c for c in df.columns if "left" in c.lower() and ("keyword" in c.lower() or "diagnos" in c.lower() or "label" in c.lower())), None)
    diag_right_col = next((c for c in df.columns if "right" in c.lower() and ("keyword" in c.lower() or "diagnos" in c.lower() or "label" in c.lower())), None)
    diag_gen_col = next((c for c in df.columns if "keyword" in c.lower() or "label" in c.lower() or "diagnos" in c.lower() or "target" in c.lower()), None)

    has_binary_flags = all(col in df.columns for col in ["N", "D", "G", "C", "A"])

    records = []
    pairings = []
    if left_col or right_col:
        if left_col:
            pairings.append((left_col, diag_left_col or diag_gen_col))
        if right_col:
            pairings.append((right_col, diag_right_col or diag_gen_col))
    elif single_col:
        pairings.append((single_col, diag_gen_col))

    for _, row in df.iterrows():
        for img_col, diag_col in pairings:
            fname = str(row[img_col]).strip() if pd.notna(row.get(img_col)) else None
            if not fname or fname.lower() == "nan":
                continue

            img_path = img_dir / fname
            if not img_path.exists():
                img_path = root / fname
            if not img_path.exists():
                for ext in [".jpg", ".jpeg", ".png"]:
                    candidate = (img_dir / fname).with_suffix(ext)
                    if candidate.exists():
                        img_path = candidate
                        break
                    candidate_root = (root / fname).with_suffix(ext)
                    if candidate_root.exists():
                        img_path = candidate_root
                        break
            if not img_path.exists():
                fname_only = Path(fname).name
                candidate_name = img_dir / fname_only
                if candidate_name.exists():
                    img_path = candidate_name

            if not img_path.exists():
                continue

            mapped = "Other"
            if has_binary_flags:
                if row.get("N", 0) == 1:
                    mapped = "Normal"
                elif row.get("D", 0) == 1:
                    mapped = "DR"
                elif row.get("G", 0) == 1:
                    mapped = "Glaucoma"
                elif row.get("C", 0) == 1:
                    mapped = "Cataract"
                elif row.get("A", 0) == 1:
                    mapped = "AMD"
                elif row.get("M", 0) == 1 or row.get("H", 0) == 1 or row.get("O", 0) == 1:
                    mapped = "Other"
            else:
                raw_label = str(row.get(diag_col, "O")) if diag_col else "O"
                raw_upper = raw_label.upper()
                if "NORMAL" in raw_upper or "N" in raw_upper.split():
                    mapped = "Normal"
                elif "DIABET" in raw_upper or "DR" in raw_upper.split() or "D" in raw_upper.split():
                    mapped = "DR"
                elif "GLAUCOMA" in raw_upper or "G" in raw_upper.split():
                    mapped = "Glaucoma"
                elif "CATARACT" in raw_upper or "C" in raw_upper.split():
                    mapped = "Cataract"
                elif "MACULAR" in raw_upper or "AMD" in raw_upper.split() or "A" in raw_upper.split():
                    mapped = "AMD"
                else:
                    mapped = "Other"

            records.append({"filepath": str(img_path), "label": mapped, "source": "ODIR"})

    return pd.DataFrame(records)


def load_rfmid_dataframe(rfmid_root):
    """
    Parse RFMiD 2.0 into a flat DataFrame mapped to our 6-class scheme.
    """
    root = Path(rfmid_root)

    splits = {
        "train": [
            (root / "Training_Set" / "Training_Set" / "RFMiD_Training_Labels.csv", root / "Training_Set" / "Training_Set" / "Training"),
            (root / "RFMiD_Training_Labels.csv", root / "Training"),
            (root / "Training_Set" / "RFMiD_Training_Labels.csv", root / "Training_Set" / "Training"),
        ],
        "val": [
            (root / "Validation_Set" / "Validation_Set" / "RFMiD_Validation_Labels.csv", root / "Validation_Set" / "Validation_Set" / "Validation"),
            (root / "RFMiD_Validation_Labels.csv", root / "Validation"),
            (root / "Validation_Set" / "RFMiD_Validation_Labels.csv", root / "Validation_Set" / "Validation"),
        ],
        "test": [
            (root / "Test_Set" / "Test_Set" / "RFMiD_Test_Labels.csv", root / "Test_Set" / "Test_Set" / "Test"),
            (root / "RFMiD_Test_Labels.csv", root / "Test"),
            (root / "Test_Set" / "RFMiD_Test_Labels.csv", root / "Test_Set" / "Test"),
        ],
    }

    rfmid_map = {
        "DR": "DR",
        "ARMD": "AMD",
        "ODC": "Glaucoma", "ODP": "Glaucoma",
        "TSLN": "Other", "MH": "Other", "DN": "Other", "DMED": "Other",
        "BRVO": "Other", "CRVO": "Other", "TV": "Other", "AH": "Other",
        "RP": "Other", "CRS": "Other", "RS": "Other", "ERM": "Other",
        "LS": "Other", "MS": "Other", "CSR": "Other", "HTR": "Other",
        "AION": "Other", "PT": "Other", "BRAO": "Other",
    }

    records = []
    for split_name, candidate_list in splits.items():
        csv_path, img_dir = None, None
        for c_csv, c_img in candidate_list:
            if c_csv.exists():
                csv_path, img_dir = c_csv, c_img
                break

        if csv_path is None and root.exists():
            for f in root.rglob("*.csv"):
                if split_name.lower() in f.name.lower() or ("label" in f.name.lower() and split_name[:3].lower() in f.name.lower()):
                    csv_path = f
                    parent = f.parent
                    for p in [parent, parent.parent, parent / split_name.capitalize(), parent / "Training", parent / "Validation", parent / "Test"]:
                        if p.exists() and any(p.glob("*.[jp][pn]g")):
                            img_dir = p
                            break
                    if img_dir is None:
                        for p in root.rglob("*"):
                            if p.is_dir() and any(p.glob("*.[jp][pn]g")):
                                img_dir = p
                                break
                    break

        if csv_path is None or not csv_path.exists():
            print(f"[RFMiD] WARNING: Cannot find {split_name} CSV -- skipping")
            continue
        df = pd.read_csv(csv_path)
        if img_dir is None or not img_dir.exists():
            print(f"[RFMiD] WARNING: Cannot find image dir for {split_name} -- skipping")
            continue

        disease_cols = [c for c in df.columns if c.upper() in rfmid_map]
        normal_col   = next(
            (c for c in df.columns if c.upper() in ("DISEASE_RISK", "NORMAL", "N")), None
        )

        for _, row in df.iterrows():
            img_id   = str(row["ID"]) if "ID" in row else str(row.iloc[0])
            img_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = img_dir / f"{img_id}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            mapped = "Normal"
            for col in disease_cols:
                if row.get(col, 0) == 1:
                    mapped = rfmid_map.get(col.upper(), "Other")
                    break
            if normal_col and row.get(normal_col, 1) == 0 and mapped == "Normal":
                mapped = "Normal"

            records.append({"filepath": str(img_path), "label": mapped, "source": "RFMiD"})

    return pd.DataFrame(records)


def build_combined_dataframe(odir_root, rfmid_root):
    """Load, merge, encode, and stratified-split ODIR + RFMiD."""
    print("\n[Data] Loading ODIR-2019 ...")
    df_odir  = load_odir_dataframe(odir_root)
    print(f"       ODIR rows : {len(df_odir)}")

    print("[Data] Loading RFMiD 2.0 ...")
    df_rfmid = load_rfmid_dataframe(rfmid_root)
    print(f"       RFMiD rows: {len(df_rfmid)}")

    df = pd.concat([df_odir, df_rfmid], ignore_index=True)
    print(f"[Data] Combined  : {len(df)} images")

    df["label_idx"] = df["label"].map(CLASS_TO_IDX)
    df = df.dropna(subset=["label_idx"]).copy()
    df["label_idx"] = df["label_idx"].astype(int)

    print("\n[Data] Class distribution:")
    for cls in CLASS_NAMES:
        cnt = (df["label"] == cls).sum()
        print(f"       {cls:12s}: {cnt:5d}  ({cnt/len(df)*100:.1f}%)")

    from sklearn.model_selection import train_test_split
    train_dfs, val_dfs, test_dfs = [], [], []
    for cls_idx in range(CFG["num_classes"]):
        cls_df = df[df["label_idx"] == cls_idx]
        if len(cls_df) < 6:
            train_dfs.append(cls_df)
            continue
        tr, tmp = train_test_split(cls_df, test_size=0.30, random_state=CFG["seed"])
        va, te  = train_test_split(tmp,    test_size=0.50, random_state=CFG["seed"])
        train_dfs.append(tr)
        val_dfs.append(va)
        test_dfs.append(te)

    df_train = pd.concat(train_dfs).reset_index(drop=True)
    df_val   = pd.concat(val_dfs).reset_index(drop=True)
    df_test  = pd.concat(test_dfs).reset_index(drop=True)

    print(f"\n[Data] Split -> train:{len(df_train)}  val:{len(df_val)}  test:{len(df_test)}")
    return df_train, df_val, df_test


def compute_class_weights(df_train):
    """Inverse-frequency weights for WeightedCrossEntropy."""
    counts  = np.bincount(df_train["label_idx"], minlength=CFG["num_classes"]).astype(float)
    counts  = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * CFG["num_classes"]
    print("\n[Loss] Class weights:", {CLASS_NAMES[i]: f"{weights[i]:.3f}" for i in range(CFG["num_classes"])})
    return torch.tensor(weights, dtype=torch.float32)


# --- 3. Enhanced Albumentations & DataLoaders ---------------------------------

def get_transforms(split, img_size):
    """
    Phase 3 Week 5: Medical Fundus Augmentation Pipeline.
    - CLAHE enhancements for retinal vascular contrast
    - Color and brightness shifts for camera sensor variance
    - Geometric transforms (rotations, flips, affine)
    - CoarseDropout to simulate retinal occlusions, artifacts, and shadows
    """
    if split == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.12, rotate_limit=45, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.7),
            A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=15, p=0.4),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),
            ], p=0.3),
            A.CoarseDropout(
                max_holes=6, max_height=24, max_width=24,
                min_holes=1, min_height=8, min_width=8, p=0.3
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class EyeDiseaseDataset(Dataset):
    def __init__(self, df, split, img_size):
        self.df        = df.reset_index(drop=True)
        self.transform = get_transforms(split, img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["filepath"]).convert("RGB"))
        img = self.transform(image=img)["image"]
        return img, int(row["label_idx"])


def build_dataloaders(df_train, df_val, df_test):
    ds_train = EyeDiseaseDataset(df_train, "train", CFG["img_size"])
    ds_val   = EyeDiseaseDataset(df_val,   "val",   CFG["img_size"])
    ds_test  = EyeDiseaseDataset(df_test,  "test",  CFG["img_size"])

    kw = dict(num_workers=CFG["num_workers"], pin_memory=True)
    loader_train = DataLoader(ds_train, batch_size=CFG["batch_size"], shuffle=True,  drop_last=True, **kw)
    loader_val   = DataLoader(ds_val,   batch_size=CFG["batch_size"], shuffle=False, **kw)
    loader_test  = DataLoader(ds_test,  batch_size=CFG["batch_size"], shuffle=False, **kw)
    return loader_train, loader_val, loader_test


# --- 4. Model Architecture & Regularization -----------------------------------

def build_model():
    """
    Builds EfficientNet-B3 with:
    - Classifier head dropout (drop_rate=0.3)
    - DropPath / Stochastic depth (drop_path_rate=0.2)
    """
    print(f"\n[Model] Initializing {CFG['model_name']} ...")
    print(f"        Head Dropout   : {CFG['drop_rate']}")
    print(f"        DropPath Rate  : {CFG['drop_path_rate']}")
    print(f"        Classes        : {CFG['num_classes']}")

    model = timm.create_model(
        CFG["model_name"],
        pretrained=CFG["pretrained"],
        num_classes=CFG["num_classes"],
        drop_rate=CFG["drop_rate"],
        drop_path_rate=CFG["drop_path_rate"],
    )
    return model.to(DEVICE)


# --- 5. Checkpoint System (PyTorch 2.6+ Safe) ---------------------------------

CHECKPOINT_META = Path(CFG["ckpt_dir"]) / "training_state.json"


def save_checkpoint(model, optimizer, scheduler, scaler,
                    epoch, best_bal_acc, best_f1, history, is_best=False):
    """
    Saves training state after every epoch:
    - checkpoint_latest.pth   : overwrite every epoch
    - checkpoint_epoch_NNN.pth: saved every 5 epochs
    - checkpoint_best.pth     : saved whenever val balanced accuracy improves
    - training_state.json     : lightweight JSON inspection
    """
    state = {
        "epoch":            epoch,
        "best_bal_acc":     best_bal_acc,
        "best_f1":          best_f1,
        "history":          history,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict() if scheduler else None,
        "scaler_state":     scaler.state_dict()    if scaler    else None,
        "cfg":              {k: v for k, v in CFG.items()},
        "timestamp":        datetime.utcnow().isoformat(),
    }

    latest_path = Path(CFG["ckpt_dir"]) / "checkpoint_latest.pth"
    torch.save(state, latest_path)

    if epoch % 5 == 0 or is_best:
        epoch_path = Path(CFG["ckpt_dir"]) / f"checkpoint_epoch_{epoch:03d}.pth"
        shutil.copy(latest_path, epoch_path)

    if is_best:
        best_path = Path(CFG["ckpt_dir"]) / "checkpoint_best.pth"
        shutil.copy(latest_path, best_path)
        print(f"  [*] Best model saved (bal_acc={best_bal_acc:.4f}  f1={best_f1:.4f})")

    meta = {
        "last_epoch":   epoch,
        "best_bal_acc": round(float(best_bal_acc), 6),
        "best_f1":      round(float(best_f1), 6),
        "timestamp":    state["timestamp"],
    }
    CHECKPOINT_META.write_text(json.dumps(meta, indent=2))


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    """Load a checkpoint with weights_only=False for PyTorch 2.6+ compatibility."""
    print(f"\n[Checkpoint] Resuming from: {path}")
    try:
        state = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=DEVICE)

    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if scheduler and state.get("scheduler_state"):
        scheduler.load_state_dict(state["scheduler_state"])
    if scaler and state.get("scaler_state"):
        scaler.load_state_dict(state["scaler_state"])

    start_epoch  = state["epoch"] + 1
    best_bal_acc = float(state.get("best_bal_acc", 0.0))
    best_f1      = float(state.get("best_f1", 0.0))
    history      = state.get("history", [])
    print(f"[Checkpoint] Resuming at Epoch {start_epoch} (Best Val Bal.Acc: {best_bal_acc:.4f})")
    return start_epoch, best_bal_acc, best_f1, history


# --- 6. Training & Validation Loops with Mixup / CutMix ------------------------

def run_epoch(model, loader, criterion, optimizer, scaler, is_train, mixup_fn=None):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if is_train and mixup_fn is not None:
                # Apply Mixup / CutMix on training batch
                imgs, targets = mixup_fn(imgs, labels)
            else:
                targets = labels

            with torch.amp.autocast("cuda", enabled=(CFG["mixed_precision"] and DEVICE == "cuda")):
                logits = model(imgs)
                loss   = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                    optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)

            # Accuracy & metrics: compare against original discrete labels
            eval_targets = targets.argmax(dim=1) if (is_train and mixup_fn is not None and targets.ndim == 2) else labels
            correct += (preds == eval_targets).sum().item()
            total   += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(eval_targets.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, bal_acc, macro_f1


# --- 7. Main Training Orchestrator --------------------------------------------

def train():
    print("\n" + "=" * 70)
    print("  EDD Phase 3 - Week 5: Data Augmentation & Regularization")
    print("=" * 70)

    # 1. Prepare Data
    df_train, df_val, df_test = build_combined_dataframe(CFG["odir_root"], CFG["rfmid_root"])

    splits_path = Path(CFG["output_dir"]) / "data_splits.json"
    splits_path.write_text(json.dumps({
        "train": df_train[["filepath", "label", "label_idx"]].to_dict("records"),
        "val":   df_val[["filepath",   "label", "label_idx"]].to_dict("records"),
        "test":  df_test[["filepath",  "label", "label_idx"]].to_dict("records"),
    }, indent=2))

    loader_train, loader_val, loader_test = build_dataloaders(df_train, df_val, df_test)
    class_weights = compute_class_weights(df_train).to(DEVICE)

    # 2. Setup Mixup / CutMix
    mixup_fn = None
    if CFG["use_mixup"]:
        print("\n[Augmentation] Initializing Mixup & CutMix ...")
        print(f"               Mixup Alpha   : {CFG['mixup_alpha']}")
        print(f"               CutMix Alpha  : {CFG['cutmix_alpha']}")
        print(f"               Batch Prob    : {CFG['mixup_prob']}")
        print(f"               Label Smooth  : {CFG['label_smoothing']}")
        mixup_fn = Mixup(
            mixup_alpha=CFG["mixup_alpha"],
            cutmix_alpha=CFG["cutmix_alpha"],
            prob=CFG["mixup_prob"],
            switch_prob=CFG["switch_prob"],
            mode="batch",
            label_smoothing=CFG["label_smoothing"],
            num_classes=CFG["num_classes"],
        )

    # 3. Model, Loss, Optimizer, Scheduler
    model     = build_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    if CFG["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG["T_0"])
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CFG["step_size"], gamma=CFG["gamma"])

    scaler = (torch.amp.GradScaler("cuda")
              if CFG["mixed_precision"] and DEVICE == "cuda" else None)

    # 4. Resume Handling
    start_epoch  = 1
    best_bal_acc = 0.0
    best_f1      = 0.0
    history      = []
    no_improve   = 0

    resume_path = CFG.get("resume_from")
    if resume_path is None:
        latest_ckpt = Path(CFG["ckpt_dir"]) / "checkpoint_latest.pth"
        if latest_ckpt.exists():
            resume_path = str(latest_ckpt)
            print(f"[Checkpoint] Auto-detected previous run: {resume_path}")

    if resume_path and Path(resume_path).exists():
        start_epoch, best_bal_acc, best_f1, history = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler
        )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n[Train] Epochs     : {start_epoch} -> {CFG['epochs']}")
    print(f"[Train] Parameters : {total_params:.1f}M")
    print(f"[Train] Batch size : {CFG['batch_size']}")
    print(f"[Train] Device     : {DEVICE}")

    # 5. Epoch Loop
    for epoch in range(start_epoch, CFG["epochs"] + 1):
        t0 = time.time()
        print(f"\n{'─' * 60}")
        print(f"  Epoch {epoch:>3}/{CFG['epochs']}   LR={optimizer.param_groups[0]['lr']:.2e}")

        tr_loss, tr_acc, tr_bal, tr_f1 = run_epoch(model, loader_train, criterion, optimizer, scaler, True, mixup_fn)
        va_loss, va_acc, va_bal, va_f1 = run_epoch(model, loader_val,   criterion, optimizer, scaler, False, None)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Train | loss:{tr_loss:.4f}  acc:{tr_acc:.4f}  bal_acc:{tr_bal:.4f}  f1:{tr_f1:.4f}")
        print(f"  Val   | loss:{va_loss:.4f}  acc:{va_acc:.4f}  bal_acc:{va_bal:.4f}  f1:{va_f1:.4f}  [{elapsed:.0f}s]")

        record = {
            "epoch": epoch,
            "train": {"loss": tr_loss, "acc": tr_acc, "bal_acc": tr_bal, "f1": tr_f1},
            "val":   {"loss": va_loss, "acc": va_acc, "bal_acc": va_bal, "f1": va_f1},
            "lr":    optimizer.param_groups[0]["lr"],
            "elapsed_s": elapsed,
        }
        history.append(record)
        (Path(CFG["log_dir"]) / f"epoch_{epoch:03d}.json").write_text(json.dumps(record, indent=2))

        is_best = False
        if va_bal > best_bal_acc or (va_bal == best_bal_acc and va_f1 > best_f1):
            best_bal_acc, best_f1 = va_bal, va_f1
            is_best    = True
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(model, optimizer, scheduler, scaler,
                        epoch, best_bal_acc, best_f1, history, is_best)

        if no_improve >= CFG["early_stop_patience"]:
            print(f"\n[EarlyStop] No improvement for {no_improve} epochs -- stopping.")
            break

    print(f"\n[Train] Done!  Best val bal_acc={best_bal_acc:.4f}  f1={best_f1:.4f}")

    # 6. Final Test Set Evaluation using Best Checkpoint
    print("\n[Eval] Loading best checkpoint for test evaluation ...")
    best_ckpt = Path(CFG["ckpt_dir"]) / "checkpoint_best.pth"
    if best_ckpt.exists():
        try:
            best_state = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        except TypeError:
            best_state = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(best_state["model_state"])

    evaluate_test(model, loader_test)
    plot_training_history(history)
    print("\nAll outputs saved to /kaggle/working/results/")


# --- 8. Evaluation & Comparison Against Baseline ------------------------------

def evaluate_test(model, loader_test):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader_test:
            logits = model(imgs.to(DEVICE))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    bal_acc  = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc      = accuracy_score(all_labels, all_preds)
    report   = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0
    )

    print(f"\n{'=' * 65}")
    print("  PHASE 3 WEEK 5 - TEST SET RESULTS")
    print(f"{'=' * 65}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Balanced Accuracy : {bal_acc:.4f}")
    print(f"  Macro F1-Score    : {macro_f1:.4f}")
    print(f"\nPer-class:\n"
          f"{classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0)}")

    # 1. Save summary and classification report
    summary = {
        "accuracy": round(acc, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "macro_f1": round(macro_f1, 6)
    }
    (Path(CFG["result_dir"]) / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(report).T.to_csv(Path(CFG["result_dir"]) / "classification_report.csv")

    # 2. Save test predictions
    df_preds = pd.DataFrame({
        "true_label": [CLASS_NAMES[i] for i in all_labels],
        "pred_label": [CLASS_NAMES[i] for i in all_preds],
        "true_idx":   all_labels,
        "pred_idx":   all_preds,
        **{f"prob_{CLASS_NAMES[i]}": [p[i] for p in all_probs]
           for i in range(CFG["num_classes"])}
    })
    df_preds.to_csv(Path(CFG["result_dir"]) / "test_predictions.csv", index=False)

    # 3. Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(
        f"Phase 3 Week 5: Multi-Disease Confusion Matrix\n"
        f"Bal.Acc={bal_acc:.3f}  Macro F1={macro_f1:.3f}", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(Path(CFG["result_dir"]) / "confusion_matrix.png", dpi=150)
    plt.show()

    # 4. Generate Baseline vs Week 5 Comparison Table & Bar Chart
    generate_baseline_comparison(bal_acc, macro_f1, acc, report)


def generate_baseline_comparison(w5_bal_acc, w5_macro_f1, w5_acc, w5_report):
    """
    Compares Phase 3 Week 5 performance directly against the Phase 2 Week 4 baseline.
    """
    base_bal_acc = CFG["baseline_bal_acc"]
    base_macro_f1 = CFG["baseline_macro_f1"]
    base_acc      = CFG["baseline_acc"]
    base_per_class = CFG["baseline_per_class_f1"]

    delta_bal_acc = (w5_bal_acc - base_bal_acc) * 100
    delta_macro_f1 = (w5_macro_f1 - base_macro_f1) * 100
    delta_acc      = (w5_acc - base_acc) * 100

    print(f"\n{'=' * 65}")
    print("  WEEK 4 BASELINE vs WEEK 5 AUGMENTED COMPARISON")
    print(f"{'=' * 65}")
    print(f"  Metric              Week 4 (Base)   Week 5 (Aug)     Delta")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  Accuracy            {base_acc*100:6.2f}%         {w5_acc*100:6.2f}%       {delta_acc:+5.2f}%")
    print(f"  Balanced Accuracy   {base_bal_acc*100:6.2f}%         {w5_bal_acc*100:6.2f}%       {delta_bal_acc:+5.2f}%")
    print(f"  Macro F1-Score      {base_macro_f1:6.4f}          {w5_macro_f1:6.4f}        {delta_macro_f1:+5.2f}%")
    print(f"  ─────────────────────────────────────────────────────────")

    comp_rows = []
    comp_rows.append({
        "Class": "OVERALL (Bal.Acc)",
        "Week 4 Baseline": base_bal_acc,
        "Week 5 Augmented": w5_bal_acc,
        "Delta": w5_bal_acc - base_bal_acc,
    })
    comp_rows.append({
        "Class": "OVERALL (Macro F1)",
        "Week 4 Baseline": base_macro_f1,
        "Week 5 Augmented": w5_macro_f1,
        "Delta": w5_macro_f1 - base_macro_f1,
    })

    classes = [c for c in CLASS_NAMES if c in w5_report]
    w5_f1s = [w5_report[c]["f1-score"] for c in classes]
    base_f1s = [base_per_class.get(c, 0.0) for c in classes]

    for c, b_f1, w_f1 in zip(classes, base_f1s, w5_f1s):
        comp_rows.append({
            "Class": c,
            "Week 4 Baseline": b_f1,
            "Week 5 Augmented": w_f1,
            "Delta": w_f1 - b_f1,
        })
        print(f"  {c:12s} F1    {b_f1:6.3f}          {w_f1:6.3f}        {(w_f1 - b_f1)*100:+5.2f}%")

    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(Path(CFG["result_dir"]) / "baseline_vs_week5_comparison.csv", index=False)

    # Side-by-side grouped bar chart
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, base_f1s, width, label="Week 4 Baseline", color="#90CAF9")
    rects2 = ax.bar(x + width/2, w5_f1s,   width, label="Week 5 (Augmented + Reg)", color="#1E88E5")

    ax.axhline(0.80, color="green", linestyle="--", linewidth=1.2, label="Target 80%")
    ax.set_ylabel("F1-Score", fontsize=11)
    ax.set_title(
        f"Per-Class F1 Score: Week 4 Baseline vs Week 5 Augmented\n"
        f"Macro F1: {base_macro_f1:.3f} -> {w5_macro_f1:.3f} ({delta_macro_f1:+5.2f}%)",
        fontsize=13
    )
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for r1, r2 in zip(rects1, rects2):
        h1 = r1.get_height()
        h2 = r2.get_height()
        ax.text(r1.get_x() + r1.get_width()/2, h1 + 0.01, f"{h1:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(r2.get_x() + r2.get_width()/2, h2 + 0.01, f"{h2:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(Path(CFG["result_dir"]) / "augmented_vs_baseline_f1.png", dpi=150)
    plt.show()


def plot_training_history(history):
    if not history:
        return
    epochs  = [h["epoch"]            for h in history]
    tr_loss = [h["train"]["loss"]     for h in history]
    va_loss = [h["val"]["loss"]       for h in history]
    tr_bac  = [h["train"]["bal_acc"]  for h in history]
    va_bac  = [h["val"]["bal_acc"]    for h in history]
    tr_f1   = [h["train"]["f1"]       for h in history]
    va_f1   = [h["val"]["f1"]         for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, tr_loss, label="Train", color="#1976D2")
    axes[0].plot(epochs, va_loss, label="Val", color="#F57C00")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, tr_bac, label="Train", color="#1976D2")
    axes[1].plot(epochs, va_bac, label="Val", color="#F57C00")
    axes[1].axhline(0.80, color="green", linestyle="--", label="Target 80%")
    axes[1].set_title("Balanced Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs, tr_f1, label="Train", color="#1976D2")
    axes[2].plot(epochs, va_f1, label="Val", color="#F57C00")
    axes[2].axhline(0.80, color="green", linestyle="--", label="Target 80%")
    axes[2].set_title("Macro F1-Score"); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(True)

    plt.suptitle("Phase 3 Week 5: Training & Validation Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(CFG["result_dir"]) / "training_curves.png", dpi=150)
    plt.show()


# --- 9. Entry Point -----------------------------------------------------------

if __name__ == "__main__":
    train()
