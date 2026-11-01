"""
===============================================================================
EDD Phase 3 - Week 6: RETFound Fine-Tuning + EfficientNet-V2-S Ensemble
===============================================================================
Project   : Eye Disease Detection (EDD)
Task      : Phase 3 Week 6 - Production-Grade 5-Class Classifier
Classes   : Normal, DR, Glaucoma, Cataract, AMD  (Other dropped)
Backbone A: RETFound (ViT-L/16, pretrained on 1.6M retinal images) — Nature 2023
Backbone B: EfficientNet-V2-S (CNN, ImageNet → retinal fine-tune)
Ensemble  : Weighted softmax average (0.65 RETFound + 0.35 EfficientNet)

Training Strategy:
  Phase A — Linear Probe      : 5  epochs, frozen backbone, LR=1e-3
  Phase B — Full Fine-Tune    : 30 epochs, layer-wise LR decay, Focal Loss
  EfficientNet Training       : 40 epochs, 288×288, light Mixup

Targets   : Balanced Accuracy ≥ 82%  |  Macro F1 ≥ 0.78  (single model)
             Balanced Accuracy ≥ 88%  |  Macro F1 ≥ 0.85  (ensemble)

Datasets:
  - ODIR-2019        (existing,  filtered to 5 classes)
  - RFMiD 2.0        (existing,  filtered to 5 classes)
  - APTOS 2019       (new,       DR severity → Normal/DR binary)
  - Eye Diseases     (new,       4 balanced classes: Normal/DR/Cataract/Glaucoma)
===============================================================================
"""

import os
import sys
import json
import time
import math
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

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
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight as sk_compute_class_weight

warnings.filterwarnings("ignore")

# ── 1. Configuration ──────────────────────────────────────────────────────────

CFG = {
    # ── Environment ──────────────────────────────────────────────────────
    "seed":           42,
    "output_dir":     "/kaggle/working",
    "ckpt_dir":       "/kaggle/working/checkpoints",
    "result_dir":     "/kaggle/working/results",

    # ── Dataset roots ─────────────────────────────────────────────────────
    # (existing datasets from previous weeks)
    "odir_root":    "/kaggle/input/datasets/andrewmvd/ocular-disease-recognition-odir5k",
    "rfmid_root":   "/kaggle/input/datasets/andrewmvd/retinal-disease-classification",
    # (new datasets — add these to your Kaggle notebook)
    "aptos_root":   "/kaggle/input/aptos2019-blindness-detection",
    "eyedis_root":  "/kaggle/input/eye-diseases-classification",

    # ── Classes (5-class: Other dropped) ──────────────────────────────────
    "num_classes": 5,

    # ── RETFound backbone ─────────────────────────────────────────────────
    "retfound_repo":     "YukunZhou/RETFound_mae_natureCFP",
    "retfound_file":     "RETFound_mae_natureCFP.pth",
    "retfound_fallback": "iszt/RETFound_mae_meh",         # public, no token needed
    "img_size":          224,                              # ViT-L/16 native size

    # ── Phase A: Linear Probe ─────────────────────────────────────────────
    "phase_a_epochs": 5,
    "phase_a_lr":     1e-3,
    "phase_a_batch":  32,

    # ── Phase B: Full Fine-Tune ───────────────────────────────────────────
    "phase_b_epochs":   30,
    "phase_b_lr":       1e-4,
    "phase_b_batch":    16,    # Conservative for ViT-L on T4
    "phase_b_warmup":   3,
    "weight_decay":     1e-4,
    "grad_clip":        1.0,
    "early_stop_patience": 7,

    # ── EfficientNet-V2-S ─────────────────────────────────────────────────
    "effnet_model":   "efficientnetv2_rw_s",
    "effnet_size":    288,
    "effnet_epochs":  40,
    "effnet_lr":      3e-4,
    "effnet_batch":   32,
    "drop_rate":      0.3,
    "drop_path_rate": 0.2,
    "mixup_alpha":    0.4,
    "mixup_prob":     0.3,     # Lighter than Week 5 (helps rare classes)

    # ── Loss ──────────────────────────────────────────────────────────────
    "focal_gamma":    2.0,

    # ── Ensemble ──────────────────────────────────────────────────────────
    "w_retfound": 0.65,
    "w_effnet":   0.35,

    # ── Misc ──────────────────────────────────────────────────────────────
    "num_workers":     2,
    "pin_memory":      True,
    "mixed_precision": True,
}

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "DR", "Glaucoma", "Cataract", "AMD"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG["seed"])

for d in [CFG["ckpt_dir"], CFG["result_dir"]]:
    Path(d).mkdir(parents=True, exist_ok=True)

print(f"Device  : {DEVICE}")
print(f"PyTorch : {torch.__version__}")
print(f"timm    : {timm.__version__}")
if DEVICE.type == "cuda":
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}   : {torch.cuda.get_device_name(i)}")


# ── 2. Data Loading ───────────────────────────────────────────────────────────

def load_odir_dataframe(odir_root):
    """
    Parse ODIR-2019 → 5-class DataFrame.
    'Other' (O=1), Hypertension (H=1), and Myopia (M=1) rows are DROPPED.
    """
    root = Path(odir_root)
    csv_candidates = [
        root / "ODIR-5K" / "ODIR-5K" / "data.xlsx",
        root / "ODIR-5K" / "ODIR-5K" / "ODIR-5K_Training_Annotations(Updated)_V2.xlsx",
        root / "full_df.csv",
        root / "data.xlsx",
        root / "training.csv",
        root / "data" / "training.csv",
        root / "ODIR-5K" / "training.csv",
        root / "ODIR-5K" / "data.xlsx",
    ]
    label_path = next((c for c in csv_candidates if c.exists()), None)
    if label_path is None and root.exists():
        all_tables = list(root.rglob("*.csv")) + list(root.rglob("*.xlsx"))
        relevant = [f for f in all_tables if any(k in f.name.lower() for k in ["data", "train", "annot", "odir"])]
        label_path = relevant[0] if relevant else (all_tables[0] if all_tables else None)
    if label_path is None:
        raise FileNotFoundError(f"Cannot find ODIR label file under {odir_root}")

    df = pd.read_excel(label_path) if label_path.suffix in [".xlsx", ".xls"] else pd.read_csv(label_path)
    print(f"[ODIR] Loaded labels from {label_path}")

    img_dir_candidates = [
        root / "ODIR-5K" / "ODIR-5K" / "Training Images",
        root / "ODIR-5K" / "Training Images",
        root / "preprocessed_images",
        root / "images",
        root / "train_images",
        root / "Training Images",
    ]
    img_dir = next((d for d in img_dir_candidates if d.exists() and any(d.iterdir())), None)
    if img_dir is None and root.exists():
        for p in root.rglob("*"):
            if p.is_dir() and any(f.suffix.lower() in [".jpg", ".jpeg", ".png"] for f in p.glob("*") if f.is_file()):
                img_dir = p
                break
    if img_dir is None:
        raise FileNotFoundError(f"Cannot find ODIR image directory under {odir_root}")
    print(f"[ODIR] Found images at {img_dir}")

    left_col  = next((c for c in df.columns if "left"  in c.lower() and any(k in c.lower() for k in ["fundus","filename","image"])), None)
    right_col = next((c for c in df.columns if "right" in c.lower() and any(k in c.lower() for k in ["fundus","filename","image"])), None)
    has_binary = all(col in df.columns for col in ["N", "D", "G", "C", "A"])

    records = []
    pairings = []
    if left_col:  pairings.append(left_col)
    if right_col: pairings.append(right_col)

    for _, row in df.iterrows():
        # ── Skip rows that belong to dropped classes ──
        if has_binary:
            if row.get("O", 0) == 1 or row.get("H", 0) == 1 or row.get("M", 0) == 1:
                continue

        for img_col in pairings:
            fname = str(row.get(img_col, "")).strip()
            if not fname or fname.lower() == "nan":
                continue

            img_path = None
            for candidate in [img_dir / fname, root / fname]:
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                for ext in [".jpg", ".jpeg", ".png"]:
                    for base in [img_dir, root]:
                        c = (base / fname).with_suffix(ext)
                        if c.exists():
                            img_path = c
                            break
                    if img_path:
                        break
            if img_path is None:
                for candidate in img_dir.glob(Path(fname).name):
                    img_path = candidate
                    break
            if img_path is None:
                continue

            if has_binary:
                if   row.get("N", 0) == 1: mapped = "Normal"
                elif row.get("D", 0) == 1: mapped = "DR"
                elif row.get("G", 0) == 1: mapped = "Glaucoma"
                elif row.get("C", 0) == 1: mapped = "Cataract"
                elif row.get("A", 0) == 1: mapped = "AMD"
                else: continue   # skip unclear rows
            else:
                continue         # no binary flags available, skip

            records.append({"filepath": str(img_path), "label": mapped, "source": "ODIR"})

    df_out = pd.DataFrame(records)
    print(f"[ODIR] Kept {len(df_out)} rows (5-class, Other/H/M dropped)")
    return df_out


def load_rfmid_dataframe(rfmid_root):
    """
    Parse RFMiD 2.0 → 5-class DataFrame.
    Only DR, ARMD(AMD), Normal, and Glaucoma-related codes are kept.
    """
    root = Path(rfmid_root)
    splits = {
        "train": [
            (root / "Training_Set" / "Training_Set" / "RFMiD_Training_Labels.csv",
             root / "Training_Set" / "Training_Set" / "Training"),
            (root / "RFMiD_Training_Labels.csv", root / "Training"),
        ],
        "val": [
            (root / "Validation_Set" / "Validation_Set" / "RFMiD_Validation_Labels.csv",
             root / "Validation_Set" / "Validation_Set" / "Validation"),
            (root / "RFMiD_Validation_Labels.csv", root / "Validation"),
        ],
        "test": [
            (root / "Test_Set" / "Test_Set" / "RFMiD_Test_Labels.csv",
             root / "Test_Set" / "Test_Set" / "Test"),
            (root / "RFMiD_Test_Labels.csv", root / "Test"),
        ],
    }
    # Only map to our 5 classes; everything else is skipped
    rfmid_keep = {
        "DR": "DR",
        "ARMD": "AMD",
        "ODC": "Glaucoma",
        "ODP": "Glaucoma",
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
                if split_name.lower() in f.name.lower():
                    csv_path = f
                    for p in root.rglob("*"):
                        if p.is_dir() and any(p.glob("*.[jp][pn]g")):
                            img_dir = p
                            break
                    break
        if csv_path is None or not csv_path.exists():
            print(f"[RFMiD] WARNING: No {split_name} CSV — skipping")
            continue
        df = pd.read_csv(csv_path)
        if img_dir is None or not img_dir.exists():
            print(f"[RFMiD] WARNING: No image dir for {split_name} — skipping")
            continue

        disease_cols = [c for c in df.columns if c.upper() in rfmid_keep]
        disease_risk_col = next((c for c in df.columns if c.upper() in ("DISEASE_RISK",)), None)

        for _, row in df.iterrows():
            img_id = str(row["ID"]) if "ID" in row else str(row.iloc[0])
            img_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                c = img_dir / f"{img_id}{ext}"
                if c.exists():
                    img_path = c
                    break
            if img_path is None:
                continue

            mapped = None
            for col in disease_cols:
                if row.get(col, 0) == 1:
                    mapped = rfmid_keep[col.upper()]
                    break

            # If no disease → Normal (disease_risk = 0 means no disease)
            if mapped is None:
                if disease_risk_col and row.get(disease_risk_col, 1) == 0:
                    mapped = "Normal"
                else:
                    continue   # Has a disease not in our 5 classes → skip

            records.append({"filepath": str(img_path), "label": mapped, "source": "RFMiD"})

    df_out = pd.DataFrame(records)
    print(f"[RFMiD] Kept {len(df_out)} rows (5-class)")
    return df_out


def load_aptos_dataframe(aptos_root):
    """
    Parse APTOS 2019 Blindness Detection dataset.
    Diagnosis: 0=No DR → Normal, 1-4=DR → DR.
    """
    root = Path(aptos_root)
    csv_candidates = [
        root / "train.csv",
        root / "train_labels.csv",
        root / "aptos2019-blindness-detection" / "train.csv",
    ]
    csv_path = next((c for c in csv_candidates if c.exists()), None)
    if csv_path is None:
        found = list(root.rglob("train*.csv"))
        csv_path = found[0] if found else None
    if csv_path is None:
        print("[APTOS] WARNING: Cannot find train.csv — skipping APTOS")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"[APTOS] Loaded {len(df)} rows from {csv_path}")

    img_dir_candidates = [
        root / "train_images",
        root / "train",
        root / "images" / "train",
        root / "aptos2019-blindness-detection" / "train_images",
    ]
    img_dir = next((d for d in img_dir_candidates if d.exists()), None)
    if img_dir is None:
        for p in root.rglob("*"):
            if p.is_dir() and any(p.glob("*.[jp][pn]g")):
                img_dir = p
                break
    if img_dir is None:
        print("[APTOS] WARNING: Cannot find image directory — skipping APTOS")
        return pd.DataFrame()
    print(f"[APTOS] Found images at {img_dir}")

    id_col   = next((c for c in df.columns if "id" in c.lower()), df.columns[0])
    diag_col = next((c for c in df.columns if "diag" in c.lower() or "label" in c.lower()), None)
    if diag_col is None:
        diag_col = df.columns[1]

    records = []
    for _, row in df.iterrows():
        img_id = str(row[id_col]).strip()
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            for base in [img_dir / img_id, img_dir / f"{img_id}"]:
                c = Path(str(base) + ext) if not img_id.endswith(ext) else img_dir / img_id
                if c.exists():
                    img_path = c
                    break
            if img_path:
                break
        if img_path is None:
            continue

        diagnosis = int(row[diag_col])
        label = "Normal" if diagnosis == 0 else "DR"
        records.append({"filepath": str(img_path), "label": label, "source": "APTOS"})

    df_out = pd.DataFrame(records)
    print(f"[APTOS] Kept {len(df_out)} rows → Normal:{(df_out['label']=='Normal').sum()}, DR:{(df_out['label']=='DR').sum()}")
    return df_out


def load_eyedis_dataframe(eyedis_root):
    """
    Parse Eye Diseases Classification dataset (folder-structure).
    Folder names → class labels.
    """
    root = Path(eyedis_root)
    # Typical structure: dataset/train/{class_name}/*.jpg
    # or: train/{class_name}/*.jpg  or just: {class_name}/*.jpg
    folder_map = {
        "normal":               "Normal",
        "diabetic_retinopathy": "DR",
        "diabeticretinopathy":  "DR",
        "diabetic retinopathy": "DR",
        "glaucoma":             "Glaucoma",
        "cataract":             "Cataract",
        "amd":                  "AMD",
    }

    records = []
    for img_path in root.rglob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        # Walk parent folders to find a class folder name
        mapped = None
        for part in reversed(img_path.parts):
            key = part.lower().replace(" ", "").replace("_", "")
            # match flexibly
            for fkey, label in folder_map.items():
                if key == fkey.replace("_", "").replace(" ", ""):
                    mapped = label
                    break
            if mapped:
                break
        if mapped is None:
            continue
        records.append({"filepath": str(img_path), "label": mapped, "source": "EyeDis"})

    df_out = pd.DataFrame(records)
    if len(df_out) > 0:
        print(f"[EyeDis] Found {len(df_out)} images across {df_out['label'].nunique()} classes")
    else:
        print("[EyeDis] WARNING: No images found — check path")
    return df_out


def build_combined_dataframe(odir_root, rfmid_root, aptos_root, eyedis_root):
    """Load all sources, merge, encode, and stratified-split (70/15/15)."""
    print("\n[Data] Loading ODIR-2019 (5-class) ...")
    df_odir = load_odir_dataframe(odir_root)

    print("[Data] Loading RFMiD 2.0 (5-class) ...")
    df_rfmid = load_rfmid_dataframe(rfmid_root)

    print("[Data] Loading APTOS 2019 ...")
    df_aptos = load_aptos_dataframe(aptos_root)

    print("[Data] Loading Eye Diseases Classification ...")
    df_eyedis = load_eyedis_dataframe(eyedis_root)

    dfs = [d for d in [df_odir, df_rfmid, df_aptos, df_eyedis] if len(d) > 0]
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n[Data] Combined raw: {len(df)} images")

    # Drop any rows whose label isn't in our 5 classes
    df = df[df["label"].isin(CLASS_NAMES)].copy()
    df["label_idx"] = df["label"].map(CLASS_TO_IDX).astype(int)

    print("\n[Data] Class distribution:")
    for cls in CLASS_NAMES:
        cnt = (df["label"] == cls).sum()
        print(f"       {cls:12s}: {cnt:5d}  ({cnt/len(df)*100:.1f}%)")

    # Stratified split per class
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
    print(f"\n[Data] Split → train:{len(df_train)}  val:{len(df_val)}  test:{len(df_test)}")

    # Save split to disk for reproducibility
    splits_path = Path(CFG["result_dir"]) / "data_splits.json"
    splits_data = {
        "train": df_train[["filepath", "label", "source"]].to_dict("records"),
        "val":   df_val[["filepath",   "label", "source"]].to_dict("records"),
        "test":  df_test[["filepath",  "label", "source"]].to_dict("records"),
    }
    splits_path.write_text(json.dumps(splits_data, indent=2))
    return df_train, df_val, df_test


def compute_class_weights(df_train):
    """Inverse-frequency weights for Focal / Cross-Entropy loss."""
    counts  = np.bincount(df_train["label_idx"], minlength=CFG["num_classes"]).astype(float)
    counts  = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * CFG["num_classes"]
    print("\n[Loss] Class weights:", {CLASS_NAMES[i]: f"{weights[i]:.3f}" for i in range(CFG["num_classes"])})
    return torch.tensor(weights, dtype=torch.float32)


# ── 3. Augmentation & Dataset ─────────────────────────────────────────────────

def get_transforms(split, img_size=224):
    """
    Medical fundus augmentation (CLAHE + geometric + noise).
    No Mixup/CutMix for RETFound fine-tuning — we want sharp boundaries.
    """
    if split == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.07, scale_limit=0.10, rotate_limit=30, p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=1.0),
            ], p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=12, p=0.4),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
            ], p=0.25),
            A.CoarseDropout(max_holes=4, max_height=20, max_width=20,
                            min_holes=1, min_height=8, min_width=8, p=0.25),
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
    def __init__(self, df, split="train", img_size=224):
        self.df        = df.reset_index(drop=True)
        self.transform = get_transforms(split, img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["filepath"]).convert("RGB"))
        img = self.transform(image=img)["image"]
        return img, int(row["label_idx"])


def build_dataloaders(df_train, df_val, df_test, img_size=224, batch_size=16):
    ds_train = EyeDiseaseDataset(df_train, "train", img_size)
    ds_val   = EyeDiseaseDataset(df_val,   "val",   img_size)
    ds_test  = EyeDiseaseDataset(df_test,  "test",  img_size)

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"],
                              drop_last=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size*2, shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"])
    loader_test  = DataLoader(ds_test,  batch_size=batch_size*2, shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"])
    return loader_train, loader_val, loader_test


# ── 4. Focal Loss ─────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss — focuses training on hard/misclassified samples."""
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.weight    = weight
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight,
                                  reduction="none")
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean() if self.reduction == "mean" else focal.sum()


# ── 5. RETFound Model ─────────────────────────────────────────────────────────

def load_retfound_weights(hf_token=None):
    """
    Download RETFound weights from Hugging Face.
    Falls back to public community version if gated repo isn't accessible.
    Returns path to the .pth file.
    """
    from huggingface_hub import hf_hub_download, snapshot_download
    import requests

    save_path = Path(CFG["output_dir"]) / "retfound_weights.pth"
    if save_path.exists():
        print(f"[RETFound] Found cached weights at {save_path}")
        return str(save_path)

    print("[RETFound] Downloading weights from Hugging Face ...")
    try:
        weight_path = hf_hub_download(
            repo_id=CFG["retfound_repo"],
            filename=CFG["retfound_file"],
            token=hf_token,
            cache_dir=CFG["output_dir"],
        )
        # Copy to stable path
        shutil.copy(weight_path, save_path)
        print(f"[RETFound] ✅ Downloaded to {save_path}")
        return str(save_path)
    except Exception as e:
        print(f"[RETFound] Primary repo failed: {e}")
        print("[RETFound] Trying fallback (iszt/RETFound_mae_meh) ...")
        try:
            weight_path = hf_hub_download(
                repo_id=CFG["retfound_fallback"],
                filename="model.safetensors",
                token=hf_token,
                cache_dir=CFG["output_dir"],
            )
            shutil.copy(weight_path, save_path.with_suffix(".safetensors"))
            print("[RETFound] ✅ Downloaded fallback safetensors")
            return str(save_path.with_suffix(".safetensors"))
        except Exception as e2:
            raise RuntimeError(
                f"Cannot download RETFound weights.\n"
                f"Primary error : {e}\n"
                f"Fallback error: {e2}\n"
                "Please check your HF_TOKEN and that you've requested access at "
                "https://huggingface.co/YukunZhou/RETFound_mae_natureCFP"
            )


def build_retfound(num_classes=5, weight_path=None):
    """
    Build RETFound classifier:
      - ViT-L/16 backbone loaded with retinal MAE weights
      - Replaced head: LayerNorm → Dropout(0.3) → Linear(1024, num_classes)
    """
    print("\n[Model] Building RETFound (ViT-L/16) ...")

    # Create ViT-L/16 via timm (no pretrained ImageNet — we load RETFound weights)
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.3,
    )

    if weight_path is not None:
        ext = Path(weight_path).suffix.lower()
        if ext == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(weight_path)
            msg = model.load_state_dict(state_dict, strict=False)
        else:
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model", ckpt)
            msg = model.load_state_dict(state_dict, strict=False)
        print(f"[Model] Loaded RETFound weights — missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)}")

    # Gradient checkpointing — essential for ViT-L on T4
    model.set_grad_checkpointing(True)
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"[Model] Gradient checkpointing: ON")
    return model.to(DEVICE)


def build_efficientnet(num_classes=5):
    """EfficientNet-V2-S for ensemble."""
    print(f"\n[Model] Building {CFG['effnet_model']} ...")
    model = timm.create_model(
        CFG["effnet_model"],
        pretrained=True,
        num_classes=num_classes,
        drop_rate=CFG["drop_rate"],
        drop_path_rate=CFG["drop_path_rate"],
    )
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model.to(DEVICE)


# ── 6. Layer-Wise LR Decay for RETFound ──────────────────────────────────────

def get_retfound_param_groups(model, base_lr, weight_decay):
    """
    Layer-wise learning rate decay for ViT:
      head          → base_lr × 1.0
      last 4 blocks → base_lr × 0.1
      earlier layers→ base_lr × 0.01
    No weight decay on LayerNorm and bias parameters.
    """
    no_decay = {"bias", "norm", "LayerNorm"}
    param_groups = []

    def lr_scale(name):
        if "head" in name:          return 1.0
        if "blocks" in name:
            try:
                block_idx = int(name.split("blocks.")[1].split(".")[0])
                total = len(model.blocks)
                if block_idx >= total - 4:   return 0.1
            except (IndexError, ValueError):
                pass
        return 0.01

    seen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        param_groups.append({
            "params": [param],
            "lr": base_lr * lr_scale(name),
            "weight_decay": wd,
        })
    return param_groups


# ── 7. Training Utilities ─────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup → cosine decay LR scheduler."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch, tag=""):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast(enabled=CFG["mixed_precision"]):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        n          += imgs.size(0)
    avg_loss = total_loss / n
    print(f"  [{tag} Epoch {epoch:02d}] Train Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, tag="Val"):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        with autocast(enabled=CFG["mixed_precision"]):
            logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"  [{tag}] Bal.Acc: {bal_acc:.4f}  |  Macro F1: {macro_f1:.4f}")
    return bal_acc, macro_f1, all_preds, all_labels


# ── 8. Phase A: Linear Probe ─────────────────────────────────────────────────

def train_phase_a(model, loader_train, loader_val, class_weights):
    """
    Phase A — Linear Probe: freeze backbone, train head only.
    5 epochs, high LR (1e-3), no augmentation benefit from backbone adaptation.
    """
    print("\n" + "="*60)
    print("  PHASE A — LINEAR PROBE (Frozen Backbone, 5 epochs)")
    print("="*60)

    # Freeze all parameters except the classification head
    for name, param in model.named_parameters():
        param.requires_grad = "head" in name
    head_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in head_params)/1e6:.2f}M  (head only)")

    criterion = FocalLoss(weight=class_weights.to(DEVICE), gamma=CFG["focal_gamma"])
    optimizer = optim.AdamW(head_params, lr=CFG["phase_a_lr"], weight_decay=0.0)
    scaler    = GradScaler(enabled=CFG["mixed_precision"])

    best_bal_acc = 0.0
    for epoch in range(1, CFG["phase_a_epochs"] + 1):
        train_one_epoch(model, loader_train, optimizer, criterion, scaler, epoch, "PhaseA")
        bal_acc, macro_f1, _, _ = evaluate(model, loader_val, "PhaseA-Val")
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "bal_acc": bal_acc, "phase": "A"},
                       Path(CFG["ckpt_dir"]) / "retfound_phase_a_best.pth")

    print(f"\n[Phase A] Best Val Bal.Acc = {best_bal_acc:.4f}")

    # Unfreeze all for Phase B
    for param in model.parameters():
        param.requires_grad = True

    return model


# ── 9. Phase B: Full Fine-Tune ────────────────────────────────────────────────

def train_phase_b(model, loader_train, loader_val, class_weights):
    """
    Phase B — Full Fine-Tune: layer-wise LR decay, cosine schedule, Focal Loss.
    30 epochs, early stopping patience=7.
    """
    print("\n" + "="*60)
    print("  PHASE B — FULL FINE-TUNE (Layer-wise LR, 30 epochs)")
    print("="*60)

    criterion  = FocalLoss(weight=class_weights.to(DEVICE), gamma=CFG["focal_gamma"])
    param_groups = get_retfound_param_groups(model, CFG["phase_b_lr"], CFG["weight_decay"])
    optimizer  = optim.AdamW(param_groups)
    scaler     = GradScaler(enabled=CFG["mixed_precision"])

    total_steps   = CFG["phase_b_epochs"] * len(loader_train)
    warmup_steps  = CFG["phase_b_warmup"] * len(loader_train)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_bal_acc    = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_bal_acc": [], "val_f1": []}

    for epoch in range(1, CFG["phase_b_epochs"] + 1):
        train_loss = train_one_epoch(model, loader_train, optimizer, criterion, scaler, epoch, "PhaseB")
        scheduler.step()
        bal_acc, macro_f1, _, _ = evaluate(model, loader_val, "PhaseB-Val")

        history["train_loss"].append(train_loss)
        history["val_bal_acc"].append(bal_acc)
        history["val_f1"].append(macro_f1)

        if bal_acc > best_bal_acc:
            best_bal_acc     = bal_acc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch":       epoch,
                "bal_acc":     bal_acc,
                "macro_f1":    macro_f1,
                "phase":       "B",
            }, Path(CFG["ckpt_dir"]) / "retfound_best.pth")
            print(f"  ★ New best saved  (Bal.Acc = {best_bal_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CFG["early_stop_patience"]:
                print(f"  [Early Stop] No improvement for {CFG['early_stop_patience']} epochs.")
                break

    # Save history
    (Path(CFG["result_dir"]) / "retfound_history.json").write_text(json.dumps(history, indent=2))
    print(f"\n[Phase B] Best Val Bal.Acc = {best_bal_acc:.4f}")
    return model, best_bal_acc, history


# ── 10. EfficientNet-V2-S Training ───────────────────────────────────────────

def train_efficientnet(model, df_train, df_val, class_weights):
    """
    Train EfficientNet-V2-S with 288×288 images, light Mixup, Focal Loss.
    """
    print("\n" + "="*60)
    print(f"  EFFICIENTNET-V2-S TRAINING ({CFG['effnet_epochs']} epochs, 288×288)")
    print("="*60)

    loader_train, loader_val, _ = build_dataloaders(
        df_train, df_val, df_val,  # df_val as test placeholder
        img_size=CFG["effnet_size"], batch_size=CFG["effnet_batch"]
    )

    mixup_fn = Mixup(
        mixup_alpha=CFG["mixup_alpha"],
        cutmix_alpha=0.0,
        prob=CFG["mixup_prob"],
        switch_prob=0.0,
        label_smoothing=0.1,
        num_classes=CFG["num_classes"],
    )

    criterion = FocalLoss(weight=class_weights.to(DEVICE), gamma=CFG["focal_gamma"])
    optimizer = optim.AdamW(model.parameters(), lr=CFG["effnet_lr"],
                            weight_decay=CFG["weight_decay"])
    scaler    = GradScaler(enabled=CFG["mixed_precision"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["effnet_epochs"])

    best_bal_acc     = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_bal_acc": [], "val_f1": []}

    for epoch in range(1, CFG["effnet_epochs"] + 1):
        # ── Train with optional Mixup ──
        model.train()
        total_loss, n = 0.0, 0
        for imgs, labels in loader_train:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            imgs, soft_labels = mixup_fn(imgs, labels)
            optimizer.zero_grad()
            with autocast(enabled=CFG["mixed_precision"]):
                logits = model(imgs)
                # Soft target loss — FocalLoss only supports hard labels,
                # so for mixup batches we use CrossEntropy with soft targets
                loss = F.cross_entropy(logits, soft_labels,
                                       weight=class_weights.to(DEVICE))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)
            n          += imgs.size(0)
        avg_loss = total_loss / n
        scheduler.step()

        bal_acc, macro_f1, _, _ = evaluate(model, loader_val, f"EffNet-Ep{epoch:02d}")
        print(f"  [EffNet Epoch {epoch:02d}] Train Loss: {avg_loss:.4f}")

        history["train_loss"].append(avg_loss)
        history["val_bal_acc"].append(bal_acc)
        history["val_f1"].append(macro_f1)

        if bal_acc > best_bal_acc:
            best_bal_acc     = bal_acc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch":       epoch,
                "bal_acc":     bal_acc,
                "macro_f1":    macro_f1,
            }, Path(CFG["ckpt_dir"]) / "effnet_best.pth")
            print(f"  ★ New best saved  (Bal.Acc = {best_bal_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CFG["early_stop_patience"]:
                print(f"  [Early Stop] No improvement for {CFG['early_stop_patience']} epochs.")
                break

    (Path(CFG["result_dir"]) / "effnet_history.json").write_text(json.dumps(history, indent=2))
    print(f"\n[EffNet] Best Val Bal.Acc = {best_bal_acc:.4f}")
    return model, best_bal_acc


# ── 11. Ensemble Evaluation ───────────────────────────────────────────────────

@torch.no_grad()
def ensemble_evaluate(retfound, effnet, loader_test, df_test):
    """
    Weighted softmax ensemble:
      pred = 0.65 × softmax(RETFound) + 0.35 × softmax(EfficientNet)
    Evaluates on the test set and prints full classification report.
    """
    print("\n" + "="*60)
    print("  ENSEMBLE EVALUATION (Test Set)")
    print("="*60)

    retfound.eval()
    effnet.eval()

    # Build EffNet loader at its native resolution
    ds_test_eff = EyeDiseaseDataset(df_test, "test", CFG["effnet_size"])
    loader_test_eff = DataLoader(ds_test_eff, batch_size=CFG["effnet_batch"]*2,
                                 shuffle=False, num_workers=CFG["num_workers"])

    # Collect softmax probabilities from both models
    probs_ret, probs_eff, all_labels = [], [], []
    for (imgs_ret, labels), (imgs_eff, _) in zip(loader_test, loader_test_eff):
        imgs_ret = imgs_ret.to(DEVICE)
        imgs_eff = imgs_eff.to(DEVICE)
        with autocast(enabled=CFG["mixed_precision"]):
            p_ret = torch.softmax(retfound(imgs_ret), dim=1).cpu()
            p_eff = torch.softmax(effnet(imgs_eff),   dim=1).cpu()
        probs_ret.append(p_ret)
        probs_eff.append(p_eff)
        all_labels.extend(labels.numpy())

    probs_ret = torch.cat(probs_ret, dim=0).numpy()
    probs_eff = torch.cat(probs_eff, dim=0).numpy()

    # Weighted ensemble
    ensemble_probs = (CFG["w_retfound"] * probs_ret) + (CFG["w_effnet"] * probs_eff)
    preds          = ensemble_probs.argmax(axis=1)

    bal_acc  = balanced_accuracy_score(all_labels, preds)
    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    acc      = accuracy_score(all_labels, preds)

    print(f"\n  Accuracy          : {acc:.4f}")
    print(f"  Balanced Accuracy : {bal_acc:.4f}")
    print(f"  Macro F1-Score    : {macro_f1:.4f}")
    print("\nPer-class:")
    print(classification_report(all_labels, preds, target_names=CLASS_NAMES, zero_division=0))

    return bal_acc, macro_f1, preds, all_labels, ensemble_probs


# ── 12. Plotting ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, title, save_path):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                linewidths=0.5, linecolor="gray")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved → {save_path}")


def plot_training_curves(history, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=3)
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(True)
    axes[1].plot(epochs, history["val_bal_acc"], "g-o", markersize=3, label="Bal.Acc")
    axes[1].plot(epochs, history["val_f1"],      "r-s", markersize=3, label="Macro F1")
    axes[1].set_title("Validation Metrics"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved → {save_path}")


# ── 13. Main Training Function ───────────────────────────────────────────────

def train(hf_token=None):
    print("=" * 70)
    print("  EDD Phase 3 Week 6 — RETFound + EfficientNet-V2-S Ensemble")
    print("=" * 70)

    # ── Data ──────────────────────────────────────────────────────────────
    df_train, df_val, df_test = build_combined_dataframe(
        CFG["odir_root"], CFG["rfmid_root"],
        CFG["aptos_root"], CFG["eyedis_root"]
    )
    class_weights = compute_class_weights(df_train)

    # ── Dataloaders (RETFound resolution: 224) ────────────────────────────
    loader_train, loader_val, loader_test = build_dataloaders(
        df_train, df_val, df_test,
        img_size=CFG["img_size"], batch_size=CFG["phase_b_batch"]
    )

    # ── Download RETFound weights ─────────────────────────────────────────
    weight_path = load_retfound_weights(hf_token=hf_token)

    # ── Build & train RETFound (Phase A → Phase B) ────────────────────────
    retfound = build_retfound(num_classes=CFG["num_classes"], weight_path=weight_path)
    retfound = train_phase_a(retfound, loader_train, loader_val, class_weights)
    retfound, retfound_val_acc, ret_history = train_phase_b(
        retfound, loader_train, loader_val, class_weights
    )

    # Load best RETFound checkpoint
    ckpt = torch.load(Path(CFG["ckpt_dir"]) / "retfound_best.pth",
                      map_location=DEVICE, weights_only=False)
    retfound.load_state_dict(ckpt["model_state"])
    print(f"\n[RETFound] Loaded best checkpoint from Epoch {ckpt['epoch']} "
          f"(Val Bal.Acc = {ckpt['bal_acc']:.4f})")

    # ── Build & train EfficientNet-V2-S ───────────────────────────────────
    effnet = build_efficientnet(num_classes=CFG["num_classes"])
    effnet, effnet_val_acc = train_efficientnet(effnet, df_train, df_val, class_weights)

    # Load best EffNet checkpoint
    ckpt_eff = torch.load(Path(CFG["ckpt_dir"]) / "effnet_best.pth",
                          map_location=DEVICE, weights_only=False)
    effnet.load_state_dict(ckpt_eff["model_state"])
    print(f"\n[EffNet] Loaded best checkpoint from Epoch {ckpt_eff['epoch']} "
          f"(Val Bal.Acc = {ckpt_eff['bal_acc']:.4f})")

    # ── Ensemble evaluation on TEST set ───────────────────────────────────
    bal_acc, macro_f1, preds, labels, _ = ensemble_evaluate(
        retfound, effnet, loader_test, df_test
    )

    # ── Generate plots ────────────────────────────────────────────────────
    plot_confusion_matrix(labels, preds, "Week 6 Ensemble — Test Confusion Matrix",
                          Path(CFG["result_dir"]) / "confusion_matrix.png")
    plot_training_curves(ret_history, "RETFound Fine-Tuning Curves",
                         Path(CFG["result_dir"]) / "retfound_curves.png")

    # ── Save summary metrics ──────────────────────────────────────────────
    summary = {
        "phase":             "Phase 3 Week 6",
        "retfound_val_acc":  retfound_val_acc,
        "effnet_val_acc":    effnet_val_acc,
        "ensemble_bal_acc":  bal_acc,
        "ensemble_macro_f1": macro_f1,
        "classes":           CLASS_NAMES,
        "ensemble_weights":  {"retfound": CFG["w_retfound"], "effnet": CFG["w_effnet"]},
    }
    (Path(CFG["result_dir"]) / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\n[Done] Summary saved to {CFG['result_dir']}/summary_metrics.json")

    print("\n" + "=" * 70)
    print("  PHASE 3 WEEK 6 — FINAL RESULTS")
    print("=" * 70)
    print(f"  RETFound Val Bal.Acc  : {retfound_val_acc:.4f}")
    print(f"  EffNet  Val Bal.Acc   : {effnet_val_acc:.4f}")
    print(f"  Ensemble Test Bal.Acc : {bal_acc:.4f}")
    print(f"  Ensemble Macro F1     : {macro_f1:.4f}")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    # Retrieve HF token from environment (set via Kaggle Secrets)
    hf_token = os.environ.get("HF_TOKEN", None)
    train(hf_token=hf_token)
