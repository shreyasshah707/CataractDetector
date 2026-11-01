#!/usr/bin/env python3
"""
=============================================================================
EDD Project  -  Phase 2, Week 4
Multi-Disease Eye Disease Classifier (ODIR-2019 + RFMiD 2.0)
6 Classes: Normal, DR, Glaucoma, Cataract, AMD, Other

Features
--------
* EfficientNet-B3 (timm) fine-tuned with ImageNet weights
* Weighted cross-entropy to handle class imbalance
* Full checkpoint/resume system -- safe against Kaggle's 12-hour GPU limit
* Per-epoch JSON logs written to /kaggle/working/logs/
* Best model tracked by Balanced Accuracy (macro F1 as tiebreaker)
* Final evaluation: confusion matrix, per-class report, summary JSON
=============================================================================
"""

# --- 0. Imports & Setup ------------------------------------------------------
import os, sys, json, time, shutil, random, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import timm
except ImportError:
    os.system("pip install timm -q")
    import timm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    os.system("pip install albumentations -q")
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score
)

warnings.filterwarnings("ignore")

# --- 1. Configuration --------------------------------------------------------
CFG = {
    # Paths
    # Kaggle input datasets (adjust to your dataset slugs)
    # Expected structure under each root:
    #   odir_root/  -> contains images/ folder and training.csv / labels.csv
    #   rfmid_root/ -> contains Training_Set/ Validation_Set/ Test_Set/ and CSVs
    "odir_root":   "/kaggle/input/odir-2019",
    "rfmid_root":  "/kaggle/input/rfmid-2020",

    "output_dir":  "/kaggle/working",
    "log_dir":     "/kaggle/working/logs",
    "ckpt_dir":    "/kaggle/working/checkpoints",
    "result_dir":  "/kaggle/working/results",

    # Resume
    # Set to path of a .pth checkpoint to resume from (e.g. after a Kaggle reset)
    # e.g. "/kaggle/input/edd-phase2-ckpt/checkpoint_epoch_05.pth"
    # Leave None to auto-detect or start fresh.
    "resume_from": None,

    # Model
    "model_name":  "efficientnet_b3",
    "num_classes": 6,
    "img_size":    300,
    "pretrained":  True,

    # Training
    "epochs":      30,
    "batch_size":  32,
    "num_workers": 2,
    "lr":          1e-4,
    "weight_decay":1e-4,
    "label_smoothing": 0.1,

    # LR scheduler: cosine annealing with warm restarts
    "scheduler":   "cosine",
    "T_0":         10,
    "step_size":   10,
    "gamma":       0.1,

    # Misc
    "seed":        42,
    "mixed_precision": True,
    "grad_clip":   1.0,
    "early_stop_patience": 8,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class definitions (order matters -- must match label encoding below)
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


# --- 2. Dataset Preparation --------------------------------------------------

def load_odir_dataframe(odir_root):
    """
    Parse ODIR-2019 into a flat DataFrame with columns: filepath, label.

    ODIR label codes -> our 6-class scheme:
        N / Normal -> Normal
        D / Diabetes -> DR
        G / Glaucoma -> Glaucoma
        C / Cataract -> Cataract
        A / AMD -> AMD
        M, H, O / Other -> Other
    """
    root = Path(odir_root)

    # 1. Locate label file (CSV or Excel)
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

    # Fallback recursive search if candidate list fails
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

    # 2. Locate image directory
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

    # Fallback search for any directory containing images
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

    # 3. Label mapping definitions
    odir_label_map = {
        "N": "Normal", "NORMAL": "Normal",
        "D": "DR", "DIABETES": "DR", "DIABETIC": "DR",
        "G": "Glaucoma", "GLAUCOMA": "Glaucoma",
        "C": "Cataract", "CATARACT": "Cataract",
        "A": "AMD", "MYOPIA": "Other",
        "M": "Other", "H": "Other", "O": "Other", "HYPERTENSION": "Other",
    }

    # Identify image columns
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
    Parse RFMiD 2.0 into a flat DataFrame.
    RFMiD has 28 disease binary-flag columns.
    We map them to our 6-class scheme.
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


# --- 3. Dataset & DataLoader -------------------------------------------------

def get_transforms(split, img_size):
    if split == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.CLAHE(clip_limit=2.0, p=1.0),
            ], p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
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


# --- 4. Model ----------------------------------------------------------------

def build_model():
    model = timm.create_model(
        CFG["model_name"],
        pretrained=CFG["pretrained"],
        num_classes=CFG["num_classes"],
    )
    return model.to(DEVICE)


# --- 5. Checkpoint System ----------------------------------------------------

CHECKPOINT_META = Path(CFG["ckpt_dir"]) / "training_state.json"


def save_checkpoint(model, optimizer, scheduler, scaler,
                    epoch, best_bal_acc, best_f1, history, is_best=False):
    """
    Save complete training state after every epoch.
    - checkpoint_latest.pth  : always overwritten (safe resume point)
    - checkpoint_epoch_NNN.pth: saved every 5 epochs as extra backup
    - checkpoint_best.pth    : overwritten whenever val balanced_accuracy improves
    - training_state.json    : lightweight meta for quick human inspection
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
        "best_bal_acc": round(best_bal_acc, 6),
        "best_f1":      round(best_f1, 6),
        "timestamp":    state["timestamp"],
    }
    CHECKPOINT_META.write_text(json.dumps(meta, indent=2))


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    """Load a checkpoint; return (start_epoch, best_bal_acc, best_f1, history)."""
    print(f"\n[Checkpoint] Resuming from: {path}")
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if scheduler and state.get("scheduler_state"):
        scheduler.load_state_dict(state["scheduler_state"])
    if scaler and state.get("scaler_state"):
        scaler.load_state_dict(state["scaler_state"])
    start_epoch  = state["epoch"] + 1
    best_bal_acc = state.get("best_bal_acc", 0.0)
    best_f1      = state.get("best_f1", 0.0)
    history      = state.get("history", [])
    print(f"[Checkpoint] Epoch={start_epoch}  best_bal_acc={best_bal_acc:.4f}")
    return start_epoch, best_bal_acc, best_f1, history


# --- 6. Training & Validation Loops ------------------------------------------

def run_epoch(model, loader, criterion, optimizer, scaler, is_train):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=(CFG["mixed_precision"] and DEVICE == "cuda")):
                logits = model(imgs)
                loss   = criterion(logits, labels)

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
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, bal_acc, macro_f1


# --- 7. Main Training Orchestrator -------------------------------------------

def train():
    print("\n" + "=" * 70)
    print("  EDD Phase 2 - Week 4: Multi-Disease Classifier (ODIR + RFMiD)")
    print("=" * 70)

    # Data
    df_train, df_val, df_test = build_combined_dataframe(CFG["odir_root"], CFG["rfmid_root"])

    # Persist splits (so we can re-run evaluation without reprocessing)
    splits_path = Path(CFG["output_dir"]) / "data_splits.json"
    splits_path.write_text(json.dumps({
        "train": df_train[["filepath", "label", "label_idx"]].to_dict("records"),
        "val":   df_val[["filepath",   "label", "label_idx"]].to_dict("records"),
        "test":  df_test[["filepath",  "label", "label_idx"]].to_dict("records"),
    }, indent=2))

    loader_train, loader_val, loader_test = build_dataloaders(df_train, df_val, df_test)
    class_weights = compute_class_weights(df_train).to(DEVICE)

    # Model / optimizer / scheduler
    model     = build_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=CFG["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    if CFG["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG["T_0"])
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CFG["step_size"], gamma=CFG["gamma"])

    scaler = (torch.amp.GradScaler("cuda")
              if CFG["mixed_precision"] and DEVICE == "cuda" else None)

    # Resume logic
    start_epoch  = 1
    best_bal_acc = 0.0
    best_f1      = 0.0
    history      = []
    no_improve   = 0

    # 1. Explicit path from CFG
    resume_path = CFG.get("resume_from")
    # 2. Auto-detect latest checkpoint from /kaggle/working/checkpoints/
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

    # Epoch loop
    for epoch in range(start_epoch, CFG["epochs"] + 1):
        t0 = time.time()
        print(f"\n{'─' * 60}")
        print(f"  Epoch {epoch:>3}/{CFG['epochs']}   LR={optimizer.param_groups[0]['lr']:.2e}")

        tr_loss, tr_acc, tr_bal, tr_f1 = run_epoch(model, loader_train, criterion, optimizer, scaler, True)
        va_loss, va_acc, va_bal, va_f1 = run_epoch(model, loader_val,   criterion, optimizer, scaler, False)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Train | loss:{tr_loss:.4f}  acc:{tr_acc:.4f}  bal_acc:{tr_bal:.4f}  f1:{tr_f1:.4f}")
        print(f"  Val   | loss:{va_loss:.4f}  acc:{va_acc:.4f}  bal_acc:{va_bal:.4f}  f1:{va_f1:.4f}  [{elapsed:.0f}s]")

        # Per-epoch log
        record = {
            "epoch": epoch,
            "train": {"loss": tr_loss, "acc": tr_acc, "bal_acc": tr_bal, "f1": tr_f1},
            "val":   {"loss": va_loss, "acc": va_acc, "bal_acc": va_bal, "f1": va_f1},
            "lr":    optimizer.param_groups[0]["lr"],
            "elapsed_s": elapsed,
        }
        history.append(record)
        (Path(CFG["log_dir"]) / f"epoch_{epoch:03d}.json").write_text(json.dumps(record, indent=2))

        # Track best
        is_best = False
        if va_bal > best_bal_acc or (va_bal == best_bal_acc and va_f1 > best_f1):
            best_bal_acc, best_f1 = va_bal, va_f1
            is_best    = True
            no_improve = 0
        else:
            no_improve += 1

        # Save checkpoint (every epoch)
        save_checkpoint(model, optimizer, scheduler, scaler,
                        epoch, best_bal_acc, best_f1, history, is_best)

        # Early stopping
        if no_improve >= CFG["early_stop_patience"]:
            print(f"\n[EarlyStop] No improvement for {no_improve} epochs -- stopping.")
            break

    print(f"\n[Train] Done!  Best val bal_acc={best_bal_acc:.4f}  f1={best_f1:.4f}")

    # Final test evaluation using best weights
    print("\n[Eval] Loading best checkpoint for test evaluation ...")
    best_ckpt = Path(CFG["ckpt_dir"]) / "checkpoint_best.pth"
    if best_ckpt.exists():
        best_state = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(best_state["model_state"])

    evaluate_test(model, loader_test)
    plot_training_history(history)
    print("\nAll outputs saved to /kaggle/working/results/")


# --- 8. Evaluation & Visualisation ------------------------------------------

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
    report   = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0
    )

    print(f"\n{'=' * 60}")
    print("  TEST SET RESULTS")
    print(f"{'=' * 60}")
    print(f"  Balanced Accuracy : {bal_acc:.4f}")
    print(f"  Macro F1-Score    : {macro_f1:.4f}")
    print(f"\nPer-class:\n"
          f"{classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0)}")

    # Save outputs
    summary = {"balanced_accuracy": round(bal_acc, 6), "macro_f1": round(macro_f1, 6)}
    (Path(CFG["result_dir"]) / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(report).T.to_csv(Path(CFG["result_dir"]) / "classification_report.csv")

    df_preds = pd.DataFrame({
        "true_label": [CLASS_NAMES[i] for i in all_labels],
        "pred_label": [CLASS_NAMES[i] for i in all_preds],
        "true_idx":   all_labels,
        "pred_idx":   all_preds,
        **{f"prob_{CLASS_NAMES[i]}": [p[i] for p in all_probs]
           for i in range(CFG["num_classes"])}
    })
    df_preds.to_csv(Path(CFG["result_dir"]) / "test_predictions.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(
        f"Multi-Disease Classifier - Test Confusion Matrix\n"
        f"Bal.Acc={bal_acc:.3f}  Macro F1={macro_f1:.3f}", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(Path(CFG["result_dir"]) / "confusion_matrix.png", dpi=150)
    plt.show()

    # Per-class F1 bar chart
    classes = [c for c in CLASS_NAMES if c in report]
    f1s     = [report[c]["f1-score"] for c in classes]
    colors  = ["#4CAF50" if f >= 0.8 else "#FF9800" if f >= 0.6 else "#F44336" for f in f1s]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(classes, f1s, color=colors)
    ax.axhline(0.80, color="green", linestyle="--", linewidth=1.2, label="Target 80%")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class F1 Score on Test Set")
    for bar, f in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{f:.3f}", ha="center", va="bottom", fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(CFG["result_dir"]) / "per_class_f1.png", dpi=150)
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

    axes[0].plot(epochs, tr_loss, label="Train")
    axes[0].plot(epochs, va_loss, label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, tr_bac, label="Train")
    axes[1].plot(epochs, va_bac, label="Val")
    axes[1].axhline(0.80, color="green", linestyle="--", label="Target 80%")
    axes[1].set_title("Balanced Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs, tr_f1, label="Train")
    axes[2].plot(epochs, va_f1, label="Val")
    axes[2].axhline(0.80, color="green", linestyle="--", label="Target 80%")
    axes[2].set_title("Macro F1-Score"); axes[2].set_xlabel("Epoch")
    axes[2].legend(); axes[2].grid(True)

    plt.suptitle("EDD Phase 2 Week 4 - Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(CFG["result_dir"]) / "training_curves.png", dpi=150)
    plt.show()


# --- 9. Entry Point ----------------------------------------------------------

if __name__ == "__main__":
    train()
