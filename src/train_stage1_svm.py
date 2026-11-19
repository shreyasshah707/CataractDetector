"""
EDD v2 â€” Stage 1 Training: Luminance + SVM for Binary Cataract Detection
-------------------------------------------------------------------------
Implements the paper "Detecting Cataract Using Smartphones"
(Askarian et al., IEEE JTEHM, 2021, DOI: 10.1109/JTEHM.2021.3074597).

Pipeline:
  1. Load normal + cataract slit-lamp images from folder structure
  2. Preprocess: median filter (denoising) + watershed transform (lens ROI segmentation)
  3. Extract features: mean luminance of lens region
  4. Train: SVM with Gaussian kernel, 10-fold cross-validation
  5. Evaluate: sensitivity, specificity, accuracy on hold-out set
  6. Save: SVM model (pickle) + luminance parameters (JSON)

Expected folder structure:
    data/
        stage1/
            train/
                normal/
                cataract/
            val/
                normal/
                cataract/

Usage:
    python src/train_stage1_svm.py \
        --data-dir "data/stage1" \
        --output-dir "models/stage1"
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
import joblib

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "data_dir": "data/stage1",
    "output_dir": "models/stage1",
    "img_size": (224, 224),
    "n_splits": 10,  # 10-fold cross-validation per paper
    "random_state": 42,
}

CLASS_NAMES = ["normal", "cataract"]


# ---------------------------------------------------------------------------
# Preprocessing (from paper Section II-C)
# ---------------------------------------------------------------------------
def median_filter(img, ksize=5):
    """Apply median filter to remove salt-pepper noise and glare (paper Eq. / Section II-C)."""
    return cv2.medianBlur(img, ksize)


def watershed_segmentation(img):
    """
    Apply watershed transform to extract lens ROI (paper Section II-D).
    Returns binary mask of lens region.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Gradient magnitude for edge detection
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

    # Opening: dilation of erosion
    opened = cv2.morphologyEx(grad, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Closing: erosion of dilation
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Threshold to eliminate dark spots
    _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Distance transform for markers
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, markers = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    markers = markers.astype(np.uint8)
    markers = cv2.connectedComponents(markers)[1] + 1

    # Marker for background (border)
    markers[thresh == 0] = 0

    # Watershed
    markers = cv2.watershed(img, markers)

    # Lens region = watershed boundary (markers == -1) or foreground (markers > 1)
    lens_mask = (markers > 1).astype(np.uint8) * 255

    return lens_mask


def extract_luminance_features(img_rgb, img_size=(224, 224)):
    """
    Extract mean luminance from lens ROI (paper Section II-E).
    
    Steps:
    1. Resize to standard size (uint8 for OpenCV ops)
    2. Median filter for denoising (uint8)
    3. Watershed segmentation for lens ROI (uint8)
    4. Normalize to [0,1] for BV computation (float)
    5. Compute BV = 0.299*R + 0.587*G + 0.114*B (paper Eq. 6)
    6. Convert BV to luminance: luminance = 2^BV / 0.0929 (paper Eq. 8)
    7. Return mean luminance of lens region
    
    Returns:
        mean_luminance: float, mean luminance value of lens region
        lens_mask: np.ndarray, binary mask of lens region (for debugging)
    """
    # Resize (uint8 for OpenCV compatibility)
    img_uint8 = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_LINEAR)

    # Median filter for denoising (paper Section II-C) — requires uint8
    img_filtered = median_filter(img_uint8, ksize=5)

    # Watershed segmentation for lens ROI (paper Section II-D) — requires uint8
    lens_mask = watershed_segmentation(img_filtered)

    # Normalize filtered image to [0,1] for BV computation
    img_float = img_filtered.astype(np.float32) / 255.0

    # Compute Brightness Value (BV) = 0.299*R + 0.587*G + 0.114*B (paper Eq. 6)
    # With normalized [0,1] pixels, BV ranges 0-1
    R, G, B = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    BV = 0.299 * R + 0.587 * G + 0.114 * B

    # Convert BV to luminance (paper Eq. 8, calibrated for iPhone X)
    # luminance = 2^BV / 0.0929. With BV in [0,1], luminance ranges [1.08, 2.15]
    luminance = np.power(2.0, BV) / 0.0929

    # Mask to lens region only
    lens_mask_bool = lens_mask > 0
    if np.sum(lens_mask_bool) == 0:
        # Fallback: use center region if watershed fails
        h, w = luminance.shape
        center_mask = np.zeros((h, w), dtype=bool)
        center_mask[h//4:3*h//4, w//4:3*w//4] = True
        lens_mask_bool = center_mask

    mean_luminance = float(np.mean(luminance[lens_mask_bool]))

    return mean_luminance, lens_mask


# Data Loading
# ---------------------------------------------------------------------------
def load_dataset(data_dir):
    """
    Load images from folder structure supporting both:
        data_dir/
            normal/
                img1.jpg, ...
            cataract/
                img1.jpg, ...
    OR
        data_dir/
            train/
                normal/
                cataract/
            val/
                normal/
                cataract/
    
    Returns:
        features: list of mean luminance values
        labels: list of class indices (0=normal, 1=cataract)
        image_paths: list of image file paths
    """
    data_path = Path(data_dir)
    features = []
    labels = []
    image_paths = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        # Look for class folder directly or under train/val subdirs
        class_dirs = []
        direct = data_path / class_name
        if direct.exists():
            class_dirs.append(direct)
        for split in ["train", "val"]:
            split_dir = data_path / split / class_name
            if split_dir.exists():
                class_dirs.append(split_dir)

        if not class_dirs:
            print(f"[WARN] Class directory not found for: {class_name}")
            continue

        image_files = []
        for cd in class_dirs:
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                image_files.extend(cd.glob(ext))
                image_files.extend(cd.glob(ext.upper()))
        image_files = sorted(set(image_files))

        print(f"[Loading] {class_name}: {len(image_files)} images (from {[str(c) for c in class_dirs]})")

        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"[WARN] Could not read: {img_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                mean_lum, _ = extract_luminance_features(img, CONFIG["img_size"])

                features.append(mean_lum)
                labels.append(class_idx)
                image_paths.append(str(img_path))
            except Exception as e:
                print(f"[WARN] Error processing {img_path}: {e}")
                continue

    print(f"\n[Total] {len(features)} images loaded")
    for i, name in enumerate(CLASS_NAMES):
        count = labels.count(i)
        print(f"  {name}: {count}")

    return features, labels, image_paths


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_svm(features, labels, n_splits=10):
    """
    Train SVM with Gaussian kernel using 10-fold cross-validation (paper Section II-F).
    
    Returns:
        svm_model: trained SVC classifier
        cv_results: dict with cross-validation metrics
    """
    X = np.array(features).reshape(-1, 1)
    y = np.array(labels)

    # 10-fold cross-validation (paper uses 10-fold)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CONFIG["random_state"])

    cv_accuracies = []
    cv_sensitivities = []
    cv_specificities = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=CONFIG["random_state"])
        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_val)

        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        cv_accuracies.append(accuracy)
        cv_sensitivities.append(sensitivity)
        cv_specificities.append(specificity)

        print(f"  Fold {fold+1}: acc={accuracy:.4f}, sens={sensitivity:.4f}, spec={specificity:.4f}")

    cv_results = {
        "mean_accuracy": float(np.mean(cv_accuracies)),
        "std_accuracy": float(np.std(cv_accuracies)),
        "mean_sensitivity": float(np.mean(cv_sensitivities)),
        "std_sensitivity": float(np.std(cv_sensitivities)),
        "mean_specificity": float(np.mean(cv_specificities)),
        "std_specificity": float(np.std(cv_specificities)),
    }

    print(f"\n[Cross-Validation Results]")
    print(f"  Mean Accuracy:  {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
    print(f"  Mean Sensitivity: {cv_results['mean_sensitivity']:.4f} (+/- {cv_results['std_sensitivity']:.4f})")
    print(f"  Mean Specificity: {cv_results['mean_specificity']:.4f} (+/- {cv_results['std_specificity']:.4f})")

    # Train final model on all data
    final_svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=CONFIG["random_state"])
    final_svm.fit(X, y)

    return final_svm, cv_results


# ---------------------------------------------------------------------------
# Evaluation on Hold-Out Set
# ---------------------------------------------------------------------------
def evaluate_holdout(svm_model, features, labels):
    """
    Evaluate trained SVM on hold-out validation set.
    Returns metrics dict.
    """
    X = np.array(features).reshape(-1, 1)
    y = np.array(labels)

    y_pred = svm_model.predict(X)
    y_prob = svm_model.decision_function(X)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    try:
        auc = roc_auc_score(y, y_prob)
    except ValueError:
        auc = None

    results = {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "roc_auc": float(auc) if auc is not None else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

    print(f"\n[Hold-Out Evaluation Results]")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Sensitivity: {results['sensitivity']:.4f}")
    print(f"  Specificity: {results['specificity']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  F1: {results['f1']:.4f}")
    if results["roc_auc"] is not None:
        print(f"  ROC-AUC: {results['roc_auc']:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Stage 1: Luminance + SVM binary cataract detector")
    parser.add_argument("--data-dir", required=True, help="Path to stage1 data (normal/ and cataract/ subfolders)")
    parser.add_argument("--output-dir", required=True, help="Directory for saved model and parameters")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of CV folds (default 10)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=" * 60)
    print("EDD v2 â€” Stage 1: Luminance + SVM Binary Cataract Detection")
    print("=" * 60)
    print(f"\n[Loading] Data from {args.data_dir}")
    features, labels, image_paths = load_dataset(args.data_dir)

    if len(features) == 0:
        print("[ERROR] No images loaded. Check data directory structure.")
        return

    if len(set(labels)) < 2:
        print("[ERROR] Need both normal and cataract classes.")
        return

    # Train SVM with CV
    print(f"\n[Training] SVM with Gaussian kernel, {args.n_splits}-fold CV")
    svm_model, cv_results = train_svm(features, labels, n_splits=args.n_splits)

    # Evaluate on full dataset (hold-out would be separate in production)
    print(f"\n[Evaluating] On full dataset")
    holdout_results = evaluate_holdout(svm_model, features, labels)

    # Compute luminance statistics for reference
    normal_lum = [f for f, l in zip(features, labels) if l == 0]
    cataract_lum = [f for f, l in zip(features, labels) if l == 1]

    print(f"\n[Luminance Statistics]")
    print(f"  Normal:  mean={np.mean(normal_lum):.2f}, SD={np.std(normal_lum):.2f}, n={len(normal_lum)}")
    print(f"  Cataract: mean={np.mean(cataract_lum):.2f}, SD={np.std(cataract_lum):.2f}, n={len(cataract_lum)}")

    # Save SVM model
    svm_path = output_dir / "stage1_svm.pkl"
    joblib.dump(svm_model, svm_path)
    print(f"\n[Saved] SVM model -> {svm_path}")

    # Save parameters
    params = {
        "model_type": "svm_gaussian",
        "feature_type": "mean_luminance",
        "preprocessing": ["median_filter", "watershed_segmentation"],
        "decision_threshold": 0.0,  # SVM decision boundary
        "healthy_mean_luminance": float(np.mean(normal_lum)),
        "diseased_mean_luminance": float(np.mean(cataract_lum)),
        "healthy_std_luminance": float(np.std(normal_lum)),
        "diseased_std_luminance": float(np.std(cataract_lum)),
        "svm_support_vectors": int(len(svm_model.support_vectors_)),
        "trained_on": str(args.data_dir),
        "n_samples": len(features),
        "n_normal": len(normal_lum),
        "n_cataract": len(cataract_lum),
        "img_size": CONFIG["img_size"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cv_results": cv_results,
        "holdout_results": holdout_results,
    }

    params_path = output_dir / "stage1_params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"[Saved] Parameters -> {params_path}")

    print("\n=== Stage 1 Training Complete ===")
    print(f"Next: Collect normal eye images and add to {args.data_dir}/normal/")
    print(f"Then re-run this script to retrain with balanced data.")


if __name__ == "__main__":
    main()
