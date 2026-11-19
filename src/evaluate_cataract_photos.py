"""
EDD v2 — Evaluation on Real Cataract Photos
-------------------------------------------
Evaluates:
  - data/real_photos_test/cataract (8 confirmed cataract eye images)
  - Full Combined evaluation (9 Normal + 8 Cataract = 17 Real Photos)

Models:
  1. Stage 1: Luminance + SVM Binary Classifier (Normal vs Cataract)
  2. Stage 2: EfficientNet-B0 Severity Grader (Immature vs Mature)
  3. EfficientNet-B0 Binary CNN (Cataract vs Normal)
  4. Two-Stage Cascade Architecture (Stage 1 -> Stage 2)
"""

import os
os.environ["KERAS_HOME"] = r"D:\Projects\EDD\v2\.keras_cache"
os.environ["MPLCONFIGDIR"] = r"D:\Projects\EDD\v2\.mpl_cache"

import csv
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import cv2
from PIL import Image
import joblib

import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Directories & Models
# ---------------------------------------------------------------------------
CATARACT_DATA_DIR = Path("data/real_photos_test/cataract")
NORMAL_DATA_DIR = Path("data/real_photos_test")
OUTPUT_DIR = Path("results/real_photos_test/cataract")
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
ROI_DIR = OUTPUT_DIR / "roi_masks"

STAGE1_SVM_PATH = "models/stage1/stage1_svm.pkl"
STAGE1_PARAMS_PATH = "models/stage1/stage1_params.json"
B0_SEVERITY_PATH = "models/efficientnet_b0_best.h5"
B0_BINARY_PATH = "models/checkpoints_b0_binary/best.h5"

IMG_SIZE = (224, 224)
STAGE1_CLASSES = ["normal", "cataract"]
SEVERITY_CLASSES = ["immature", "mature"]
BINARY_CLASSES = ["cataract", "normal"]

for d in [OUTPUT_DIR, GRADCAM_DIR, ROI_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------
def median_filter(img, ksize=5):
    return cv2.medianBlur(img, ksize)


def watershed_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    opened = cv2.morphologyEx(grad, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, markers = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    markers = markers.astype(np.uint8)
    markers = cv2.connectedComponents(markers)[1] + 1
    markers[thresh == 0] = 0
    markers = cv2.watershed(img, markers)
    lens_mask = (markers > 1).astype(np.uint8) * 255
    return lens_mask


def extract_luminance(img_rgb, img_size=(224, 224)):
    img_uint8 = cv2.resize(img_rgb, img_size, interpolation=cv2.INTER_LINEAR)
    img_filtered = median_filter(img_uint8, ksize=5)
    lens_mask = watershed_segmentation(img_filtered)

    img_float = img_filtered.astype(np.float32) / 255.0
    R, G, B = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    BV = 0.299 * R + 0.587 * G + 0.114 * B
    luminance = np.power(2.0, BV) / 0.0929

    lens_mask_bool = lens_mask > 0
    fallback = False
    if np.sum(lens_mask_bool) == 0:
        fallback = True
        h, w = luminance.shape
        center_mask = np.zeros((h, w), dtype=bool)
        center_mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        lens_mask_bool = center_mask
        lens_mask = center_mask.astype(np.uint8) * 255

    mean_lum = float(np.mean(luminance[lens_mask_bool]))
    std_lum = float(np.std(luminance[lens_mask_bool]))

    overlay = img_uint8.copy()
    overlay[lens_mask > 0] = (overlay[lens_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    side_by_side = np.hstack([img_uint8, overlay, cv2.cvtColor(lens_mask, cv2.COLOR_GRAY2RGB)])

    return mean_lum, std_lum, fallback, side_by_side

# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def compute_gradcam(model, img_array, class_idx):
    try:
        base_model = model.get_layer("efficientnetb0")
        last_conv = base_model.get_layer("top_conv")
        sub_base = keras.Model(inputs=base_model.inputs, outputs=[last_conv.output, base_model.output])

        classifier_layers = []
        found_base = False
        for l in model.layers:
            if found_base:
                classifier_layers.append(l)
            if l.name == "efficientnetb0":
                found_base = True

        with tf.GradientTape() as tape:
            conv_out, pooled_out = sub_base(img_array)
            tape.watch(conv_out)
            x = pooled_out
            for l in classifier_layers:
                x = l(x, training=False)
            loss = x[:, class_idx]

        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"[Grad-CAM Error]: {e}")
        return None


def generate_gradcam_overlay(orig_pil, heatmap, img_size=(224, 224)):
    if heatmap is None:
        return None
    orig = orig_pil.convert("RGB").resize(img_size, Image.Resampling.BILINEAR)
    orig_arr = np.array(orig).astype(np.float32) / 255.0

    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], img_size).numpy()[..., 0]
    jet = matplotlib.colormaps["jet"]
    heatmap_colored = jet(heatmap_resized)[..., :3]

    overlay = 0.6 * heatmap_colored + 0.4 * orig_arr
    overlay = np.clip(overlay, 0, 1)
    overlay_img = (overlay * 255).astype(np.uint8)
    return Image.fromarray(overlay_img)

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def evaluate_image(img_path, svm_model, b0_severity_model, b0_binary_model, stage1_params, ground_truth):
    pil_img = Image.open(img_path).convert("RGB")
    img_rgb = np.array(pil_img)
    orig_w, orig_h = pil_img.size

    pil_resized = pil_img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_arr = np.expand_dims(np.array(pil_resized).astype(np.float32) / 255.0, axis=0)

    # Stage 1
    mean_lum, std_lum, fallback, roi_vis = extract_luminance(img_rgb, IMG_SIZE)
    s1_pred = int(svm_model.predict([[mean_lum]])[0])
    s1_decision = float(svm_model.decision_function([[mean_lum]])[0])
    s1_label = STAGE1_CLASSES[s1_pred]
    s1_correct = (s1_label == ground_truth)

    # Save ROI vis
    safe_stem = img_path.stem.replace(" ", "_")
    roi_save_path = ROI_DIR / f"roi_{safe_stem}.png"
    cv2.imwrite(str(roi_save_path), cv2.cvtColor(roi_vis, cv2.COLOR_RGB2BGR))

    # Stage 2 Severity
    sev_probs = b0_severity_model.predict(img_arr, verbose=0)[0]
    sev_pred = int(np.argmax(sev_probs))
    sev_conf = float(np.max(sev_probs))
    sev_label = SEVERITY_CLASSES[sev_pred]
    p_immature = float(sev_probs[0])
    p_mature = float(sev_probs[1])

    sev_cam = compute_gradcam(b0_severity_model, img_arr, sev_pred)
    sev_overlay = generate_gradcam_overlay(pil_img, sev_cam, IMG_SIZE)
    sev_cam_path = None
    if sev_overlay is not None:
        sev_cam_path = GRADCAM_DIR / f"gradcam_severity_{safe_stem}.png"
        sev_overlay.save(sev_cam_path)

    # B0 Binary
    bin_probs = b0_binary_model.predict(img_arr, verbose=0)[0]
    bin_pred = int(np.argmax(bin_probs))
    bin_conf = float(np.max(bin_probs))
    bin_label = BINARY_CLASSES[bin_pred]
    bin_correct = (bin_label == ground_truth)
    p_cataract = float(bin_probs[0])
    p_normal = float(bin_probs[1])

    bin_cam = compute_gradcam(b0_binary_model, img_arr, bin_pred)
    bin_overlay = generate_gradcam_overlay(pil_img, bin_cam, IMG_SIZE)
    bin_cam_path = None
    if bin_overlay is not None:
        bin_cam_path = GRADCAM_DIR / f"gradcam_binary_{safe_stem}.png"
        bin_overlay.save(bin_cam_path)

    # Two-Stage Pipeline
    if s1_label == "normal":
        two_stage_status = "No Cataract Detected (Normal)"
        two_stage_severity = None
        two_stage_correct = (ground_truth == "normal")
    else:
        two_stage_status = f"Cataract Detected ({sev_label.title()})"
        two_stage_severity = sev_label
        two_stage_correct = (ground_truth == "cataract")

    return {
        "image": img_path.name,
        "ground_truth": ground_truth,
        "resolution": {"width": orig_w, "height": orig_h},
        "stage1_luminance_svm": {
            "mean_luminance": round(mean_lum, 3),
            "std_luminance": round(std_lum, 3),
            "decision_function_score": round(s1_decision, 4),
            "prediction": s1_label,
            "correct": s1_correct,
            "fallback_center_roi_used": fallback,
            "roi_visualization": str(roi_save_path.relative_to(OUTPUT_DIR))
        },
        "stage2_severity_efficientnet_b0": {
            "prediction": sev_label,
            "confidence": round(sev_conf, 4),
            "probabilities": {
                "immature": round(p_immature, 4),
                "mature": round(p_mature, 4)
            },
            "gradcam_overlay": str(sev_cam_path.relative_to(OUTPUT_DIR)) if sev_cam_path else None
        },
        "efficientnet_b0_binary_model": {
            "prediction": bin_label,
            "confidence": round(bin_conf, 4),
            "probabilities": {
                "cataract": round(p_cataract, 4),
                "normal": round(p_normal, 4)
            },
            "correct": bin_correct,
            "gradcam_overlay": str(bin_cam_path.relative_to(OUTPUT_DIR)) if bin_cam_path else None
        },
        "two_stage_pipeline": {
            "final_decision": two_stage_status,
            "stage1_gate": s1_label,
            "stage2_severity": two_stage_severity,
            "correct": two_stage_correct
        }
    }


def main():
    print("=================================================================")
    print("EDD v2 — Evaluation on Real Cataract Photos")
    print("=================================================================")

    # 1. Load models
    svm_model = joblib.load(STAGE1_SVM_PATH)
    with open(STAGE1_PARAMS_PATH, "r", encoding="utf-8") as f:
        stage1_params = json.load(f)

    b0_severity_model = keras.models.load_model(B0_SEVERITY_PATH)
    b0_binary_model = keras.models.load_model(B0_BINARY_PATH)

    # 2. Cataract photos
    valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"]
    cataract_files = sorted([p for p in CATARACT_DATA_DIR.iterdir() if p.suffix.lower() in valid_exts])
    print(f"Found {len(cataract_files)} cataract photos in {CATARACT_DATA_DIR}")

    cataract_results = []
    for idx, p in enumerate(cataract_files, 1):
        print(f"Processing Cataract [{idx}/{len(cataract_files)}]: {p.name}")
        rec = evaluate_image(p, svm_model, b0_severity_model, b0_binary_model, stage1_params, "cataract")
        cataract_results.append(rec)
        print(f"  Mean Lum: {rec['stage1_luminance_svm']['mean_luminance']:.2f}")
        print(f"  Stage 1 SVM: {rec['stage1_luminance_svm']['prediction'].title()} (Score: {rec['stage1_luminance_svm']['decision_function_score']:.3f})")
        print(f"  Two-Stage Decision: {rec['two_stage_pipeline']['final_decision']}")
        print(f"  B0 Binary: {rec['efficientnet_b0_binary_model']['prediction'].title()} ({rec['efficientnet_b0_binary_model']['confidence']:.1%})")

    # Metrics for cataract subset
    cat_total = len(cataract_results)
    s1_cat_correct = sum(1 for r in cataract_results if r["stage1_luminance_svm"]["correct"])
    pipe_cat_correct = sum(1 for r in cataract_results if r["two_stage_pipeline"]["correct"])
    bin_cat_correct = sum(1 for r in cataract_results if r["efficientnet_b0_binary_model"]["correct"])

    cat_metrics = {
        "total_cataract_samples": cat_total,
        "stage1_svm_recall_sensitivity": round(s1_cat_correct / cat_total, 4),
        "two_stage_pipeline_recall_sensitivity": round(pipe_cat_correct / cat_total, 4),
        "b0_binary_recall_sensitivity": round(bin_cat_correct / cat_total, 4),
        "stage1_correct_count": s1_cat_correct,
        "two_stage_correct_count": pipe_cat_correct,
        "b0_binary_correct_count": bin_cat_correct
    }

    # Save cataract results JSON
    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": cat_metrics,
            "results": cataract_results
        }, f, indent=2)

    # Save cataract results CSV
    with open(OUTPUT_DIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image", "ground_truth", "mean_luminance", "stage1_prediction", "stage1_correct",
            "two_stage_decision", "two_stage_correct", "stage2_severity", "b0_binary_prediction", "b0_binary_correct"
        ])
        writer.writeheader()
        for r in cataract_results:
            writer.writerow({
                "image": r["image"],
                "ground_truth": r["ground_truth"],
                "mean_luminance": r["stage1_luminance_svm"]["mean_luminance"],
                "stage1_prediction": r["stage1_luminance_svm"]["prediction"],
                "stage1_correct": "YES" if r["stage1_luminance_svm"]["correct"] else "NO (False Neg.)",
                "two_stage_decision": r["two_stage_pipeline"]["final_decision"],
                "two_stage_correct": "YES" if r["two_stage_pipeline"]["correct"] else "NO (False Neg.)",
                "stage2_severity": r["stage2_severity_efficientnet_b0"]["prediction"],
                "b0_binary_prediction": r["efficientnet_b0_binary_model"]["prediction"],
                "b0_binary_correct": "YES" if r["efficientnet_b0_binary_model"]["correct"] else "NO (False Neg.)"
            })

    # Generate Cataract Summary Markdown
    with open(OUTPUT_DIR / "SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("# Cataract Photos Test Report (`data/real_photos_test/cataract`)\n\n")
        f.write(f"**Total Samples:** {cat_total} confirmed Cataract eyes  \n")
        f.write(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  \n\n")

        f.write("## 1. Sensitivity / Recall on Cataract Photos\n\n")
        f.write("| Model / Pipeline | Total Cataracts | Correctly Detected (TP) | Missed (FN) | Sensitivity (Recall) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Stage 1 (Luminance + SVM)** | {cat_total} | {s1_cat_correct} | {cat_total - s1_cat_correct} | **{cat_metrics['stage1_svm_recall_sensitivity']:.1%}** |\n")
        f.write(f"| **Two-Stage Pipeline** | {cat_total} | {pipe_cat_correct} | {cat_total - pipe_cat_correct} | **{cat_metrics['two_stage_pipeline_recall_sensitivity']:.1%}** |\n")
        f.write(f"| **EfficientNet-B0 Binary CNN** | {cat_total} | {bin_cat_correct} | {cat_total - bin_cat_correct} | **{cat_metrics['b0_binary_recall_sensitivity']:.1%}** |\n\n")

        f.write("## 2. Per-Image Breakdown\n\n")
        f.write("| Image | Mean Lum. | Stage 1 SVM | Two-Stage Pipeline Output | Stage 2 Severity | B0 Binary CNN |\n")
        f.write("| :--- | :---: | :---: | :--- | :--- | :--- |\n")
        for r in cataract_results:
            img = r["image"]
            lum = r["stage1_luminance_svm"]["mean_luminance"]
            s1_p = "PASS (Cataract)" if r["stage1_luminance_svm"]["correct"] else "FAIL (Missed Normal)"
            pipe_p = r["two_stage_pipeline"]["final_decision"]
            s2_p = f"{r['stage2_severity_efficientnet_b0']['prediction'].title()} ({r['stage2_severity_efficientnet_b0']['confidence']:.1%})"
            bin_p = f"{r['efficientnet_b0_binary_model']['prediction'].title()} ({r['efficientnet_b0_binary_model']['confidence']:.1%})"
            f.write(f"| `{img}` | {lum:.2f} | **{s1_p}** | {pipe_p} | {s2_p} | {bin_p} |\n")

    # -------------------------------------------------------------
    # Combined Overall Evaluation (9 Normal + 8 Cataract = 17 Photos)
    # -------------------------------------------------------------
    with open(NORMAL_DATA_DIR.parent.parent / "results/real_photos_test/results.json", "r", encoding="utf-8") as f:
        normal_data = json.load(f)
    normal_results = normal_data["results"]

    all_results = normal_results + cataract_results
    total_samples = len(all_results)
    
    # Calculate confusion matrix for Stage 1 SVM
    s1_tp = sum(1 for r in all_results if r["ground_truth"] == "cataract" and r["stage1_luminance_svm"]["prediction"] == "cataract")
    s1_tn = sum(1 for r in all_results if r["ground_truth"] == "normal" and r["stage1_luminance_svm"]["prediction"] == "normal")
    s1_fp = sum(1 for r in all_results if r["ground_truth"] == "normal" and r["stage1_luminance_svm"]["prediction"] == "cataract")
    s1_fn = sum(1 for r in all_results if r["ground_truth"] == "cataract" and r["stage1_luminance_svm"]["prediction"] == "normal")

    s1_acc = (s1_tp + s1_tn) / total_samples
    s1_sens = s1_tp / (s1_tp + s1_fn) if (s1_tp + s1_fn) > 0 else 0
    s1_spec = s1_tn / (s1_tn + s1_fp) if (s1_tn + s1_fp) > 0 else 0
    s1_prec = s1_tp / (s1_tp + s1_fp) if (s1_tp + s1_fp) > 0 else 0
    s1_f1 = 2 * s1_prec * s1_sens / (s1_prec + s1_sens) if (s1_prec + s1_sens) > 0 else 0

    # Binary CNN
    b0_tp = sum(1 for r in all_results if r["ground_truth"] == "cataract" and r["efficientnet_b0_binary_model"]["prediction"] == "cataract")
    b0_tn = sum(1 for r in all_results if r["ground_truth"] == "normal" and r["efficientnet_b0_binary_model"]["prediction"] == "normal")
    b0_fp = sum(1 for r in all_results if r["ground_truth"] == "normal" and r["efficientnet_b0_binary_model"]["prediction"] == "cataract")
    b0_fn = sum(1 for r in all_results if r["ground_truth"] == "cataract" and r["efficientnet_b0_binary_model"]["prediction"] == "normal")

    b0_acc = (b0_tp + b0_tn) / total_samples
    b0_sens = b0_tp / (b0_tp + b0_fn) if (b0_tp + b0_fn) > 0 else 0
    b0_spec = b0_tn / (b0_tn + b0_fp) if (b0_tn + b0_fp) > 0 else 0
    b0_prec = b0_tp / (b0_tp + b0_fp) if (b0_tp + b0_fp) > 0 else 0
    b0_f1 = 2 * b0_prec * b0_sens / (b0_prec + b0_sens) if (b0_prec + b0_sens) > 0 else 0

    # Save Combined Master Report
    comb_report_path = Path("results/real_photos_test/COMBINED_17_PHOTOS_REPORT.md")
    with open(comb_report_path, "w", encoding="utf-8") as f:
        f.write("# EDD v2 — Full Benchmark Report on Real Eye Photos\n\n")
        f.write(f"**Total Real Photos Evaluated:** {total_samples} (9 Normal + 8 Cataract)\n")
        f.write(f"**Report Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")

        f.write("## 1. Overall Performance Comparison\n\n")
        f.write("| Architecture | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Stage 1 (Luminance + SVM)** | **{s1_acc:.1%}** ({s1_tp + s1_tn}/{total_samples}) | **{s1_sens:.1%}** ({s1_tp}/8) | **{s1_spec:.1%}** ({s1_tn}/9) | **{s1_prec:.1%}** | **{s1_f1:.1%}** |\n")
        f.write(f"| **Two-Stage Pipeline** | **{s1_acc:.1%}** ({s1_tp + s1_tn}/{total_samples}) | **{s1_sens:.1%}** ({s1_tp}/8) | **{s1_spec:.1%}** ({s1_tn}/9) | **{s1_prec:.1%}** | **{s1_f1:.1%}** |\n")
        f.write(f"| **EfficientNet-B0 Binary CNN** | **{b0_acc:.1%}** ({b0_tp + b0_tn}/{total_samples}) | **{b0_sens:.1%}** ({b0_tp}/8) | **{b0_spec:.1%}** ({b0_tn}/9) | **{b0_prec:.1%}** | **{b0_f1:.1%}** |\n\n")

        f.write("## 2. Confusion Matrices\n\n")
        f.write("### Stage 1 (Luminance + SVM):\n")
        f.write(f"- **True Negatives (Normal correctly identified):** {s1_tn} / 9\n")
        f.write(f"- **False Positives (Normal falsely flagged as Cataract):** {s1_fp} / 9\n")
        f.write(f"- **True Positives (Cataract correctly identified):** {s1_tp} / 8\n")
        f.write(f"- **False Negatives (Cataract missed):** {s1_fn} / 8\n\n")

        f.write("### EfficientNet-B0 Binary CNN:\n")
        f.write(f"- **True Negatives (Normal correctly identified):** {b0_tn} / 9\n")
        f.write(f"- **False Positives (Normal falsely flagged as Cataract):** {b0_fp} / 9\n")
        f.write(f"- **True Positives (Cataract correctly identified):** {b0_tp} / 8\n")
        f.write(f"- **False Negatives (Cataract missed):** {b0_fn} / 8\n\n")

    print("\nCompleted successfully!")
    print(f"Cataract reports saved to: {OUTPUT_DIR}")
    print(f"Combined report saved to: {comb_report_path}")

if __name__ == "__main__":
    main()
