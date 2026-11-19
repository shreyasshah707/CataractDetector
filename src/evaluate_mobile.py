"""
EDD v2 — Mobile Validation Evaluation
-------------------------------------
Evaluates the trained EfficientNet-B7 TFLite model on mobile-captured eye photos.
Mobile photos are NEVER used in training — this is pure validation / domain-gap measurement.

Expected inputs:
- A folder of mobile eye photos
- A labels CSV: image_id, ground_truth_label (normal / incipient / immature / mature / hypermature)

Outputs:
- Confusion matrix, per-class precision/recall/F1, ROC-AUC, PR-AUC
- Grad-CAM heatmaps on sample images
- Comparison with slit-lamp hold-out results (if provided)
"""

import os
import json
import csv
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "tflite_model": "/kaggle/working/output/efficientnet_b7.tflite",
    "mobile_images_dir": "/kaggle/working/mobile_validation/photos",
    "mobile_labels_csv": "/kaggle/working/mobile_validation/labels.csv",
    "slit_lamp_eval_json": "/kaggle/working/output/evaluation_slit_lamp.json",  # optional, for comparison
    "output_dir": "/kaggle/working/output",
    "img_size": (600, 600),
    "class_names": ["normal", "incipient", "immature", "mature", "hypermature"],
    "grad_cam_samples": 5,  # number of sample heatmaps to generate
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================
def load_mobile_labels(csv_path):
    """Returns dict: {image_id: label_index}"""
    labels = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["image_id"].strip()
            label_name = row["ground_truth_label"].strip().lower()
            if label_name in CONFIG["class_names"]:
                labels[img_id] = CONFIG["class_names"].index(label_name)
            else:
                print(f"[WARN] Unknown label '{label_name}' for {img_id}, skipping.")
    return labels


def list_mobile_images(images_dir):
    """Returns list of image file paths."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(images_dir, ext)))
        files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    return sorted(files)


def preprocess_image(image_path, img_size):
    """Load, resize, normalize — must match training preprocessing exactly."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img


# =============================================================================
# GRAD-CAM
# =============================================================================
def make_gradcam_heatmap(model, img_array, class_idx, layer_name=None):
    """
    Simple Grad-CAM for the last conv layer.
    Returns heatmap (H, W) in [0, 1].
    """
    if layer_name is None:
        # Find last conv layer in the base model
        layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                layer_name = layer.name
                break
        if layer_name is None:
            raise ValueError("No Conv2D layer found for Grad-CAM")

    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


# =============================================================================
# MAIN EVALUATION
# =============================================================================
def main():
    print("=" * 60)
    print("EDD v2 — Mobile Validation Evaluation")
    print("=" * 60)

    # 1. Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=CONFIG["tflite_model"])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"[Loaded] TFLite model: {CONFIG['tflite_model']}")

    # 2. Load labels + images
    labels = load_mobile_labels(CONFIG["mobile_labels_csv"])
    image_files = list_mobile_images(CONFIG["mobile_images_dir"])
    print(f"[Found] {len(image_files)} mobile images, {len(labels)} labels")

    # Filter to images that have labels
    matched = []
    for fpath in image_files:
        fname = Path(fpath).stem
        if fname in labels:
            matched.append((fpath, labels[fname]))
        else:
            print(f"[SKIP] No label for {fname}")

    print(f"[Matched] {len(matched)} images with ground truth")

    if not matched:
        print("[ERROR] No labeled mobile images found. Exiting.")
        return

    # 3. Run inference on all mobile images
    y_true = []
    y_pred = []
    y_prob = []
    results = []

    for fpath, true_label in matched:
        img = preprocess_image(fpath, CONFIG["img_size"])
        img = np.expand_dims(img, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        pred_idx = int(np.argmax(output))
        conf = float(np.max(output))

        y_true.append(true_label)
        y_pred.append(pred_idx)
        y_prob.append(output)

        results.append({
            "image": Path(fpath).name,
            "true_label": CONFIG["class_names"][true_label],
            "pred_label": CONFIG["class_names"][pred_idx],
            "confidence": conf,
            "correct": true_label == pred_idx,
        })

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # 4. Metrics
    print("\n=== Mobile Validation Results ===")
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    report = classification_report(
        y_true, y_pred, target_names=CONFIG["class_names"], digits=4
    )
    print("\nClassification Report:")
    print(report)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    for i, name in enumerate(CONFIG["class_names"]):
        print(f"  {name:12s} P={precision[i]:.4f} R={recall[i]:.4f} F1={f1[i]:.4f}")

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    accuracy = np.mean(y_true == y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro   Precision={p_macro:.4f} Recall={r_macro:.4f} F1={f1_macro:.4f}")

    try:
        auc_ovr = roc_auc_score(
            keras.utils.to_categorical(y_true, num_classes=len(CONFIG["class_names"])),
            y_prob,
            multi_class="ovr",
        )
        print(f"ROC-AUC (OVR): {auc_ovr:.4f}")
    except ValueError as e:
        print(f"[WARN] ROC-AUC: {e}")
        auc_ovr = None

    try:
        pr_auc = average_precision_score(
            keras.utils.to_categorical(y_true, num_classes=len(CONFIG["class_names"])),
            y_prob,
            average="macro",
        )
        print(f"PR-AUC (macro): {pr_auc:.4f}")
    except ValueError as e:
        print(f"[WARN] PR-AUC: {e}")
        pr_auc = None

    # 5. Compare with slit-lamp hold-out (domain gap)
    domain_gap = {}
    if os.path.exists(CONFIG["slit_lamp_eval_json"]):
        with open(CONFIG["slit_lamp_eval_json"]) as f:
            slit_eval = json.load(f)
        print("\n=== Domain Gap: Slit-Lamp vs Mobile ===")
        print(f"  Slit-lamp accuracy:  {slit_eval.get('accuracy', 'N/A')}")
        print(f"  Mobile accuracy:     {accuracy:.4f}")
        if slit_eval.get("accuracy") is not None:
            gap = slit_eval["accuracy"] - accuracy
            print(f"  Accuracy drop:       {gap:.4f}")
        domain_gap = {
            "slit_lamp_accuracy": slit_eval.get("accuracy"),
            "mobile_accuracy": float(accuracy),
            "accuracy_drop": slit_eval.get("accuracy") - accuracy
            if slit_eval.get("accuracy") is not None else None,
        }

    # 6. Save results
    output = {
        "evaluated_at": datetime.utcnow().isoformat(),
        "model": CONFIG["tflite_model"],
        "num_images": len(matched),
        "class_names": CONFIG["class_names"],
        "confusion_matrix": cm.tolist(),
        "accuracy": float(accuracy),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "roc_auc_ovr": float(auc_ovr) if auc_ovr is not None else None,
        "pr_auc_macro": float(pr_auc) if pr_auc is not None else None,
        "domain_gap": domain_gap,
        "per_image": results,
    }

    out_path = os.path.join(CONFIG["output_dir"], "evaluation_mobile.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Saved] Mobile evaluation -> {out_path}")

    # 7. Grad-CAM heatmaps on a few samples (requires Keras model, not TFLite)
    print(f"\n=== Grad-CAM (first {CONFIG['grad_cam_samples']} samples) ===")
    print("[INFO] Grad-CAM requires the Keras model (.h5), not TFLite.")
    print("       Run Grad-CAM separately in the training notebook if needed.")

    print("\n=== Mobile Validation Complete ===")


if __name__ == "__main__":
    main()
