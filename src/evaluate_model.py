"""
EDD v2 — Model Evaluation on Slit-Lamp Validation Set
-----------------------------------------------------
Loads the trained EfficientNet-B7 model and evaluates on the slit-lamp
hold-out validation set (data/prepared/val/). Computes full metrics suite
and Grad-CAM heatmaps. Results saved to results/.

Usage:
    python src/evaluate_model.py \
        --model "models/efficientnet_b7_final.h5" \
        --val-dir "data/prepared/val" \
        --output-dir "results" \
        --img-size 600
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

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


def load_model(model_path):
    """Load Keras model from .h5 or SavedModel directory."""
    model_path = Path(model_path)

    if model_path.is_dir() and (model_path / "saved_model.pb").exists():
        print(f"[Loading] SavedModel from {model_path}")
        return keras.models.load_model(str(model_path))

    if model_path.suffix.lower() in (".h5", ".keras"):
        print(f"[Loading] Keras model from {model_path}")
        return keras.models.load_model(str(model_path))

    raise ValueError(f"Unsupported model path: {model_path}")


def preprocess_image(image_path, img_size):
    """Load, resize, normalize — must match training preprocessing exactly."""
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_gradcam_heatmap(model, img_array, class_idx, layer_name=None):
    """
    Grad-CAM heatmap for the last conv layer.
    Returns heatmap (H, W) in [0, 1].
    """
    if layer_name is None:
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate cataract severity model")
    parser.add_argument("--model", required=True, help="Path to .h5/.keras model or SavedModel dir")
    parser.add_argument("--val-dir", required=True, help="Validation directory with class subfolders")
    parser.add_argument("--output-dir", required=True, help="Directory for results output")
    parser.add_argument("--img-size", type=int, default=600, help="Input image size (default 600)")
    parser.add_argument(
        "--class-names", nargs="+", default=["cataract", "normal"],
        help="Class names in alphabetical order (default: cataract normal)"
    )
    parser.add_argument("--gradcam-samples", type=int, default=0,
                        help="Number of Grad-CAM heatmaps to generate (default 0, set >0 to enable)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = args.class_names
    num_classes = len(class_names)

    # Load model
    model = load_model(args.model)
    model.summary()

    # Build validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.val_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(args.img_size, args.img_size),
        batch_size=32,
        shuffle=False,
        interpolation="bilinear",
    )

    # Normalize layer to match training (rescale 1/255)
    normalization_layer = keras.layers.Rescaling(1.0 / 255)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    print(f"\n[Eval] Validation batches: {len(val_ds)}")

    # Run inference on all validation images
    y_true = []
    y_pred = []
    y_prob = []
    image_paths = []
    image_labels = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        y_prob.extend(preds)

    # Collect file paths for Grad-CAM samples
    val_dir = Path(args.val_dir)
    for class_name in class_names:
        class_dir = val_dir / class_name
        if class_dir.exists():
            for img_file in sorted(class_dir.iterdir()):
                image_paths.append(img_file)
                image_labels.append(class_names.index(class_name))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    print(f"[Eval] Total validation images: {len(y_true)}")

    # --- Metrics ---
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=False
    )
    print("\nClassification Report:")
    print(report)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    for i, name in enumerate(class_names):
        print(f"  {name:12s} P={precision[i]:.4f} R={recall[i]:.4f} F1={f1[i]:.4f}")

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    accuracy = np.mean(y_true == y_pred)
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Macro P:   {p_macro:.4f}")
    print(f"Macro R:   {r_macro:.4f}")
    print(f"Macro F1:  {f1_macro:.4f}")

    # ROC-AUC one-vs-rest
    try:
        auc_ovr = roc_auc_score(
            keras.utils.to_categorical(y_true, num_classes=num_classes),
            y_prob,
            multi_class="ovr",
        )
        print(f"ROC-AUC (OVR): {auc_ovr:.4f}")
    except ValueError as e:
        print(f"[WARN] ROC-AUC: {e}")
        auc_ovr = None

    # PR-AUC macro
    try:
        pr_auc = average_precision_score(
            keras.utils.to_categorical(y_true, num_classes=num_classes),
            y_prob,
            average="macro",
        )
        print(f"PR-AUC (macro): {pr_auc:.4f}")
    except ValueError as e:
        print(f"[WARN] PR-AUC: {e}")
        pr_auc = None

    # --- Save results (before Grad-CAM, so metrics are saved even if Grad-CAM is slow) ---
    results = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(Path(args.model).resolve()),
        "val_dir": str(Path(args.val_dir).resolve()),
        "img_size": args.img_size,
        "class_names": class_names,
        "num_classes": num_classes,
        "num_images": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
        "accuracy": float(accuracy),
        "precision_per_class": [float(p) for p in precision],
        "recall_per_class": [float(r) for r in recall],
        "f1_per_class": [float(f) for f in f1],
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "roc_auc_ovr": float(auc_ovr) if auc_ovr is not None else None,
        "pr_auc_macro": float(pr_auc) if pr_auc is not None else None,
        "gradcam": [],
    }

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Evaluation results -> {results_path}")

    # Also save confusion matrix as image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Confusion matrix image -> {cm_path}")

    # --- Grad-CAM heatmaps (optional, can be slow on CPU) ---
    if args.gradcam_samples > 0:
        print(f"\n[Grad-CAM] Generating {args.gradcam_samples} heatmaps...")
        gradcam_dir = output_dir / "gradcam"
        gradcam_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib.cm as cm_mpl

        samples_per_class = max(1, args.gradcam_samples // num_classes)
        selected = []
        class_count = {i: 0 for i in range(num_classes)}
        for idx, (img_path, label) in enumerate(zip(image_paths, image_labels)):
            if class_count[label] < samples_per_class:
                selected.append((idx, img_path, label))
                class_count[label] += 1
            if all(v >= samples_per_class for v in class_count.values()):
                break

        gradcam_results = []
        for idx, (img_idx, img_path, true_label) in enumerate(selected):
            img = preprocess_image(img_path, args.img_size)
            img_batch = np.expand_dims(img.numpy(), axis=0).astype(np.float32)

            pred_prob = model.predict(img_batch, verbose=0)[0]
            pred_label = int(np.argmax(pred_prob))
            pred_conf = float(np.max(pred_prob))

            heatmap = make_gradcam_heatmap(model, img_batch, pred_label)

            orig_img = tf.keras.utils.load_img(
                str(img_path), target_size=(args.img_size, args.img_size)
            )
            orig_arr = np.array(orig_img).astype(np.float32) / 255.0

            heatmap_resized = tf.image.resize(
                heatmap[..., tf.newaxis], (args.img_size, args.img_size)
            ).numpy()[..., 0]

            jet = cm_mpl.get_cmap("jet")
            heatmap_colored = jet(heatmap_resized)[..., :3]

            overlay = 0.6 * heatmap_colored + 0.4 * orig_arr
            overlay = np.clip(overlay, 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(orig_arr)
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(heatmap_resized, cmap="jet")
            axes[1].axis("off")
            axes[2].imshow(overlay)
            axes[2].set_title(
                f"Overlay (pred={class_names[pred_label]}, conf={pred_conf:.2f})"
            )
            axes[2].axis("off")

            gradcam_path = gradcam_dir / (
                f"gradcam_{idx}_true_{class_names[true_label]}_"
                f"pred_{class_names[pred_label]}.png"
            )
            plt.tight_layout()
            plt.savefig(gradcam_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            gradcam_results.append({
                "image": str(img_path),
                "true_label": class_names[true_label],
                "pred_label": class_names[pred_label],
                "confidence": pred_conf,
                "correct": true_label == pred_label,
                "heatmap_path": str(gradcam_path),
            })

        print(f"[Grad-CAM] Saved {len(gradcam_results)} heatmaps to {gradcam_dir}")

        # Update results JSON with gradcam info
        results["gradcam"] = gradcam_results
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
