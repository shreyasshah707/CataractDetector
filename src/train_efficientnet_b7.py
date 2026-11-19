"""
EDD v2 — Kaggle Training Script: EfficientNet-B7 for Cataract Severity Grading
-------------------------------------------------------------------------------
Run this on Kaggle (GPU accelerator). Upload the slit-lamp dataset to the Kaggle
dataset directory or mount via Kaggle API.

Key behaviours (per project decisions D-001 / D-002 / D-004):
- Slit-lamp dataset only, maximize photo count, resolution deprioritized.
- EfficientNet-B7, ImageNet pretrained, fine-tuned for severity classes.
- Checkpoints saved every 1/5/10 epochs; resume from latest on session limit hit.
- Class imbalance NOT handled (per user constraint).
- Whole-photo input (no eye segmentation).
- Exports best model to TFLite (float32).
"""

import os
import json
import glob
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

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
    # Data paths (local: D:\Projects\EDD\v2\data\prepared)
    "train_dir": "/kaggle/input/<your-username>/cataract-slit-lamp-sheemazain/train",
    "val_dir":   "/kaggle/input/<your-username>/cataract-slit-lamp-sheemazain/val",
    # For Kaggle run, change to:
    # "/kaggle/input/<your-username>/cataract-slit-lamp-sheemazain/train"
    # "/kaggle/input/<your-username>/cataract-slit-lamp-sheemazain/val"
    # If you have a single folder, set val_dir to None and use validation_split below
    "use_validation_split": False,
    "validation_split": 0.2,

    # Image / model
    "img_size": (600, 600),          # EfficientNet-B7 default input is 600x600
    "batch_size": 4,                 # Small because B7 is heavy; adjust to your Kaggle VRAM (4-8)
    "num_classes": 2,                # immature, mature (per available dataset)
    "class_names": ["immature", "mature"],  # 2-class: immature vs mature cataract

    # Training
    "epochs": 50,
    "base_learning_rate": 1e-4,
    "fine_tune_learning_rate": 1e-5,
    "fine_tune_at_epoch": 10,        # Unfreeze backbone after this epoch

    # Checkpointing (per user: every 1/5/10 epochs)
    "checkpoint_every": 1,           # 1 = every epoch; set to 5 or 10 as needed
    "checkpoint_dir": "/kaggle/working/checkpoints",
    "log_dir": "/kaggle/working/logs",
    "output_dir": "/kaggle/working/output",

    # Resume
    "resume_from_checkpoint": None,  # e.g. "/kaggle/working/checkpoints/epoch_10.h5"
}

# =============================================================================
# PATH SETUP
# =============================================================================
for d in [CONFIG["checkpoint_dir"], CONFIG["log_dir"], CONFIG["output_dir"]]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================
def get_class_names(data_dir):
    """Return sorted list of class folder names."""
    return sorted([d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))])

def create_data_generators():
    """
    Creates training and validation ImageDataGenerators.
    Expects folder structure: data_dir/class_name/image.jpg
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    if CONFIG["use_validation_split"]:
        # Single directory with split
        train_gen = train_datagen.flow_from_directory(
            CONFIG["train_dir"],
            target_size=CONFIG["img_size"],
            batch_size=CONFIG["batch_size"],
            class_mode="categorical",
            subset="training",
            seed=42,
        )
        val_gen = val_datagen.flow_from_directory(
            CONFIG["train_dir"],
            target_size=CONFIG["img_size"],
            batch_size=CONFIG["batch_size"],
            class_mode="categorical",
            subset="validation",
            seed=42,
        )
    else:
        train_gen = train_datagen.flow_from_directory(
            CONFIG["train_dir"],
            target_size=CONFIG["img_size"],
            batch_size=CONFIG["batch_size"],
            class_mode="categorical",
            seed=42,
        )
        val_gen = val_datagen.flow_from_directory(
            CONFIG["val_dir"],
            target_size=CONFIG["img_size"],
            batch_size=CONFIG["batch_size"],
            class_mode="categorical",
            seed=42,
        )

    # Verify class names match config
    dataset_classes = sorted(train_gen.class_indices.keys(),
                             key=lambda k: train_gen.class_indices[k])
    if dataset_classes != CONFIG["class_names"]:
        print(f"[WARN] Dataset class order {dataset_classes} != config {CONFIG['class_names']}")
        print("       Update CONFIG['class_names'] to match dataset folder order.")

    return train_gen, val_gen


# =============================================================================
# MODEL BUILDING
# =============================================================================
def build_efficientnet_b7(num_classes, img_size, fine_tune=False):
    """
    Builds EfficientNet-B7 with a classification head.
    """
    input_shape = (*img_size, 3)
    base_model = EfficientNetB7(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        pooling="avg",
    )

    if fine_tune:
        base_model.trainable = True
        # Fine-tune from a specific layer onwards (optional; here all layers trainable)
        for layer in base_model.layers:
            layer.trainable = True
    else:
        base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=fine_tune)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)

    return model, base_model


# =============================================================================
# CALLBACKS
# =============================================================================
class EpochCheckpoint(Callback):
    """
    Saves checkpoint every N epochs (user-specified cadence: 1/5/10).
    Also saves 'latest' checkpoint for resume.
    """
    def __init__(self, checkpoint_dir, every=1, save_best=False):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.every = every
        self.save_best = save_best
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = epoch + 1

        # Save latest (for resume)
        latest_path = os.path.join(self.checkpoint_dir, "latest.h5")
        self.model.save(latest_path, include_optimizer=True)

        # Save cadence checkpoint
        if epoch_num % self.every == 0:
            ckpt_path = os.path.join(
                self.checkpoint_dir, f"epoch_{epoch_num:03d}.h5"
            )
            self.model.save(ckpt_path, include_optimizer=True)
            print(f"\n[Checkpoint] Saved {ckpt_path}")

        # Track best
        val_acc = logs.get("val_accuracy", 0)
        if self.save_best and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            best_path = os.path.join(self.checkpoint_dir, "best.h5")
            self.model.save(best_path, include_optimizer=True)
            print(f"[Checkpoint] New best val_accuracy={val_acc:.4f} -> {best_path}")


class ResumeManager:
    """Handles finding and loading the latest checkpoint for resume."""
    @staticmethod
    def find_latest(checkpoint_dir):
        candidates = glob.glob(os.path.join(checkpoint_dir, "epoch_*.h5"))
        if not candidates:
            return None
        # Sort by epoch number in filename
        def epoch_num(p):
            try:
                return int(os.path.basename(p).split("_")[1].split(".")[0])
            except Exception:
                return -1
        candidates.sort(key=epoch_num)
        return candidates[-1]

    @staticmethod
    def load_model_weights(model, path):
        if path and os.path.exists(path):
            model.load_weights(path, by_name=True, skip_mismatch=True)
            print(f"[Resume] Loaded weights from {path}")
            # Extract epoch from filename for logging
            try:
                epoch = int(os.path.basename(path).split("_")[1].split(".")[0])
                print(f"[Resume] Resuming from epoch {epoch}")
                return epoch
            except Exception:
                return 0
        return 0


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_model(model, val_gen, class_names, output_path=None):
    """
    Runs full evaluation suite: confusion matrix, precision/recall/F1,
    ROC-AUC (one-vs-rest), PR curves.
    """
    print("\n=== Running Evaluation ===")

    # Reset generator for full pass
    val_gen.reset()
    steps = int(np.ceil(val_gen.samples / val_gen.batch_size))

    y_true = []
    y_pred = []
    y_prob = []

    for _ in range(steps):
        x_batch, y_batch = next(val_gen)
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        y_prob.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print("\nClassification Report:")
    print(report)

    # Per-class precision/recall/F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    for i, name in enumerate(class_names):
        print(f"  {name:12s} P={precision[i]:.4f} R={recall[i]:.4f} F1={f1[i]:.4f}")

    # Macro averages
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    print(f"\nMacro  Precision={p_macro:.4f} Recall={r_macro:.4f} F1={f1_macro:.4f}")

    # ROC-AUC one-vs-rest (multi-class)
    try:
        auc_ovr = roc_auc_score(
            keras.utils.to_categorical(y_true, num_classes=len(class_names)),
            y_prob,
            multi_class="ovr",
        )
        print(f"ROC-AUC (OVR): {auc_ovr:.4f}")
    except ValueError as e:
        print(f"[WARN] ROC-AUC could not be computed: {e}")
        auc_ovr = None

    # PR-AUC macro
    try:
        pr_auc = average_precision_score(
            keras.utils.to_categorical(y_true, num_classes=len(class_names)),
            y_prob,
            average="macro",
        )
        print(f"PR-AUC (macro): {pr_auc:.4f}")
    except ValueError as e:
        print(f"[WARN] PR-AUC could not be computed: {e}")
        pr_auc = None

    # Save results
    results = {
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "roc_auc_ovr": float(auc_ovr) if auc_ovr is not None else None,
        "pr_auc_macro": float(pr_auc) if pr_auc is not None else None,
        "evaluated_at": datetime.utcnow().isoformat(),
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] Evaluation results -> {output_path}")

    return results


# =============================================================================
# TFLITE EXPORT
# =============================================================================
def export_tflite(model, export_path, quantize=False):
    """
    Exports Keras model to TFLite. float32 by default; int8 optional.
    """
    print(f"\n=== Exporting TFLite to {export_path} ===")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Note: int8 quantization needs a representative dataset for calibration.
        # Skipping calibration here; float16 fallback if needed.
    tflite_model = converter.convert()

    with open(export_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(export_path) / (1024 * 1024)
    print(f"[Exported] TFLite model size: {size_mb:.2f} MB")
    return export_path


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def main():
    print("=" * 60)
    print("EDD v2 — EfficientNet-B7 Cataract Severity Training")
    print("=" * 60)

    # 1. Data
    train_gen, val_gen = create_data_generators()
    print(f"\nTrain samples: {train_gen.samples}, Val samples: {val_gen.samples}")
    print(f"Class indices: {train_gen.class_indices}")

    # 2. Model
    model, base_model = build_efficientnet_b7(
        num_classes=CONFIG["num_classes"],
        img_size=CONFIG["img_size"],
        fine_tune=False,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["base_learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 3. Resume from checkpoint if specified
    start_epoch = 0
    if CONFIG["resume_from_checkpoint"]:
        start_epoch = ResumeManager.load_model_weights(
            model, CONFIG["resume_from_checkpoint"]
        )
    else:
        latest = ResumeManager.find_latest(CONFIG["checkpoint_dir"])
        if latest:
            print(f"[Resume] Latest checkpoint found: {latest}")
            start_epoch = ResumeManager.load_model_weights(model, latest)

    # 4. Callbacks
    checkpoint_cb = EpochCheckpoint(
        CONFIG["checkpoint_dir"],
        every=CONFIG["checkpoint_every"],
        save_best=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
    )

    callbacks = [checkpoint_cb, early_stop_cb]

    # 5. Phase 1 — Train head only (backbone frozen)
    print("\n--- Phase 1: Training classification head (backbone frozen) ---")
    history_1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG["fine_tune_at_epoch"],
        initial_epoch=start_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Phase 2 — Fine-tune backbone
    print("\n--- Phase 2: Fine-tuning EfficientNet-B7 backbone ---")
    model, _ = build_efficientnet_b7(
        num_classes=CONFIG["num_classes"],
        img_size=CONFIG["img_size"],
        fine_tune=True,
    )
    # Re-load latest weights so fine-tuning starts from phase-1 state
    latest = ResumeManager.find_latest(CONFIG["checkpoint_dir"])
    if latest:
        ResumeManager.load_model_weights(model, latest)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["fine_tune_learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG["epochs"],
        initial_epoch=CONFIG["fine_tune_at_epoch"],
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Save final model
    final_path = os.path.join(CONFIG["output_dir"], "efficientnet_b7_final.h5")
    model.save(final_path, include_optimizer=False)
    print(f"\n[Saved] Final Keras model -> {final_path}")

    # 8. Evaluate on validation (slit-lamp hold-out)
    eval_path = os.path.join(CONFIG["output_dir"], "evaluation_slit_lamp.json")
    evaluate_model(model, val_gen, CONFIG["class_names"], output_path=eval_path)

    # 9. Export TFLite
    tflite_path = os.path.join(CONFIG["output_dir"], "efficientnet_b7.tflite")
    export_tflite(model, tflite_path, quantize=False)

    # Also export a copy with a versioned name
    versioned = os.path.join(
        CONFIG["output_dir"],
        f"efficientnet_b7_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.tflite",
    )
    shutil.copy(tflite_path, versioned)
    print(f"[Saved] Versioned TFLite -> {versioned}")

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
