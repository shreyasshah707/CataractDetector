"""
EDD v2 — Training Script: EfficientNet-B0 for Cataract Severity Grading
------------------------------------------------------------------------
Smaller architecture (5.3M params vs 64M for B7) to reduce overfitting
on the small sheemazain dataset (~410 images).

Run locally on CPU or on Kaggle GPU.
"""

import os
import json
import glob
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, EarlyStopping

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
    # Data paths (local: D:\Projects\EDD\v2\data\stage1 - binary normal vs cataract)
    "train_dir": "D:/Projects/EDD/v2/data/stage1/train",
    "val_dir":   "D:/Projects/EDD/v2/data/stage1/val",

    # Image / model
    "img_size": (224, 224),          # EfficientNet-B0 default input is 224x224
    "batch_size": 16,                # B0 is small; larger batch OK on CPU
    "num_classes": 2,                # cataract, normal (alphabetical order)
    "class_names": ["cataract", "normal"],

    # Training
    "epochs": 30,
    "base_learning_rate": 1e-4,
    "fine_tune_learning_rate": 1e-5,
    "fine_tune_at_epoch": 10,

    # Checkpointing
    "checkpoint_every": 5,
    "checkpoint_dir": "models/checkpoints_b0_binary",
    "log_dir": "models/logs_b0_binary",
    "output_dir": "models/b0_binary",

    # Resume
    "resume_from_checkpoint": None,
}

for d in [CONFIG["checkpoint_dir"], CONFIG["log_dir"], CONFIG["output_dir"]]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================
def create_data_generators():
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

    dataset_classes = sorted(train_gen.class_indices.keys(),
                             key=lambda k: train_gen.class_indices[k])
    if dataset_classes != CONFIG["class_names"]:
        print(f"[WARN] Dataset class order {dataset_classes} != config {CONFIG['class_names']}")

    return train_gen, val_gen


# =============================================================================
# MODEL BUILDING
# =============================================================================
def build_efficientnet_b0(num_classes, img_size, fine_tune=False):
    input_shape = (*img_size, 3)
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        pooling="avg",
    )

    if fine_tune:
        base_model.trainable = True
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
    def __init__(self, checkpoint_dir, every=5, save_best=False):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.every = every
        self.save_best = save_best
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = epoch + 1

        latest_path = os.path.join(self.checkpoint_dir, "latest.h5")
        self.model.save(latest_path, include_optimizer=True)

        if epoch_num % self.every == 0:
            ckpt_path = os.path.join(
                self.checkpoint_dir, f"epoch_{epoch_num:03d}.h5"
            )
            self.model.save(ckpt_path, include_optimizer=True)
            print(f"\n[Checkpoint] Saved {ckpt_path}")

        val_acc = logs.get("val_accuracy", 0)
        if self.save_best and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            best_path = os.path.join(self.checkpoint_dir, "best.h5")
            self.model.save(best_path, include_optimizer=True)
            print(f"[Checkpoint] New best val_accuracy={val_acc:.4f} -> {best_path}")


# =============================================================================
# EVALUATION
# =============================================================================
def evaluate_model(model, val_gen, class_names, output_path=None):
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

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
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

    try:
        auc_ovr = roc_auc_score(
            keras.utils.to_categorical(y_true, num_classes=len(class_names)),
            y_prob,
            multi_class="ovr",
        )
        print(f"ROC-AUC (OVR): {auc_ovr:.4f}")
    except ValueError as e:
        print(f"[WARN] ROC-AUC: {e}")
        auc_ovr = None

    try:
        pr_auc = average_precision_score(
            keras.utils.to_categorical(y_true, num_classes=len(class_names)),
            y_prob,
            average="macro",
        )
        print(f"PR-AUC (macro): {pr_auc:.4f}")
    except ValueError as e:
        print(f"[WARN] PR-AUC: {e}")
        pr_auc = None

    results = {
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "accuracy": float(accuracy),
        "precision_per_class": [float(p) for p in precision],
        "recall_per_class": [float(r) for r in recall],
        "f1_per_class": [float(f) for f in f1],
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "roc_auc_ovr": float(auc_ovr) if auc_ovr is not None else None,
        "pr_auc_macro": float(pr_auc) if pr_auc is not None else None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] Evaluation results -> {output_path}")

    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("EDD v2 — EfficientNet-B0 Binary Cataract Detection")
    print("=" * 60)

    # Data
    train_gen, val_gen = create_data_generators()
    print(f"\nTrain samples: {train_gen.samples}, Val samples: {val_gen.samples}")
    print(f"Class indices: {train_gen.class_indices}")

    # Model — unfreeze backbone from the start (single-phase training)
    model, base_model = build_efficientnet_b0(
        num_classes=CONFIG["num_classes"],
        img_size=CONFIG["img_size"],
        fine_tune=True,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG["fine_tune_learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    checkpoint_cb = EpochCheckpoint(
        CONFIG["checkpoint_dir"],
        every=CONFIG["checkpoint_every"],
        save_best=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
    )
    callbacks = [checkpoint_cb, early_stop_cb]

    # Single-phase training
    print("\n--- Training EfficientNet-B0 (fine-tuned from start) ---")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model
    final_path = os.path.join(CONFIG["output_dir"], "efficientnet_b0_binary_final.h5")
    model.save(final_path, include_optimizer=False)
    print(f"\n[Saved] Final Keras model -> {final_path}")

    # Evaluate
    eval_path = os.path.join(CONFIG["output_dir"], "evaluation_slit_lamp.json")
    evaluate_model(model, val_gen, CONFIG["class_names"], output_path=eval_path)

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
