import json

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}


def md_cell(lines):
    return {
        "cell_type": "markdown",
        "id": f"md-{len(nb['cells'])}",
        "metadata": {},
        "source": [l + "\n" for l in lines],
    }


def code_cell(lines):
    return {
        "cell_type": "code",
        "id": f"code-{len(nb['cells'])}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


# Markdown title cell
nb["cells"].append(
    md_cell(
        [
            "# EDD v2 — EfficientNet-B7 Cataract Severity Training (Kaggle)",
            "",
            "**Dataset:** Kaggle sheemazain (slit-lamp, immature vs mature, 2 classes)",
            "**Model:** EfficientNet-B7, ImageNet pretrained, fine-tuned",
            "**Checkpoints:** Every epoch + latest for resume",
            "**Export:** TFLite float32 for Android research app",
            "",
            "## Before running",
            "1. Add your private Kaggle dataset to this notebook: Add Data → Dataset → search your dataset",
            "2. Update dataset_slug below to match your Kaggle dataset path",
            "3. Set accelerator to GPU in notebook settings",
            "4. This trains on 2 classes only: immature vs mature (no normal class)",
        ]
    )
)

# Step 1: Config
nb["cells"].append(
    code_cell(
        [
            "# STEP 1: Setup & Configuration",
            "import os",
            "import json",
            "import glob",
            "import shutil",
            "from datetime import datetime",
            "from pathlib import Path",
            "",
            "import numpy as np",
            "import tensorflow as tf",
            "from tensorflow import keras",
            "from tensorflow.keras import layers, Model",
            "from tensorflow.keras.applications import EfficientNetB7",
            "from tensorflow.keras.preprocessing.image import ImageDataGenerator",
            "from tensorflow.keras.callbacks import Callback, EarlyStopping",
            "",
            "from sklearn.metrics import (",
            "    confusion_matrix,",
            "    classification_report,",
            "    precision_recall_fscore_support,",
            "    roc_auc_score,",
            "    average_precision_score,",
            ")",
            "",
            "print('TensorFlow version:', tf.__version__)",
            "print('GPU available:', tf.config.list_physical_devices('GPU'))",
            "",
            "CONFIG = {",
            "    'dataset_root': '/kaggle/input/',",
            "    'dataset_slug': 'YOUR_USERNAME/cataract-slit-lamp-sheemazain',",
            "    'img_size': (600, 600),",
            "    'batch_size': 4,",
            "    'num_classes': 2,",
            "    'class_names': ['immature', 'mature'],",
            "    'epochs': 50,",
            "    'base_learning_rate': 1e-4,",
            "    'fine_tune_learning_rate': 1e-5,",
            "    'fine_tune_at_epoch': 10,",
            "    'checkpoint_every': 1,",
            "    'checkpoint_dir': '/kaggle/working/checkpoints',",
            "    'log_dir': '/kaggle/working/logs',",
            "    'output_dir': '/kaggle/working/output',",
            "    'resume_from_checkpoint': None,",
            "}",
            "",
            'CONFIG["train_dir"] = f"{CONFIG[chr(39)]dataset_root{chr(39)]}{CONFIG[chr(39)]dataset_slug{chr(39)]}/train"',
            'CONFIG["val_dir"] = f"{CONFIG[chr(39)]dataset_root{chr(39)]}{CONFIG[chr(39)]dataset_slug{chr(39)]}/val"',
            "",
            "for d in [CONFIG['checkpoint_dir'], CONFIG['log_dir'], CONFIG['output_dir']]:",
            "    os.makedirs(d, exist_ok=True)",
            "",
            "print('Train dir:', CONFIG['train_dir'])",
            "print('Val dir:', CONFIG['val_dir'])",
            "print('Num classes:', CONFIG['num_classes'])",
            "print('Class names:', CONFIG['class_names'])",
        ]
    )
)

# Step 2: Data generators
nb["cells"].append(
    code_cell(
        [
            "# STEP 2: Data Generators",
            "train_datagen = ImageDataGenerator(",
            "    rescale=1.0 / 255.0,",
            "    rotation_range=20,",
            "    width_shift_range=0.1,",
            "    height_shift_range=0.1,",
            "    shear_range=0.1,",
            "    zoom_range=0.1,",
            "    horizontal_flip=True,",
            "    brightness_range=[0.8, 1.2],",
            "    fill_mode='nearest',",
            ")",
            "",
            "val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)",
            "",
            "train_gen = train_datagen.flow_from_directory(",
            "    CONFIG['train_dir'],",
            "    target_size=CONFIG['img_size'],",
            "    batch_size=CONFIG['batch_size'],",
            "    class_mode='categorical',",
            "    seed=42,",
            ")",
            "",
            "val_gen = val_datagen.flow_from_directory(",
            "    CONFIG['val_dir'],",
            "    target_size=CONFIG['img_size'],",
            "    batch_size=CONFIG['batch_size'],",
            "    class_mode='categorical',",
            "    seed=42,",
            ")",
            "",
            "print('Train samples:', train_gen.samples)",
            "print('Val samples:', val_gen.samples)",
            "print('Class indices:', train_gen.class_indices)",
            "",
            "dataset_classes = sorted(train_gen.class_indices.keys(), key=lambda k: train_gen.class_indices[k])",
            "if dataset_classes != CONFIG['class_names']:",
            "    print('[WARN] Dataset class order', dataset_classes, '!= config', CONFIG['class_names'])",
            "    print('Update CONFIG[chr(39)]class_names{chr(39)] to match dataset folder order.')",
        ]
    )
)

# Step 3: Build model
nb["cells"].append(
    code_cell(
        [
            "# STEP 3: Build EfficientNet-B7 Model",
            "",
            "def build_efficientnet_b7(num_classes, img_size, fine_tune=False):",
            "    input_shape = (*img_size, 3)",
            "    base_model = EfficientNetB7(",
            "        weights='imagenet',",
            "        include_top=False,",
            "        input_shape=input_shape,",
            "        pooling='avg',",
            "    )",
            "    ",
            "    if fine_tune:",
            "        base_model.trainable = True",
            "        for layer in base_model.layers:",
            "            layer.trainable = True",
            "    else:",
            "        base_model.trainable = False",
            "    ",
            "    inputs = keras.Input(shape=input_shape)",
            "    x = base_model(inputs, training=fine_tune)",
            "    x = layers.Dropout(0.3)(x)",
            "    outputs = layers.Dense(num_classes, activation='softmax')(x)",
            "    model = Model(inputs, outputs)",
            "    ",
            "    return model, base_model",
            "",
            "model, base_model = build_efficientnet_b7(",
            "    num_classes=CONFIG['num_classes'],",
            "    img_size=CONFIG['img_size'],",
            "    fine_tune=False,",
            ")",
            "",
            "model.compile(",
            "    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['base_learning_rate']),",
            "    loss='categorical_crossentropy',",
            "    metrics=['accuracy'],",
            ")",
            "",
            "model.summary()",
        ]
    )
)

# Step 4: Callbacks
nb["cells"].append(
    code_cell(
        [
            "# STEP 4: Checkpoint Callbacks",
            "",
            "class EpochCheckpoint(Callback):",
            "    def __init__(self, checkpoint_dir, every=1, save_best=False):",
            "        super().__init__()",
            "        self.checkpoint_dir = checkpoint_dir",
            "        self.every = every",
            "        self.save_best = save_best",
            "        self.best_val_acc = 0.0",
            "    ",
            "    def on_epoch_end(self, epoch, logs=None):",
            "        logs = logs or {}",
            "        epoch_num = epoch + 1",
            "        ",
            "        latest_path = os.path.join(self.checkpoint_dir, 'latest.h5')",
            "        self.model.save(latest_path, include_optimizer=True)",
            "        ",
            "        if epoch_num % self.every == 0:",
            "            ckpt_path = os.path.join(",
            "                self.checkpoint_dir, f'epoch_{epoch_num:03d}.h5'",
            "            )",
            "            self.model.save(ckpt_path, include_optimizer=True)",
            "            print('[Checkpoint] Saved', ckpt_path)",
            "        ",
            "        val_acc = logs.get('val_accuracy', 0)",
            "        if self.save_best and val_acc > self.best_val_acc:",
            "            self.best_val_acc = val_acc",
            "            best_path = os.path.join(self.checkpoint_dir, 'best.h5')",
            "            self.model.save(best_path, include_optimizer=True)",
            "            print('[Checkpoint] New best val_accuracy=', val_acc, '->', best_path)",
            "",
            "checkpoint_cb = EpochCheckpoint(",
            "    CONFIG['checkpoint_dir'],",
            "    every=CONFIG['checkpoint_every'],",
            "    save_best=True,",
            ")",
            "",
            "early_stop_cb = EarlyStopping(",
            "    monitor='val_loss', patience=8, restore_best_weights=True, verbose=1",
            ")",
            "",
            "callbacks = [checkpoint_cb, early_stop_cb]",
            "",
            "print('Callbacks ready.')",
        ]
    )
)

# Step 5: Phase 1
nb["cells"].append(
    code_cell(
        [
            "# STEP 5: Phase 1 — Train classification head (backbone frozen)",
            "",
            "print('--- Phase 1: Training classification head (backbone frozen) ---')",
            "",
            "history_1 = model.fit(",
            "    train_gen,",
            "    validation_data=val_gen,",
            "    epochs=CONFIG['fine_tune_at_epoch'],",
            "    callbacks=callbacks,",
            "    verbose=1,",
            ")",
            "",
            "print('Phase 1 complete.')",
        ]
    )
)

# Step 6: Phase 2
nb["cells"].append(
    code_cell(
        [
            "# STEP 6: Phase 2 — Fine-tune full EfficientNet-B7 backbone",
            "",
            "print('--- Phase 2: Fine-tuning EfficientNet-B7 backbone ---')",
            "",
            "model, _ = build_efficientnet_b7(",
            "    num_classes=CONFIG['num_classes'],",
            "    img_size=CONFIG['img_size'],",
            "    fine_tune=True,",
            ")",
            "",
            "latest_ckpt = os.path.join(CONFIG['checkpoint_dir'], 'latest.h5')",
            "if os.path.exists(latest_ckpt):",
            "    model.load_weights(latest_ckpt, by_name=True, skip_mismatch=True)",
            "    print('[Resume] Loaded weights from', latest_ckpt)",
            "",
            "model.compile(",
            "    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['fine_tune_learning_rate']),",
            "    loss='categorical_crossentropy',",
            "    metrics=['accuracy'],",
            ")",
            "",
            "history_2 = model.fit(",
            "    train_gen,",
            "    validation_data=val_gen,",
            "    epochs=CONFIG['epochs'],",
            "    initial_epoch=CONFIG['fine_tune_at_epoch'],",
            "    callbacks=callbacks,",
            "    verbose=1,",
            ")",
            "",
            "print('Phase 2 complete.')",
        ]
    )
)

# Step 7: Save final model
nb["cells"].append(
    code_cell(
        [
            "# STEP 7: Save final Keras model",
            "",
            "final_path = os.path.join(CONFIG['output_dir'], 'efficientnet_b7_final.h5')",
            "model.save(final_path, include_optimizer=False)",
            "print('[Saved] Final Keras model ->', final_path)",
        ]
    )
)

# Step 8: Evaluate
nb["cells"].append(
    code_cell(
        [
            "# STEP 8: Evaluate on slit-lamp hold-out set",
            "",
            "val_gen.reset()",
            "steps = int(np.ceil(val_gen.samples / val_gen.batch_size))",
            "",
            "y_true = []",
            "y_pred = []",
            "y_prob = []",
            "",
            "for _ in range(steps):",
            "    x_batch, y_batch = next(val_gen)",
            "    preds = model.predict(x_batch, verbose=0)",
            "    y_true.extend(np.argmax(y_batch, axis=1))",
            "    y_pred.extend(np.argmax(preds, axis=1))",
            "    y_prob.extend(preds)",
            "",
            "y_true = np.array(y_true)",
            "y_pred = np.array(y_pred)",
            "y_prob = np.array(y_prob)",
            "",
            "cm = confusion_matrix(y_true, y_pred)",
            "print('\\nConfusion Matrix:')",
            "print(cm)",
            "",
            "report = classification_report(",
            "    y_true, y_pred, target_names=CONFIG['class_names'], digits=4",
            ")",
            "print('\\nClassification Report:')",
            "print(report)",
            "",
            "precision, recall, f1, _ = precision_recall_fscore_support(",
            "    y_true, y_pred, average=None",
            ")",
            "for i, name in enumerate(CONFIG['class_names']):",
            "    print(' ', f'{name:12s} P={precision[i]:.4f} R={recall[i]:.4f} F1={f1[i]:.4f}')",
            "",
            "p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(",
            "    y_true, y_pred, average='macro'",
            ")",
            "accuracy = np.mean(y_true == y_pred)",
            "print('\\nAccuracy:', accuracy)",
            "print('Macro   Precision=', p_macro, 'Recall=', r_macro, 'F1=', f1_macro)",
            "",
            "try:",
            "    auc_ovr = roc_auc_score(",
            "        keras.utils.to_categorical(y_true, num_classes=len(CONFIG['class_names'])),",
            "        y_prob,",
            "        multi_class='ovr',",
            "    )",
            "    print('ROC-AUC (OVR):', auc_ovr)",
            "except ValueError as e:",
            "    print('[WARN] ROC-AUC:', e)",
            "    auc_ovr = None",
            "",
            "results = {",
            "    'confusion_matrix': cm.tolist(),",
            "    'class_names': CONFIG['class_names'],",
            "    'accuracy': float(accuracy),",
            "    'precision_per_class': precision.tolist(),",
            "    'recall_per_class': recall.tolist(),",
            "    'f1_per_class': f1.tolist(),",
            "    'precision_macro': float(p_macro),",
            "    'recall_macro': float(r_macro),",
            "    'f1_macro': float(f1_macro),",
            "    'roc_auc_ovr': float(auc_ovr) if auc_ovr is not None else None,",
            "    'evaluated_at': datetime.utcnow().isoformat(),",
            "}",
            "",
            "eval_path = os.path.join(CONFIG['output_dir'], 'evaluation_slit_lamp.json')",
            "with open(eval_path, 'w') as f:",
            "    json.dump(results, f, indent=2)",
            "print('[Saved] Evaluation ->', eval_path)",
        ]
    )
)

# Step 9: Export TFLite
nb["cells"].append(
    code_cell(
        [
            "# STEP 9: Export TFLite model",
            "",
            "tflite_path = os.path.join(CONFIG['output_dir'], 'efficientnet_b7.tflite')",
            "",
            "converter = tf.lite.TFLiteConverter.from_keras_model(model)",
            "tflite_model = converter.convert()",
            "",
            "with open(tflite_path, 'wb') as f:",
            "    f.write(tflite_model)",
            "",
            "size_mb = os.path.getsize(tflite_path) / (1024 * 1024)",
            "print('[Exported] TFLite model ->', tflite_path)",
            "print('[Exported] Size:', size_mb, 'MB')",
            "",
            "versioned = os.path.join(",
            "    CONFIG['output_dir'],",
            "    f'efficientnet_b7_{datetime.utcnow().strftime(chr(37) + chr(89) + chr(109) + chr(100) + chr(95) + chr(72) + chr(77) + chr(83))}.tflite',",
            ")",
            "shutil.copy(tflite_path, versioned)",
            "print('[Saved] Versioned TFLite ->', versioned)",
        ]
    )
)

# Step 10: List files
nb["cells"].append(
    code_cell(
        [
            "# STEP 10: List output files for download",
            "",
            "print('\\n=== Output files ready for download ===')",
            "for f in sorted(os.listdir(CONFIG['output_dir'])):",
            "    fpath = os.path.join(CONFIG['output_dir'], f)",
            "    size_mb = os.path.getsize(fpath) / (1024 * 1024)",
            "    print(' ', f, f'({size_mb:.2f} MB)')",
            "",
            "print('\\n=== Checkpoint files ===')",
            "for f in sorted(os.listdir(CONFIG['checkpoint_dir'])):",
            "    fpath = os.path.join(CONFIG['checkpoint_dir'], f)",
            "    size_mb = os.path.getsize(fpath) / (1024 * 1024)",
            "    print(' ', f, f'({size_mb:.2f} MB)')",
            "",
            "print('\\nDownload from the Kaggle notebook output panel on the right.')",
            "print('Key files:')",
            "print(' -', tflite_path)",
            "print(' -', eval_path)",
            "print(' -', final_path)",
        ]
    )
)

# Markdown next steps
nb["cells"].append(
    md_cell(
        [
            "## After training",
            "",
            "1. Download efficientnet_b7.tflite from the output panel",
            "2. Use it in your Android research app (TensorFlow Lite inference)",
            "3. Download evaluation_slit_lamp.json for your domain-gap report",
            "4. Collect mobile-captured eye photos separately (never used in training)",
            "5. Evaluate the TFLite model on mobile photos using src/evaluate_mobile.py locally",
            "",
            "## Resume after Kaggle timeout",
            "",
            "If your Kaggle session times out, set resume_from_checkpoint to the latest .h5 file and re-run.",
        ]
    )
)

out_path = "D:/Projects/EDD/v2/src/train_efficientnet_b7_kaggle.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook written to", out_path)
