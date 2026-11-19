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


# ------------------------------------------------------------------
# Cell 0: Title / instructions
# ------------------------------------------------------------------
nb["cells"].append(
    md_cell(
        [
            "# EDD v2 — EfficientNet-B0 Fine-Tuning (Kaggle GPU)",
            "",
            "**Goal:** Fine-tune EfficientNet-B0 on slit-lamp cataract dataset (immature vs mature, 2 classes).",
            "**Phase 1:** Train classification head (backbone frozen).",
            "**Phase 2:** Fine-tune full backbone (GPU-accelerated).",
            "**Checkpoints:** Every epoch — resume if Kaggle session times out.",
            "**Export:** TensorFlow SavedModel + ONNX.",
            "",
            "## Before running",
            "1. **Add your dataset:** Add Data → Dataset → search `cataract-slit-lamp-sheemazain` → Add to notebook",
            "2. **Update `dataset_slug`** below to match your Kaggle dataset path",
            "3. **Set accelerator: GPU** in notebook settings (right sidebar)",
            "4. This trains on **2 classes**: immature vs mature (no normal class)",
        ]
    )
)

# ------------------------------------------------------------------
# Cell 1: Config + imports
# ------------------------------------------------------------------
nb["cells"].append(
    code_cell(
        [
            "# =============================================================================",
            "# STEP 1: Setup & Configuration",
            "# =============================================================================",
            "import os",
            "import json",
            "import shutil",
            "import subprocess",
            "import sys",
            "from datetime import datetime, timezone",
            "from pathlib import Path",
            "",
            "import numpy as np",
            "import tensorflow as tf",
            "from tensorflow import keras",
            "from tensorflow.keras import layers, Model",
            "from tensorflow.keras.applications import EfficientNetB0",
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
            "    # EDIT: Replace with your Kaggle dataset slug (from Add Data sidebar)",
            "    'dataset_slug': 'YOUR_USERNAME/cataract-slit-lamp-sheemazain',",
            "    'dataset_root': '/kaggle/input/',",
            "    'img_size': (224, 224),",
            "    'batch_size': 16,",
            "    'num_classes': 2,",
            "    'class_names': ['immature', 'mature'],",
            "    'epochs': 30,",
            "    'base_learning_rate': 1e-4,",
            "    'fine_tune_learning_rate': 1e-5,",
            "    'fine_tune_at_epoch': 10,",
            "    'checkpoint_every': 1,",
            "    'checkpoint_dir': '/kaggle/working/checkpoints_b0',",
            "    'log_dir': '/kaggle/working/logs_b0',",
            "    'output_dir': '/kaggle/working/output_b0',",
            "    'resume_from_checkpoint': None,  # e.g. '/kaggle/working/checkpoints_b0/latest.h5'",
            "}",
            "",
            'CONFIG["train_dir"] = CONFIG["dataset_root"] + CONFIG["dataset_slug"] + "/train"',
            'CONFIG["val_dir"] = CONFIG["dataset_root"] + CONFIG["dataset_slug"] + "/val"',
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

# ------------------------------------------------------------------
# Cell 2: Data generators
# ------------------------------------------------------------------
nb["cells"].append(
    code_cell(
        [
            "# =============================================================================",
            "# STEP 2: Data Generators",
            "# =============================================================================",
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
        ]
    )
)

# ------------------------------------------------------------------
# Cell 3: Build model
# ------------------------------------------------------------------
nb["cells"].append(
    code_cell(
        [
            "# =============================================================================",
            "# STEP 3: Build EfficientNet-B0 Model",
            "# =============================================================================",
            "",
            "def build_efficientnet_b0(num_classes, img_size, fine_tune=False):",
            "    input_shape = (*img_size, 3)",
            "    base_model = EfficientNetB0(",
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
            "model, base_model = build_efficientnet_b0(",
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

# ------------------------------------------------------------------
# Cell 4: Callbacks
# ------------------------------------------------------------------
nb["cells"].append(
    code_cell(
        [
            "# =============================================================================",
            "# STEP 4: Checkpoint Callbacks",
            "# =============================================================================",
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

# ------------------------------------------------------------------
# Cell 5: Phase 1
# ------------------------------------------------------------------
nb["cells"].append(
    code_cell(
        [
            "# =============================================================================",
            "# STEP 5: Phase 1 — Train classification head (backbone frozen)",
            "# =============================================================================",
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

# ------------------------------------------------------------------
# Cell 6: Phase 2
# ------------------------------------------------------------------
nb["cells"].append(
    code_cell(
        [
            "# =============================================================================",
            "# STEP 6: Phase 2 — Fine-tune full EfficientNet-B0 backbone (GPU)",
            "# =============================================================================",
            "",
            "print('--- Phase 2: Fine-tuning EfficientNet-B0 backbone ---')",
            "",
            "model, _ = build_efficientnet_b0(",
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

# ------------------------------------------------------------------
# Cell 7: Save final model
# ------------------------------------------------------------------
nb["cells"].append(
    code_cell(
        [
            "# =============================================================================",
            "# STEP 7: Save final Keras model",
            "# =============================================================================",
            "",
            "final_path = os.path.join(CONFIG['output_dir'], 'efficientnet_b0_final.h5')",
            "model.save(final_path, include_optimizer=False)",
            "print('[Saved] Final Keras model ->', final_path)",
        ]
    )
)

# ------------------------------------------------------------------
# Cell 8: Evaluate
# ------------------------------------------------------------------
nb["cells"].append(
    code_cell(
        [
            "# =============================================================================",
            "# STEP 8: Evaluate on slit-lamp hold-out set",
            "# =============================================================================",
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
            "try:",
            "    pr_auc = average_precision_score(",
            "        keras.utils.to_categorical(y_true, num_classes=len(CONFIG['class_names'])),",
            "        y_prob,",
            "        average='macro',",
            "    )",
            "    print('PR-AUC (macro):', pr_auc)",
            "except ValueError as e:",
            "    print('[WARN] PR-AUC:', e)",
            "    pr_auc = None",
            "",
            "results = {",
            "    'confusion_matrix': cm.tolist(),",
            "    'class_names': CONFIG['class_names'],",
            "    'accuracy': float(accuracy),",
            "    'precision_per_class': [float(p) for p in precision],",
            "    'recall_per_class': [float(r) for r in recall],",
            "    'f1_per_class': [float(f) for f in f1],",
            "    'precision_macro': float(p_macro),",
            "    'recall_macro': float(r_macro),",
            "    'f1_macro': float(f1_macro),",
            "    'roc_auc_ovr': float(auc_ovr) if auc_ovr is not None else None,",
            "    'pr_auc_macro': float(pr_auc) if pr_auc is not None else None,",
            "    'evaluated_at': datetime.now(timezone.utc).isoformat(),",
            "}",
            "",
            "eval_path = os.path.join(CONFIG['output_dir'], 'evaluation_slit_lamp.json')",
            "with open(eval_path, 'w') as f:",
            "    json.dump(results, f, indent=2)",
            "print('[Saved] Evaluation ->', eval_path)",
        ]
    )
)

# ------------------------------------------------------------------
# Cell 9: Export SavedModel
# ------------------------------------------------------------------
nb["cells"].append(
    code_cell(
        [
            "# =============================================================================",
            "# STEP 9: Export TensorFlow SavedModel",
            "# =============================================================================",
            "",
            "savedmodel_dir = Path(CONFIG['output_dir']) / 'savedmodel'",
            "savedmodel_dir.mkdir(parents=True, exist_ok=True)",
            "",
            "timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')",
            "export_path = savedmodel_dir / f'efficientnet_b0_{timestamp}'",
            "",
            "print('[Exporting] TensorFlow SavedModel ->', export_path)",
            "model.export(str(export_path))",
            "",
            "size_mb = sum(",
            "    f.stat().st_size for f in export_path.rglob('*') if f.is_file()",
            ") / (1024 * 1024)",
            "print('[Saved] SavedModel size:', size_mb, 'MB')",
        ]
    )
)

# ------------------------------------------------------------------
# Cell 10: Export ONNX
# ------------------------------------------------------------------
nb["cells"].append(
    code_cell(
        [
            "# =============================================================================",
            "# STEP 10: Export ONNX (inline, isolated with try/except)",
            "# =============================================================================",
            "",
            "try:",
            "    import tf2onnx",
            "",
            "    onnx_dir = Path(CONFIG['output_dir']) / 'onnx'",
            "    onnx_dir.mkdir(parents=True, exist_ok=True)",
            "",
            "    onnx_timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')",
            "    onnx_path = onnx_dir / f'efficientnet_b0_{onnx_timestamp}.onnx'",
            "",
            "    print('[Exporting] ONNX ->', onnx_path)",
            "",
            "    input_signature = [",
            "        tf.TensorSpec(",
            "            shape=[None, CONFIG['img_size'][0], CONFIG['img_size'][1], 3],",
            "            dtype=tf.float32,",
            "            name='input',",
            "        )",
            "    ]",
            "",
            "    model_proto, _ = tf2onnx.convert.from_keras(",
            "        model,",
            "        input_signature=input_signature,",
            "        opset=13,",
            "    )",
            "",
            "    with open(onnx_path, 'wb') as f:",
            "        f.write(model_proto.SerializeToString())",
            "",
            "    size_mb = onnx_path.stat().st_size / (1024 * 1024)",
            "    print('[Saved] ONNX size:', size_mb, 'MB')",
            "except ImportError:",
            "    print('[SKIP] ONNX export: tf2onnx not installed. Install with: pip install tf2onnx')",
            "except Exception as e:",
            "    print('[ERROR] ONNX export failed:', e)",
        ]
    )
)

# ------------------------------------------------------------------
# Cell 11: List outputs
# ------------------------------------------------------------------
nb["cells"].append(
    code_cell(
        [
            "# =============================================================================",
            "# STEP 11: List output files for download",
            "# =============================================================================",
            "",
            "print('\\n=== Output files ready for download ===')",
            "for f in sorted(os.listdir(CONFIG['output_dir'])):",
            "    fpath = Path(CONFIG['output_dir']) / f",
            "    if fpath.is_dir():",
            "        size_mb = sum(",
            "            g.stat().st_size for g in fpath.rglob('*') if g.is_file()",
            "        ) / (1024 * 1024)",
            "        print(' ', f, '/  (dir, {:.2f} MB)'.format(size_mb))",
            "    else:",
            "        size_mb = fpath.stat().st_size / (1024 * 1024)",
            "        print(' ', f, '({:.2f} MB)'.format(size_mb))",
            "",
            "print('\\n=== Checkpoint files ===')",
            "for f in sorted(os.listdir(CONFIG['checkpoint_dir'])):",
            "    fpath = Path(CONFIG['checkpoint_dir']) / f",
            "    size_mb = fpath.stat().st_size / (1024 * 1024)",
            "    print(' ', f, '({:.2f} MB)'.format(size_mb))",
            "",
            "print('\\nDownload from the Kaggle notebook output panel on the right.')",
            "print('Key files:')",
            "print(' -', final_path)",
            "print(' -', eval_path)",
            "print(' - SavedModel directory:', savedmodel_dir)",
        ]
    )
)

# ------------------------------------------------------------------
# Cell 12: Markdown next steps
# ------------------------------------------------------------------
nb["cells"].append(
    md_cell(
        [
            "## After training",
            "",
            "1. Download `efficientnet_b0_final.h5` from the output panel",
            "2. Download `evaluation_slit_lamp.json` for your domain-gap report",
            "3. Download the SavedModel directory for research use",
            "4. Collect mobile-captured eye photos separately (never used in training)",
            "5. Evaluate on mobile photos locally using `src/evaluate_model.py`",
            "",
            "## Resume after Kaggle timeout",
            "",
            "If your Kaggle session times out, set `resume_from_checkpoint` to the latest `.h5` file and re-run.",
            "",
            "## Expected results",
            "",
            "Based on local CPU testing (Phase 1 only, frozen backbone):",
            "- Accuracy: ~59% on slit-lamp validation set",
            "- Both classes predicted (unlike EfficientNet-B7 which collapsed to one class)",
            "- Fine-tuning on GPU should improve accuracy further",
        ]
    )
)

out_path = "D:/Projects/EDD/v2/src/train_efficientnet_b0_kaggle.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook written to", out_path)
