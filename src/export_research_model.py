"""
EDD v2 — Research Model Export
------------------------------
Exports a trained EfficientNet-B7 cataract severity model to research-friendly formats:

1. TensorFlow SavedModel (primary) — native TF format, preserves full graph,
   supports resuming training, easy inspection in Python.
2. ONNX (secondary) — cross-framework interoperability for research comparison.

Usage:
    python src/export_research_model.py \
        --model "D:/Projects/EDD/v2/models/efficientnet_b7_final.h5" \
        --output-dir "D:/Projects/EDD/v2/models/exported" \
        --img-size 600 \
        --num-classes 2 \
        --class-names immature mature

If you have a Keras checkpoint (.h5) from Kaggle training, point --model at it.
"""

import os
import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from tensorflow import keras

HAS_TF2ONNX = True  # We install it; if missing, subprocess will fail gracefully


def load_model(model_path, num_classes, img_size):
    """Load a Keras model from .h5 or SavedModel directory."""
    model_path = Path(model_path)

    if model_path.is_dir() and (model_path / "saved_model.pb").exists():
        print(f"[Loading] SavedModel from {model_path}")
        return keras.models.load_model(str(model_path))

    if model_path.suffix.lower() in (".h5", ".keras"):
        print(f"[Loading] Keras model from {model_path}")
        return keras.models.load_model(str(model_path))

    raise ValueError(f"Unsupported model path: {model_path}")


def export_savedmodel(model, output_dir):
    """Export model to TensorFlow SavedModel format."""
    savedmodel_dir = Path(output_dir) / "savedmodel"
    savedmodel_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_path = savedmodel_dir / f"efficientnet_b7_{timestamp}"

    print(f"[Exporting] TensorFlow SavedModel -> {export_path}")
    model.export(str(export_path))  # Keras 3 / TF 2.x export API

    size_mb = sum(
        f.stat().st_size for f in export_path.rglob("*") if f.is_file()
    ) / (1024 * 1024)
    print(f"[Saved] SavedModel size: {size_mb:.2f} MB")
    return export_path


def export_onnx_via_subprocess(model_h5_path, output_dir, img_size):
    """
    Run ONNX export in a separate Python subprocess so a tf2onnx crash
    cannot take down the main export script.
    """
    onnx_dir = Path(output_dir) / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    onnx_path = onnx_dir / f"efficientnet_b7_{timestamp}.onnx"

    script = f"""
import sys
sys.path.insert(0, r'{Path(__file__).parent.resolve()}')
import tensorflow as tf
from tensorflow import keras
import tf2onnx

model = keras.models.load_model(r'{model_h5_path}')

input_signature = [
    tf.TensorSpec(
        shape=[None, {img_size}, {img_size}, 3],
        dtype=tf.float32,
        name='input',
    )
]

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
)
with open(r'{onnx_path}', 'wb') as f:
    f.write(model_proto.SerializeToString())

size_mb = Path(r'{onnx_path}').stat().st_size / (1024 * 1024)
print(f'[Saved] ONNX size: {{size_mb:.2f}} MB')
"""

    script_path = Path(output_dir) / "_onnx_export_tmp.py"
    script_path.write_text(script, encoding="utf-8")

    print(f"[Exporting] ONNX (subprocess) -> {onnx_path}")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        # Print subprocess output for debugging
        for line in result.stdout.splitlines():
            print(f"  [onnx] {line}")
        for line in result.stderr.splitlines():
            if "WARNING" in line or "Error" in line or "error" in line:
                print(f"  [onnx] {line}")

        if result.returncode != 0:
            print(f"[ERROR] ONNX subprocess failed (exit code {result.returncode})")
            return None

        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / (1024 * 1024)
            print(f"[Saved] ONNX size: {size_mb:.2f} MB")
            return onnx_path
        else:
            print("[ERROR] ONNX file not created")
            return None

    except subprocess.TimeoutExpired:
        print("[ERROR] ONNX subprocess timed out (>5 min)")
        return None
    except Exception as e:
        print(f"[ERROR] ONNX subprocess error: {e}")
        return None
    finally:
        # Clean up temp script
        try:
            script_path.unlink()
        except Exception:
            pass


def save_export_manifest(output_dir, savedmodel_path, onnx_path, class_names, img_size, num_classes):
    """Save a manifest JSON describing the exported model."""
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "model_architecture": "EfficientNet-B7",
        "task": "cataract_severity_grading",
        "classes": class_names,
        "num_classes": num_classes,
        "input_size": [img_size, img_size, 3],
        "input_preprocessing": "rescale 1/255, resize to 600x600",
        "formats": {
            "savedmodel": str(savedmodel_path) if savedmodel_path else None,
            "onnx": str(onnx_path) if onnx_path else None,
        },
        "notes": [
            "Research-only export. Not for mobile deployment.",
            "Trained on slit-lamp dataset (immature vs mature, 2 classes).",
            "No normal class — model cannot detect healthy eyes.",
        ],
    }

    manifest_path = Path(output_dir) / "export_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Saved] Export manifest -> {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Export trained model to research formats")
    parser.add_argument("--model", required=True, help="Path to trained .h5/.keras model or SavedModel dir")
    parser.add_argument("--output-dir", required=True, help="Directory for exported models")
    parser.add_argument("--img-size", type=int, default=600, help="Input image size (default 600)")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes (default 2)")
    parser.add_argument(
        "--class-names", nargs="+", default=["immature", "mature"],
        help="Class names in order (default: immature mature)"
    )
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model, args.num_classes, args.img_size)
    model.summary()

    # Export SavedModel (primary)
    savedmodel_path = export_savedmodel(model, str(output_dir))

    # Export ONNX (secondary) — runs in subprocess to isolate crashes
    onnx_path = None
    if not args.no_onnx:
        onnx_path = export_onnx_via_subprocess(args.model, str(output_dir), args.img_size)

    # Save manifest (always, even if ONNX failed)
    save_export_manifest(
        str(output_dir),
        savedmodel_path,
        onnx_path,
        args.class_names,
        args.img_size,
        args.num_classes,
    )

    print("\n=== Export Complete ===")
    print(f"Output directory: {output_dir}")
    if savedmodel_path:
        print(f"SavedModel: {savedmodel_path}")
    if onnx_path:
        print(f"ONNX: {onnx_path}")
    else:
        print("ONNX: skipped or failed")


if __name__ == "__main__":
    main()
