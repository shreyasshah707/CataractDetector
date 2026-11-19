"""
EDD v2 — Dataset Preparation Utility
------------------------------------
Helps combine multiple slit-lamp cataract datasets into the folder structure
expected by train_efficientnet_b7.py:

    output/
        train/
            immature/
            mature/
        val/
            immature/
            mature/

Usage:
    python src/dataset_prepare.py \
        --source "/path/to/kaggle/sheemazain" \
        --source "/path/to/mendeley/6wv33nbcvv" \
        --output "./prepared_dataset" \
        --val-split 0.2

Each source directory should contain subfolders named after classes
(e.g., immature/, mature/). Images are copied (not moved) so originals are preserved.
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict

CLASS_NAMES = ["immature", "mature"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(source_dir):
    """
    Walk a source directory and return {class_name: [image_paths]}.
    Looks for immediate subfolders matching CLASS_NAMES.
    """
    found = defaultdict(list)
    source = Path(source_dir)

    for class_name in CLASS_NAMES:
        class_dir = source / class_name
        if not class_dir.exists():
            # Try case-insensitive match
            for child in source.iterdir():
                if child.is_dir() and child.name.lower() == class_name.lower():
                    class_dir = child
                    break

        if class_dir.exists():
            for ext in VALID_EXTENSIONS:
                found[class_name].extend(class_dir.glob(f"*{ext}"))
                found[class_name].extend(class_dir.glob(f"*{ext.upper()}"))

    return found


def split_train_val(image_lists, val_split=0.2, seed=42):
    """Split each class's images into train/val lists."""
    random.seed(seed)
    train = defaultdict(list)
    val = defaultdict(list)

    for class_name, paths in image_lists.items():
        paths = sorted(set(paths))  # dedupe
        random.shuffle(paths)
        n_val = max(1, int(len(paths) * val_split))
        val[class_name] = paths[:n_val]
        train[class_name] = paths[n_val:]

    return train, val


def copy_images(image_lists, output_root, split):
    """Copy images into output_root/split/class_name/."""
    for class_name, paths in image_lists.items():
        dest_dir = Path(output_root) / split / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            dst = dest_dir / src.name
            # Handle name collisions by adding suffix
            if dst.exists():
                stem = src.stem
                suffix = src.suffix
                i = 1
                while dst.exists():
                    dst = dest_dir / f"{stem}_{i}{suffix}"
                    i += 1
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Prepare slit-lamp cataract dataset for training")
    parser.add_argument("--source", action="append", required=True,
                        help="Source dataset directory (can specify multiple times)")
    parser.add_argument("--output", required=True, help="Output directory for prepared dataset")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Collect images from all sources
    all_images = defaultdict(list)
    for source in args.source:
        print(f"[Scanning] {source}")
        found = collect_images(source)
        for class_name in CLASS_NAMES:
            count = len(found.get(class_name, []))
            print(f"  {class_name}: {count} images")
            all_images[class_name].extend(found.get(class_name, []))

    # Dedupe across sources (by path)
    for class_name in CLASS_NAMES:
        all_images[class_name] = list(set(all_images[class_name]))

    print("\n[Total]")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {len(all_images[class_name])} images")

    total = sum(len(v) for v in all_images.values())
    if total == 0:
        print("[ERROR] No images found. Check source folder structure.")
        return

    # Split
    train, val = split_train_val(all_images, val_split=args.val_split, seed=args.seed)

    print("\n[Split]")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: train={len(train[class_name])}, val={len(val[class_name])}")

    # Copy
    copy_images(train, args.output, "train")
    copy_images(val, args.output, "val")

    print(f"\n[Done] Prepared dataset at: {args.output}")
    print(f"  train/ and val/ folders with immature/ and mature/ subfolders")
    print(f"  Update train_efficientnet_b7.py CONFIG:")
    print(f'    "train_dir": "{args.output}/train"')
    print(f'    "val_dir": "{args.output}/val"')


if __name__ == "__main__":
    main()
