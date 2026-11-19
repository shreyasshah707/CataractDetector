"""
EDD v2 — Stage 1 Data Preparation
---------------------------------
Combines normal eye images + cataract images (immature/mature) into the
folder structure expected by train_stage1_svm.py:

    output/
        train/
            normal/
            cataract/
        val/
            normal/
            cataract/

Usage:
    python src/prepare_stage1_data.py \
        --normal-dir "D:/Projects/EDD/v2/data/normal/Eye disease _Model/Normal_Eyee" \
        --cataract-dir "D:/Projects/EDD/v2/data/Cataract Classification Dataset in DS" \
        --output "D:/Projects/EDD/v2/data/stage1" \
        --val-split 0.2 \
        --seed 42
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict

CLASS_NORMAL = "normal"
CLASS_CATARACT = "cataract"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(source_dir, class_label):
    """
    Collect all image files from a source directory.
    If class_label == 'cataract', combines immature + mature subfolders.
    If class_label == 'normal', uses all images directly in the folder.
    """
    found = []
    source = Path(source_dir)

    if class_label == CLASS_CATARACT:
        # Look for immature/ and mature/ subfolders
        for sub in ["immature", "mature"]:
            sub_dir = source / sub
            if not sub_dir.exists():
                # Try case-insensitive
                for child in source.iterdir():
                    if child.is_dir() and child.name.lower() == sub.lower():
                        sub_dir = child
                        break
            if sub_dir.exists():
                for ext in VALID_EXTENSIONS:
                    found.extend(sub_dir.glob(f"*{ext}"))
                    found.extend(sub_dir.glob(f"*{ext.upper()}"))
    else:
        # Normal: all images directly in the folder
        for ext in VALID_EXTENSIONS:
            found.extend(source.glob(f"*{ext}"))
            found.extend(source.glob(f"*{ext.upper()}"))

    return sorted(set(found))


def split_train_val(image_lists, val_split=0.2, seed=42):
    """Split each class's images into train/val lists."""
    random.seed(seed)
    train = {}
    val = {}

    for class_name, paths in image_lists.items():
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
            # Handle name collisions
            if dst.exists():
                stem = src.stem
                suffix = src.suffix
                i = 1
                while dst.exists():
                    dst = dest_dir / f"{stem}_{i}{suffix}"
                    i += 1
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Prepare Stage 1 binary dataset (normal vs cataract)")
    parser.add_argument("--normal-dir", required=True, help="Directory containing normal eye images")
    parser.add_argument("--cataract-dir", required=True, help="Directory containing cataract images (immature/ + mature/ subfolders)")
    parser.add_argument("--output", required=True, help="Output directory for prepared dataset")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Collect images
    all_images = {}
    for class_label, source in [(CLASS_NORMAL, args.normal_dir), (CLASS_CATARACT, args.cataract_dir)]:
        print(f"[Scanning] {source} ({class_label})")
        found = collect_images(source, class_label)
        print(f"  {class_label}: {len(found)} images")
        all_images[class_label] = found

    total_normal = len(all_images[CLASS_NORMAL])
    total_cataract = len(all_images[CLASS_CATARACT])
    print(f"\n[Total]")
    print(f"  normal:  {total_normal} images")
    print(f"  cataract: {total_cataract} images")
    print(f"  grand total: {total_normal + total_cataract} images")

    if total_normal == 0 or total_cataract == 0:
        print("[ERROR] Need both normal and cataract images.")
        return

    # Split
    train, val = split_train_val(all_images, val_split=args.val_split, seed=args.seed)

    print(f"\n[Split]")
    print(f"  normal:  train={len(train[CLASS_NORMAL])}, val={len(val[CLASS_NORMAL])}")
    print(f"  cataract: train={len(train[CLASS_CATARACT])}, val={len(val[CLASS_CATARACT])}")

    # Copy
    copy_images(train, args.output, "train")
    copy_images(val, args.output, "val")

    print(f"\n[Done] Prepared Stage 1 dataset at: {args.output}")
    print(f"  train/ and val/ folders with normal/ and cataract/ subfolders")
    print(f"\nNext step:")
    print(f'  python src/train_stage1_svm.py --data-dir "{args.output}" --output-dir "models/stage1"')


if __name__ == "__main__":
    main()
