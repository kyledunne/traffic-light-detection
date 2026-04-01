"""Collect all images containing 'off' class lights from train and val into off_images/."""

import shutil
from pathlib import Path

OFF_CLASS_ID = "1"
OUTPUT_DIR = Path("off_images")

splits = {
    "train": (Path("data/train/images"), Path("data/train/labels")),
    "val": (Path("data/val/images"), Path("data/val/labels")),
}

OUTPUT_DIR.mkdir(exist_ok=True)

total = 0
for split_name, (images_dir, labels_dir) in splits.items():
    for label_file in sorted(labels_dir.glob("*.txt")):
        with open(label_file) as f:
            if not any(line.strip().split()[0] == OFF_CLASS_ID for line in f):
                continue

        stem = label_file.stem
        image_matches = list(images_dir.glob(f"{stem}.*"))
        if not image_matches:
            print(f"  WARNING: no image found for {label_file.name}, skipping")
            continue

        image_file = image_matches[0]
        new_stem = f"{split_name}_{stem}"

        shutil.copy2(image_file, OUTPUT_DIR / f"{new_stem}{image_file.suffix}")
        shutil.copy2(label_file, OUTPUT_DIR / f"{new_stem}.txt")
        total += 1
        print(f"  [{split_name}] {image_file.name}")

print(f"\nDone. {total} image+label pairs written to {OUTPUT_DIR}/")
