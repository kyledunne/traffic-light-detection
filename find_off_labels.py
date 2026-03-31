"""Find all training images that contain a bounding box for the 'off' traffic light class."""

import os
from pathlib import Path

LABELS_DIR = Path("data/train/labels")
IMAGES_DIR = Path("data/train/images")
OFF_CLASS_ID = "1"

off_images = []

for label_file in sorted(LABELS_DIR.glob("*.txt")):
    with open(label_file) as f:
        for line in f:
            if line.strip().split()[0] == OFF_CLASS_ID:
                # Find matching image file (could be .jpg, .png, etc.)
                stem = label_file.stem
                matches = list(IMAGES_DIR.glob(f"{stem}.*"))
                if matches:
                    off_images.append(matches[0].name)
                else:
                    off_images.append(f"{stem} (no matching image found)")
                break

print(f"Images with 'off' class boxes ({len(off_images)} total):\n")
for name in off_images:
    print(f"  {name}")
