"""Create oversampled copies of training images containing the 'off' traffic light class.

Produces 10 copies of each image+label pair with new filenames, written to
oversampled_off_class/images/ and oversampled_off_class/labels/ for manual
review before merging into data/train/.
"""

import shutil
from pathlib import Path

LABELS_DIR = Path("data/train/labels")
IMAGES_DIR = Path("data/train/images")
OUTPUT_DIR = Path("oversampled_off_class")
OFF_CLASS_ID = "1"
NUM_COPIES = 10

# Find label files containing the 'off' class
off_label_files = []
for label_file in sorted(LABELS_DIR.glob("*.txt")):
    with open(label_file) as f:
        for line in f:
            if line.strip().split()[0] == OFF_CLASS_ID:
                off_label_files.append(label_file)
                break

print(f"Found {len(off_label_files)} label files with 'off' class")

# Create output directories
out_images = OUTPUT_DIR / "images"
out_labels = OUTPUT_DIR / "labels"
out_images.mkdir(parents=True, exist_ok=True)
out_labels.mkdir(parents=True, exist_ok=True)

total = 0
for label_file in off_label_files:
    stem = label_file.stem
    image_matches = list(IMAGES_DIR.glob(f"{stem}.*"))
    if not image_matches:
        print(f"  WARNING: no image found for {label_file.name}, skipping")
        continue

    image_file = image_matches[0]
    image_ext = image_file.suffix

    for i in range(1, NUM_COPIES + 1):
        new_stem = f"{stem}_oversample_{i}"
        shutil.copy2(image_file, out_images / f"{new_stem}{image_ext}")
        shutil.copy2(label_file, out_labels / f"{new_stem}.txt")
        total += 1

    print(f"  Created {NUM_COPIES} copies of {image_file.name}")

print(f"\nDone. {total} image+label pairs written to {OUTPUT_DIR}/")
print("Review the output, then copy the contents of images/ and labels/ into data/train/.")
