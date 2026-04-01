"""
Copy dataset from data/ to data_fixed/, removing the 'off' class and remapping yellow.

Original classes: 0=green, 1=off, 2=red, 3=wait_on, 4=yellow
New classes:      0=green, 1=yellow, 2=red, 3=wait_on
"""

import shutil
from pathlib import Path

SRC = Path("data")
DST = Path("data_fixed")

CLASS_REMAP = {
    0: 0,   # green -> green
    # 1 (off) -> removed
    2: 2,   # red -> red
    3: 3,   # wait_on -> wait_on
    4: 1,   # yellow -> yellow (new position)
}

for split in ("train", "val"):
    src_images = SRC / split / "images"
    src_labels = SRC / split / "labels"
    dst_images = DST / split / "images"
    dst_labels = DST / split / "labels"

    # Copy images as-is
    if dst_images.exists():
        shutil.rmtree(dst_images)
    shutil.copytree(src_images, dst_images)
    print(f"[{split}] Copied {len(list(dst_images.iterdir()))} images")

    # Process labels
    dst_labels.mkdir(parents=True, exist_ok=True)
    removed = 0
    remapped = 0
    file_count = 0

    for label_file in sorted(src_labels.glob("*.txt")):
        new_lines = []
        for line in label_file.read_text().strip().splitlines():
            parts = line.split()
            cls_id = int(parts[0])
            if cls_id == 1:
                removed += 1
                continue
            new_cls = CLASS_REMAP[cls_id]
            if new_cls != cls_id:
                remapped += 1
            parts[0] = str(new_cls)
            new_lines.append(" ".join(parts))

        (dst_labels / label_file.name).write_text("\n".join(new_lines) + "\n" if new_lines else "")
        file_count += 1

    print(f"[{split}] Processed {file_count} label files")
    print(f"[{split}] Removed {removed} 'off' boxes, remapped {remapped} yellow boxes (4->1)")

print("\nDone. data_fixed/ is ready.")
