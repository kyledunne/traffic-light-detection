import os
import random
import shutil
from pathlib import Path

EMPTY_PROPORTION = .15
FILTERED_PATCHES_DATASET_FOLDER = 'data_patches_filtered/'

SRC = Path('data_patches')
DST = Path(FILTERED_PATCHES_DATASET_FOLDER)

# Partition training labels into non-empty and empty
train_labels = SRC / 'train' / 'labels'
non_empty = []
empty = []
for f in sorted(train_labels.glob('*.txt')):
    if f.stat().st_size == 0:
        empty.append(f.stem)
    else:
        non_empty.append(f.stem)

# Compute how many empty patches to keep to reach target proportion
num_empty_to_keep = int(len(non_empty) * EMPTY_PROPORTION / (1 - EMPTY_PROPORTION))
random.seed(42)
sampled_empty = random.sample(empty, min(num_empty_to_keep, len(empty)))

keep_stems = non_empty + sampled_empty

# Create output directories
for sub in ['train/images', 'train/labels', 'val/images', 'val/labels']:
    (DST / sub).mkdir(parents=True, exist_ok=True)

# Copy selected training patches
for stem in keep_stems:
    shutil.copy2(SRC / 'train' / 'images' / f'{stem}.jpg', DST / 'train' / 'images' / f'{stem}.jpg')
    shutil.copy2(SRC / 'train' / 'labels' / f'{stem}.txt', DST / 'train' / 'labels' / f'{stem}.txt')

# Copy entire validation set
for sub in ['val/images', 'val/labels']:
    for f in (SRC / sub).iterdir():
        shutil.copy2(f, DST / sub / f.name)

# Copy data.yaml
shutil.copy2(SRC / 'data.yaml', DST / 'data.yaml')

# Summary
total = len(keep_stems)
kept_empty = len(sampled_empty)
print(f'Non-empty patches: {len(non_empty)}')
print(f'Empty patches kept: {kept_empty} / {len(empty)}')
print(f'Total train patches: {total}')
print(f'Empty proportion: {kept_empty / total:.2%}')