"""Find all images that contain a bounding box for the 'off' traffic light class."""

from pathlib import Path

OFF_CLASS_ID = '1'


def _find_off_in_split(labels_dir, images_dir, class_id=OFF_CLASS_ID):
    """Return (image_paths, label_paths) for all images with the given class in one split."""
    image_paths = []
    label_paths = []
    for label_file in sorted(Path(labels_dir).glob('*.txt')):
        with open(label_file) as f:
            for line in f:
                if line.strip().split()[0] == class_id:
                    stem = label_file.stem
                    matches = list(Path(images_dir).glob(f'{stem}.*'))
                    if matches:
                        image_paths.append(str(matches[0]))
                        label_paths.append(str(label_file))
                    break
    return image_paths, label_paths


def find_all_off_images(data_dir):
    """Search train and val splits, returning two parallel lists: (image_paths, label_paths)."""
    all_image_paths = []
    all_label_paths = []
    for split in ('train', 'val'):
        images, labels = _find_off_in_split(
            f'{data_dir}{split}/labels',
            f'{data_dir}{split}/images',
        )
        all_image_paths.extend(images)
        all_label_paths.extend(labels)
    return all_image_paths, all_label_paths


if __name__ == '__main__':
    image_paths, label_paths = find_all_off_images('data/')
    print(f"Images with 'off' class boxes ({len(image_paths)} total):\n")
    for img in image_paths:
        print(f'  {img}')
