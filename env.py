from dataclasses import dataclass
import numpy as np
from pathlib import Path
import os

@dataclass
class Environment:
    env_name: str
    patches_yolo_config_yaml: str
    fixed_yolo_config_yaml: str
    patches_train_images_folder: str
    patches_train_labels_folder: str
    patches_val_images_folder: str
    patches_val_labels_folder: str
    fixed_val_images_folder: str
    fixed_val_labels_folder: str
    original_dataset_root_folder: str
    training_output_folder: str
    saved_weights_filepath: str
    video_input_filepath: str
    video_output_filepath: str
    device: str

    def fetch_original_val_ids(self):
        return np.array([Path(f).stem for f in os.listdir(self.fixed_val_images_folder)])


local_env = Environment(
    env_name='local',
    patches_yolo_config_yaml='data_patches_filtered/data.yaml',
    fixed_yolo_config_yaml='data_fixed/data.yaml',
    patches_train_images_folder='data_patches_filtered/train/images/',
    patches_train_labels_folder='data_patches_filtered/train/labels/',
    patches_val_images_folder='data_patches_filtered/val/images/',
    patches_val_labels_folder='data_patches_filtered/val/labels/',
    fixed_val_images_folder='data_fixed/val/images/',
    fixed_val_labels_folder='data_fixed/val/labels/',
    original_dataset_root_folder='data/',
    training_output_folder='data_gen/',
    saved_weights_filepath='data_gen/best.pt',
    video_input_filepath='data/inference_traffic_light_video.mp4',
    video_output_filepath='data_gen/traffic_light_detection_output_video.mp4',
    device='cpu',
)
kaggle_env = Environment(
    env_name='kaggle',
    patches_yolo_config_yaml='/kaggle/input/datasets/kyledunne/traffic-lights-patches-dataset/data.yaml',
    fixed_yolo_config_yaml='/kaggle/input/datasets/kyledunne/traffic-lights-dataset/data.yaml',
    patches_train_images_folder='/kaggle/input/datasets/kyledunne/traffic-lights-patches-dataset/train/images/',
    patches_train_labels_folder='/kaggle/input/datasets/kyledunne/traffic-lights-patches-dataset/train/labels/',
    patches_val_images_folder='/kaggle/input/datasets/kyledunne/traffic-lights-patches-dataset/val/images/',
    patches_val_labels_folder='/kaggle/input/datasets/kyledunne/traffic-lights-patches-dataset/val/labels/',
    fixed_val_images_folder='/kaggle/input/datasets/kyledunne/traffic-lights-dataset/val/images/',
    fixed_val_labels_folder='/kaggle/input/datasets/kyledunne/traffic-lights-dataset/val/labels/',
    original_dataset_root_folder='N/A',
    training_output_folder='/kaggle/working/',
    saved_weights_filepath='/kaggle/input/datasets/kyledunne/traffic-lights-yolo-best-weights/best.pt',
    video_input_filepath='/kaggle/input/datasets/kyledunne/traffic-lights-dataset/inference_traffic_light_video.mp4',
    video_output_filepath='/kaggle/working/traffic_light_detection_output_video.mp4',
    device='cuda',
)
colab_home_folder = '/content/drive/My Drive/Colab Notebooks/traffic-light-detection/'
colab_env = Environment(
    env_name='colab',
    patches_yolo_config_yaml=colab_home_folder + 'data_patches_filtered/data.yaml',
    fixed_yolo_config_yaml=colab_home_folder + 'data_fixed/data.yaml',
    patches_train_images_folder=colab_home_folder + 'data_patches_filtered/train/images/',
    patches_train_labels_folder=colab_home_folder + 'data_patches_filtered/train/labels/',
    patches_val_images_folder=colab_home_folder + 'data_patches_filtered/val/images/',
    patches_val_labels_folder=colab_home_folder + 'data_patches_filtered/val/labels/',
    fixed_val_images_folder=colab_home_folder + 'data_fixed/val/images/',
    fixed_val_labels_folder=colab_home_folder + 'data_fixed/val/labels/',
    original_dataset_root_folder=colab_home_folder + 'data/',
    training_output_folder=colab_home_folder + 'data_gen/',
    saved_weights_filepath=colab_home_folder + 'data_gen/best.pt',
    video_input_filepath=colab_home_folder + 'data/inference_traffic_light_video.mp4',
    video_output_filepath=colab_home_folder + 'data_gen/traffic_light_detection_output_video.mp4',
    device='cuda',
)