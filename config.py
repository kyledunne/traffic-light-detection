import os
import albumentations as A
from env import Environment


class Config:
    def __init__(self, env: Environment, training, verbose=False):
        self.env: Environment = env
        self.verbose = verbose
        self.training = training
        if self.training:
            os.makedirs(env.training_output_folder, exist_ok=True)

        self.class_labels = ['green', 'yellow', 'red', 'wait_on']

        self.seed = 8675309
        self.batch_size = 24
        self.starting_learning_rate = 0.001
        self.max_epochs = 500
        self.patience = 75
        self.num_workers = 8 if env.device == 'cuda' else 0
        self.pin_memory = self.num_workers > 0
        self.use_amp = env.device == 'cuda'

        self.original_image_width = 1920
        self.original_image_height = 1080

        self.model_name = 'yolo26x-p2.yaml'

        self.train_transforms = [
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.3),
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.PlanckianJitter(mode='blackbody', p=0.15),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.MotionBlur(blur_limit=(3, 5), p=0.1),
            A.GaussNoise(std_range=(0.03, 0.12), p=0.15),
            A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=0.1),
            A.RandomRain(brightness_coefficient=0.8, rain_type='drizzle', p=0.08),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3), shadow_intensity_range=(0.3, 0.5), p=0.1),
            A.ImageCompression(quality_range=(60, 95), p=0.15),
            A.Downscale(scale_range=(0.5, 0.9), p=0.1),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.2), p=0.1),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=0.1),
        ]

        self.hsv_h = 0.005
        self.hsv_s = 0.3
        self.hsv_v = 0.25

    def named_dict(self):
        return {
            'model': self.model_name,
            'epochs': self.max_epochs,
            'patience': self.patience,
            'seed': self.seed,
            'env': self.env.env_name,
            'device': self.env.device,
        }