import cv2
import numpy as np
from typing import Optional

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    A = None
    ToTensorV2 = None


class AugmentationPipeline:
    """AugMix-style augmentation pipeline with environmental simulations.

    Applies deterministic augmentations that simulate real-world field conditions:
    lighting variation, weather effects, and camera artifacts.
    """

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.train_transform = self._build_train_transform()
        self.val_transform = self._build_val_transform()

    def _build_train_transform(self):
        if A is None:
            return None
        return A.Compose([
            A.RandomResizedCrop(
                height=self.image_size, width=self.image_size, scale=(0.8, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.CoarseDropout(
                max_holes=8, max_height=16, max_width=16, p=0.2
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _build_val_transform(self):
        if A is None:
            return None
        return A.Compose([
            A.Resize(height=self.image_size, width=self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def apply_weather_simulation(self, image: np.ndarray, weather: str) -> np.ndarray:
        if weather == "overcast":
            overlay = np.full_like(image, 180, dtype=np.uint8)
            return cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        elif weather == "rainy":
            result = image.copy()
            h, w = result.shape[:2]
            for _ in range(200):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                length = np.random.randint(10, 25)
                cv2.line(result, (x, y), (x - 2, y + length), (200, 200, 200), 1)
            return result
        elif weather == "sunny":
            overlay = np.full_like(image, 40, dtype=np.uint8)
            return cv2.addWeighted(image, 0.85, overlay, 0.15, 0)
        elif weather == "foggy":
            fog = np.random.uniform(0.7, 1.0, image.shape).astype(np.float32)
            fog_img = (image.astype(np.float32) * fog).astype(np.uint8)
            return fog_img
        return image

    def simulate_camera_variations(self, image: np.ndarray) -> dict[str, np.ndarray]:
        results = {}
        h, w = image.shape[:2]
        # Low resolution
        small = cv2.resize(image, (w // 4, h // 4))
        results["low_res"] = cv2.resize(small, (w, h))
        # JPEG compression artifact
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        results["compressed"] = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        # Different white balance (warm shift)
        warm = image.copy().astype(np.float32)
        warm[:, :, 2] = np.clip(warm[:, :, 2] * 1.2, 0, 255)
        results["warm_wb"] = warm.astype(np.uint8)
        return results

    def augment_batch(
        self, images: list[np.ndarray], weather: Optional[str] = None
    ) -> list[np.ndarray]:
        augmented = []
        for img in images:
            if weather:
                img = self.apply_weather_simulation(img, weather)
            if self.train_transform:
                result = self.train_transform(image=img)
                img = result["image"]
            augmented.append(img)
        return augmented
