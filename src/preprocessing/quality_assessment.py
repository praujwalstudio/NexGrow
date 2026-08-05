from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class QualityReport:
    path: str
    is_blurry: bool
    is_overexposed: bool
    is_underexposed: bool
    laplacian_var: float
    mean_brightness: float
    passed: bool


class QualityAssessor:
    """Filters images based on blur, brightness, and clarity metrics.

    Uses Laplacian variance for blur detection and histogram analysis
    for exposure assessment.
    """

    def __init__(
        self,
        blur_threshold: float = 100.0,
        brightness_low: float = 40.0,
        brightness_high: float = 220.0,
    ):
        self.blur_threshold = blur_threshold
        self.brightness_low = brightness_low
        self.brightness_high = brightness_high

    def compute_laplacian_variance(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def compute_mean_brightness(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return float(np.mean(gray))

    def assess(self, image: np.ndarray, path: str = "") -> QualityReport:
        lap_var = self.compute_laplacian_variance(image)
        brightness = self.compute_mean_brightness(image)

        is_blurry = lap_var < self.blur_threshold
        is_overexposed = brightness > self.brightness_high
        is_underexposed = brightness < self.brightness_low
        passed = not (is_blurry or is_overexposed or is_underexposed)

        return QualityReport(
            path=path,
            is_blurry=is_blurry,
            is_overexposed=is_overexposed,
            is_underexposed=is_underexposed,
            laplacian_var=lap_var,
            mean_brightness=brightness,
            passed=passed,
        )

    def assess_file(self, image_path: str) -> QualityReport:
        image = cv2.imread(image_path)
        if image is None:
            return QualityReport(
                path=image_path,
                is_blurry=True,
                is_overexposed=False,
                is_underexposed=False,
                laplacian_var=0.0,
                mean_brightness=0.0,
                passed=False,
            )
        return self.assess(image, path=image_path)

    def filter_directory(
        self, directory: str, extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")
    ) -> list[QualityReport]:
        results = []
        dir_path = Path(directory)
        for ext in extensions:
            for img_path in dir_path.rglob(f"*{ext}"):
                results.append(self.assess_file(str(img_path)))
        return results

    def get_passing_images(self, directory: str) -> list[str]:
        reports = self.filter_directory(directory)
        return [r.path for r in reports if r.passed]
