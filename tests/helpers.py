"""Shared helpers for the test suite."""

import cv2
import numpy as np

from src.utils.config import DISEASE_CLASSES, SEVERITY_LEVELS


def make_leaf_dataset(root, diseases=4, severities=3, per_class=2, size=112):
    """Create a minimal on-disk dataset and return its path."""
    rng = np.random.default_rng(0)
    for di in range(diseases):
        for si in range(severities):
            ddir = root / DISEASE_CLASSES[di] / SEVERITY_LEVELS[si]
            ddir.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
                cv2.imwrite(str(ddir / f"leaf_{i}.png"), img)
    return root


def make_leaf_image(size=224):
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 60), (170, 190), (120, 180, 60), -1)
    return image
