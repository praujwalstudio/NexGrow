"""Tests for NexGrow preprocessing pipeline."""

import numpy as np


class TestQualityAssessor:
    def test_blur_detection(self):
        from src.preprocessing.quality_assessment import QualityAssessor

        assessor = QualityAssessor(blur_threshold=100.0)
        sharp = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        report = assessor.assess(sharp, path="test.jpg")
        assert report.laplacian_var >= 0
        assert isinstance(report.passed, bool)

    def test_brightness_detection(self):
        from src.preprocessing.quality_assessment import QualityAssessor

        assessor = QualityAssessor()
        dark = np.zeros((224, 224, 3), dtype=np.uint8)
        report = assessor.assess(dark, path="dark.jpg")
        assert report.is_underexposed is True
        assert report.passed is False


class TestAugmentation:
    def test_pipeline_creates_output(self):
        from src.preprocessing.augmentation import AugmentationPipeline

        pipeline = AugmentationPipeline(image_size=224)
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = pipeline.apply_weather_simulation(image, "rainy")
        assert result.shape == image.shape
