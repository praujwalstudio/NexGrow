"""Reusable end-to-end diagnostic pipeline.

Wires quality assessment -> leaf detection -> SAM isolation -> multi-task
classification so CLI, API, and tests all share the same inference path.
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from src.models.factory import MultiTaskModel
from src.preprocessing.quality_assessment import QualityAssessor
from src.preprocessing.sam_isolation import SAMIsolator
from src.utils.config import DISEASE_CLASSES, SEVERITY_LEVELS, resolve_device


@dataclass
class LeafDiagnosis:
    bbox: tuple
    det_conf: float
    disease: str
    disease_idx: int
    severity: str
    severity_idx: int
    confidence: float
    isolated_image: np.ndarray


@dataclass
class DiagnosisResult:
    passed: bool
    quality: object
    leaves: list
    raw_image: np.ndarray


class DiseaseHeadModel(nn.Module):
    """Wraps a MultiTaskModel to output only disease logits (for Grad-CAM)."""

    def __init__(self, multitask_model: MultiTaskModel):
        super().__init__()
        self.model = multitask_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.features(x)
        return self.model.disease_head(features)


def find_target_layer(backbone: nn.Module) -> nn.Module | None:
    """Best-guess last convolutional layer for Grad-CAM heatmaps."""
    if hasattr(backbone, "features"):
        return backbone.features[-1]
    for module in reversed(list(backbone.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    return None


class DiagnosticPipeline:
    """Full pipeline: quality -> detection -> isolation -> classification."""

    def __init__(
        self,
        model_path: str,
        detector_path: str | None = None,
        device: str = "auto",
        image_size: int = 224,
        conf_threshold: float = 0.25,
    ):
        self.device = torch.device(resolve_device(device))
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.assessor = QualityAssessor()
        self.isolator = SAMIsolator()
        self.model = MultiTaskModel.from_checkpoint(model_path)
        self.model.eval().to(self.device)
        self.detector = self._load_detector(detector_path)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(image_size + 32),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _load_detector(path: str | None):
        try:
            from ultralytics import YOLO

            if path and os.path.exists(path):
                return YOLO(path)
        except Exception:
            return None
        return None

    def _detect(self, image: np.ndarray) -> list:
        if self.detector is not None:
            try:
                results = self.detector(image, conf=self.conf_threshold, verbose=False)[0]
                boxes = []
                if results.boxes is not None:
                    for box in results.boxes:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        boxes.append(((x1, y1, x2, y2), float(box.conf[0])))
                if boxes:
                    return boxes
            except Exception:
                pass
        h, w = image.shape[:2]
        return [((0, 0, w, h), 1.0)]

    @torch.no_grad()
    def classify(self, image: np.ndarray) -> tuple:
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        disease_logits, severity_logits = self.model(input_tensor)
        disease_probs = torch.softmax(disease_logits, dim=1)[0]
        severity_probs = torch.softmax(severity_logits, dim=1)[0]
        disease_idx = int(disease_probs.argmax())
        severity_idx = int(severity_probs.argmax())
        confidence = float(disease_probs.max())
        return disease_idx, severity_idx, confidence, disease_logits

    def run(self, image: np.ndarray) -> DiagnosisResult:
        quality = self.assessor.assess(image, path="input")
        leaves = []
        for (x1, y1, x2, y2), det_conf in self._detect(image):
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                isolated = self.isolator.isolate(crop)
            except Exception:
                isolated = crop
            disease_idx, severity_idx, confidence, _ = self.classify(isolated)
            leaves.append(
                LeafDiagnosis(
                    bbox=(x1, y1, x2, y2),
                    det_conf=det_conf,
                    disease=DISEASE_CLASSES[disease_idx]
                    if disease_idx < len(DISEASE_CLASSES)
                    else "Unknown",
                    disease_idx=disease_idx,
                    severity=SEVERITY_LEVELS[severity_idx]
                    if severity_idx < len(SEVERITY_LEVELS)
                    else "Unknown",
                    severity_idx=severity_idx,
                    confidence=confidence,
                    isolated_image=isolated,
                )
            )
        return DiagnosisResult(
            passed=quality.passed,
            quality=quality,
            leaves=leaves,
            raw_image=image,
        )

    def run_file(self, image_path: str) -> DiagnosisResult:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        result = self.run(image)
        result.quality.path = image_path
        return result
