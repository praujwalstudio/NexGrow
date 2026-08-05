"""Model factory for NexGrow architectures.

Builds the teacher ensemble (Swin Transformer + ConvNeXt), the lightweight
student backbone, feature extractors for domain generalization, and unified
multi-task models with a single checkpoint schema.

Backbones are resolved from torchvision at runtime so the factory always uses
architectures that exist in the installed version.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as tv_models

from src.utils.config import N_DISEASES, N_SEVERITY

_BACKBONES = {
    "mobilenet_v2": (tv_models.mobilenet_v2, "classifier"),
    "mobilenet_v3_large": (tv_models.mobilenet_v3_large, "classifier"),
    "mobilenet_v3_small": (tv_models.mobilenet_v3_small, "classifier"),
    "swin_v2_b": (tv_models.swin_v2_b, "head"),
    "swin_v2_t": (tv_models.swin_v2_t, "head"),
    "convnext_base": (tv_models.convnext_base, "classifier"),
    "convnext_tiny": (tv_models.convnext_tiny, "classifier"),
    "resnet50": (tv_models.resnet50, "fc"),
}


def _build_backbone(name: str, pretrained: bool) -> nn.Module:
    if name not in _BACKBONES:
        available = ", ".join(sorted(_BACKBONES))
        raise ValueError(f"Unsupported backbone '{name}'. Available: {available}")
    builder, _ = _BACKBONES[name]
    weights = "DEFAULT" if pretrained else None
    model = builder(weights=weights)
    model.backbone_name = name
    return model


def _classifier_of(model: nn.Module) -> nn.Module:
    _, attr = _BACKBONES[model.backbone_name]
    return getattr(model, attr)


def _last_linear(module: nn.Module) -> nn.Linear:
    if isinstance(module, nn.Linear):
        return module
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            return child
    raise ValueError("Classifier contains no linear layer")


def _feature_dim_of(module: nn.Module) -> int:
    return _last_linear(module).in_features


def _reshape_last_linear(module: nn.Module, n_out: int) -> None:
    linear = _last_linear(module)
    in_features = linear.in_features
    linear.weight = nn.Parameter(torch.empty(n_out, in_features))
    linear.bias = nn.Parameter(torch.empty(n_out))
    linear.out_features = n_out
    linear.reset_parameters()


def _replace_last_linear_with_identity(model: nn.Module, classifier: nn.Module) -> None:
    """Replace only the final linear layer with a passthrough.

    Preserves pooling / normalization / intermediate head transforms so the
    extractor emits a clean (B, feature_dim) feature vector.
    """
    last_name = ""
    found = False
    for name, module in classifier.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
            found = True
    if not found:
        raise ValueError("Classifier contains no linear layer")
    if last_name == "":
        setattr(model, _BACKBONES[model.backbone_name][1], nn.Identity())
    elif "." in last_name:
        parent_name, child_name = last_name.rsplit(".", 1)
        setattr(classifier.get_submodule(parent_name), child_name, nn.Identity())
    else:
        setattr(classifier, last_name, nn.Identity())


def build_classifier(backbone_name: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    """Backbone with its final classifier resized to `n_classes`."""
    model = _build_backbone(backbone_name, pretrained)
    _reshape_last_linear(_classifier_of(model), n_classes)
    model.num_classes = n_classes
    model.feature_dim = _feature_dim_of(_classifier_of(model))
    return model


def build_student(
    n_classes: int, backbone_name: str = "mobilenet_v3_large", pretrained: bool = True
) -> nn.Module:
    """Lightweight student classifier for knowledge distillation / edge use."""
    return build_classifier(backbone_name, n_classes, pretrained=pretrained)


class TeacherEnsemble(nn.Module):
    """Logits-averaging ensemble of two teacher backbones."""

    def __init__(self, model_a: nn.Module, model_b: nn.Module):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.feature_dim = int(getattr(model_a, "feature_dim", 1024))
        self.num_classes = int(getattr(model_a, "num_classes", 1000))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.model_a(x) + self.model_b(x)) / 2


def build_teacher(
    n_classes: int,
    backbone_names: Sequence[str] = ("swin_v2_b", "convnext_base"),
    pretrained: bool = True,
) -> nn.Module:
    """Teacher ensemble producing averaged logits over the given backbones."""
    if len(backbone_names) < 1:
        raise ValueError("build_teacher requires at least one backbone")
    members = [build_classifier(name, n_classes, pretrained=pretrained) for name in backbone_names]
    if len(members) == 1:
        return members[0]
    return TeacherEnsemble(members[0], members[1])


class FeatureExtractor(nn.Module):
    """Backbone with its classification head removed; outputs feature tensors."""

    def __init__(self, model: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = model
        self.feature_dim = feature_dim
        _replace_last_linear_with_identity(model, _classifier_of(model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_feature_extractor(backbone_name: str, pretrained: bool = True) -> FeatureExtractor:
    """Backbone without classifier, exposing `.feature_dim` for task heads."""
    model = _build_backbone(backbone_name, pretrained)
    feature_dim = _feature_dim_of(_classifier_of(model))
    return FeatureExtractor(model, feature_dim)


class MultiTaskModel(nn.Module):
    """Feature extractor plus disease and severity heads.

    Checkpoints follow the unified NexGrow schema:
      {model_state_dict, disease_head_state_dict, severity_head_state_dict,
       backbone_name, n_diseases, n_severity, [epoch, metrics]}
    """

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_large",
        n_diseases: int = N_DISEASES,
        n_severity: int = N_SEVERITY,
        pretrained: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.n_diseases = n_diseases
        self.n_severity = n_severity
        self.model = build_feature_extractor(backbone_name, pretrained=pretrained)
        self.disease_head = nn.Linear(self.model.feature_dim, n_diseases)
        self.severity_head = nn.Linear(self.model.feature_dim, n_severity)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model(x)
        return self.disease_head(features), self.severity_head(features)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def save_checkpoint(
        self,
        path: str,
        epoch: int | None = None,
        metrics: dict | None = None,
        **extra,
    ) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "disease_head_state_dict": self.disease_head.state_dict(),
            "severity_head_state_dict": self.severity_head.state_dict(),
            "backbone_name": self.backbone_name,
            "n_diseases": self.n_diseases,
            "n_severity": self.n_severity,
        }
        if epoch is not None:
            payload["epoch"] = epoch
        if metrics is not None:
            payload["metrics"] = metrics
        payload.update(extra)
        torch.save(payload, path)
        return path

    def load_checkpoint(self, checkpoint) -> None:
        if isinstance(checkpoint, (str, os.PathLike)):
            checkpoint = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "disease_head_state_dict" in checkpoint:
            self.disease_head.load_state_dict(checkpoint["disease_head_state_dict"], strict=False)
        if "severity_head_state_dict" in checkpoint:
            self.severity_head.load_state_dict(checkpoint["severity_head_state_dict"], strict=False)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, map_location: str = "cpu") -> "MultiTaskModel":
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model = cls(
            backbone_name=checkpoint.get("backbone_name", "mobilenet_v3_large"),
            n_diseases=checkpoint.get("n_diseases", N_DISEASES),
            n_severity=checkpoint.get("n_severity", N_SEVERITY),
        )
        model.load_checkpoint(checkpoint)
        return model
