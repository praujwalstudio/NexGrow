"""Central configuration for NexGrow training and inference.

Provides the single source of truth for class taxonomies and runtime
settings. The `DISEASE_CLASSES` / `SEVERITY_LEVELS` lists must stay in sync
with the on-disk dataset layout under `data/processed/`.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"

DISEASE_CLASSES = [
    "Bacterial_Blight",
    "CLCuD",
    "Alternaria_Leaf_Spot",
    "Grey_Mildew",
    "Leaf_Curl",
    "Leaf_Spot",
    "Red_Rust",
    "Target_Spot",
    "Bacterial_Light",
    "Healthy",
    "Fusarium_Wilt",
    "Verticillium_Wilt",
    "Anthracnose",
    "Cercospora_Leaf_Spot",
    "Root_Rot",
]

SEVERITY_LEVELS = ["Low", "Moderate", "High"]

N_DISEASES = len(DISEASE_CLASSES)
N_SEVERITY = len(SEVERITY_LEVELS)


@dataclass
class DataConfig:
    train_dir: str = "data/processed/train"
    val_dir: str = "data/processed/val"
    image_size: int = 224


@dataclass
class ModelConfig:
    student_backbone: str = "mobilenet_v3_large"
    teacher_backbones: tuple = ("swin_v2_b", "convnext_base")
    teacher_pretrained: bool = True
    student_pretrained: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 4.0
    alpha: float = 0.7
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"


@dataclass
class PathConfig:
    student_out: str = "models/student_model/student.pth"
    dg_checkpoint: str = "models/student_model/dg_checkpoint.pth"
    teacher_path: str | None = None
    quantized_dir: str = "models/quantized"


@dataclass
class TrainConfig:
    """Aggregate training configuration loaded from `configs/train.yaml`."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def load(cls, path: str | None = None) -> "TrainConfig":
        config_path = Path(path or DEFAULT_CONFIG_PATH)
        if not config_path.exists():
            return cls()
        with open(config_path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(
                student_backbone=raw.get("model", {}).get("student_backbone", "mobilenet_v3_large"),
                teacher_backbones=tuple(
                    raw.get("model", {}).get("teacher_backbones", ("swin_v2_b", "convnext_base"))
                ),
                teacher_pretrained=raw.get("model", {}).get("teacher_pretrained", True),
                student_pretrained=raw.get("model", {}).get("student_pretrained", True),
            ),
            training=TrainingConfig(**raw.get("training", {})),
            paths=PathConfig(**raw.get("paths", {})),
        )


def resolve_device(requested: str = "auto") -> str:
    """Resolve a device string to 'cuda' or 'cpu' based on availability."""
    if requested == "auto":
        return "cuda" if _cuda_available() else "cpu"
    if requested in ("cuda", "cpu"):
        return requested
    raise ValueError(f"Invalid device: {requested}")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def getenv_float(key: str, default: float) -> float:
    value = os.getenv(key)
    return float(value) if value is not None else default
