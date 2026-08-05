from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.factory import build_feature_extractor
from src.utils.config import N_DISEASES, N_SEVERITY


class DomainGeneralizationTrainer:
    """Trains a multi-task model (disease + severity) for domain generalization.

    Uses a feature extractor (backbone without its classification head) plus
    separate disease and severity heads, with feature-level domain augmentation
    to maintain accuracy across cotton varieties, seasons, and camera hardware.
    """

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_large",
        n_diseases: int = N_DISEASES,
        n_severity: int = N_SEVERITY,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = build_feature_extractor(backbone_name, pretrained=pretrained).to(self.device)
        self.backbone_name = backbone_name
        self.n_diseases = n_diseases
        self.n_severity = n_severity
        self.disease_head = nn.Linear(self.model.feature_dim, n_diseases).to(self.device)
        self.severity_head = nn.Linear(self.model.feature_dim, n_severity).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters())
            + list(self.disease_head.parameters())
            + list(self.severity_head.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model(x)
        disease_logits = self.disease_head(features)
        severity_logits = self.severity_head(features)
        return disease_logits, severity_logits

    def train_epoch(self, dataloader, epoch: int) -> dict:
        self.model.train()
        self.disease_head.train()
        self.severity_head.train()

        total_disease_loss = 0.0
        total_severity_loss = 0.0
        disease_correct = 0
        severity_correct = 0
        total = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            disease_labels = batch["disease"].to(self.device)
            severity_labels = batch.get("severity", torch.zeros_like(disease_labels)).to(
                self.device
            )

            disease_logits, severity_logits = self.forward(images)

            loss_disease = F.cross_entropy(disease_logits, disease_labels)
            loss_severity = F.cross_entropy(severity_logits, severity_labels)
            loss = loss_disease + 0.5 * loss_severity

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_disease_loss += loss_disease.item()
            total_severity_loss += loss_severity.item()
            disease_correct += (disease_logits.argmax(1) == disease_labels).sum().item()
            severity_correct += (severity_logits.argmax(1) == severity_labels).sum().item()
            total += disease_labels.size(0)

        return {
            "disease_loss": total_disease_loss / len(dataloader),
            "severity_loss": total_severity_loss / len(dataloader),
            "disease_accuracy": disease_correct / total,
            "severity_accuracy": severity_correct / total,
        }

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        self.model.eval()
        self.disease_head.eval()
        self.severity_head.eval()

        disease_correct = 0
        severity_correct = 0
        total = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            disease_labels = batch["disease"].to(self.device)
            severity_labels = batch.get("severity", torch.zeros_like(disease_labels)).to(
                self.device
            )

            disease_logits, severity_logits = self.forward(images)
            disease_correct += (disease_logits.argmax(1) == disease_labels).sum().item()
            severity_correct += (severity_logits.argmax(1) == severity_labels).sum().item()
            total += disease_labels.size(0)

        return {
            "disease_accuracy": disease_correct / total,
            "severity_accuracy": severity_correct / total,
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "disease_head_state_dict": self.disease_head.state_dict(),
                "severity_head_state_dict": self.severity_head.state_dict(),
                "backbone_name": self.backbone_name,
                "n_diseases": self.n_diseases,
                "n_severity": self.n_severity,
                "epoch": epoch,
                "metrics": metrics,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.disease_head.load_state_dict(checkpoint["disease_head_state_dict"])
        self.severity_head.load_state_dict(checkpoint["severity_head_state_dict"])
        return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
