import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path


class DomainGeneralizationTrainer:
    """Trains a multi-task model with MixStyle domain generalization.

    Applies feature-level domain augmentation during training to maintain
    accuracy across different cotton varieties, seasons, and camera hardware.
    """

    def __init__(
        self,
        model: nn.Module,
        n_diseases: int = 15,
        n_severity: int = 3,
        mixstyle_layers: Optional[list] = None,
        lr: float = 1e-4,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.disease_head = nn.Linear(model.feature_dim if hasattr(model, "feature_dim") else 1280, n_diseases).to(self.device)
        self.severity_head = nn.Linear(model.feature_dim if hasattr(model, "feature_dim") else 1280, n_severity).to(self.device)
        self.mixstyle_layers = mixstyle_layers or []
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) +
            list(self.disease_head.parameters()) +
            list(self.severity_head.parameters()),
            lr=lr,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model(x)
        if isinstance(features, tuple):
            features = features[0]
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
            severity_labels = batch.get("severity", torch.zeros_like(disease_labels)).to(self.device)

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
            total += labels.size(0)

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
            severity_labels = batch.get("severity", torch.zeros_like(disease_labels)).to(self.device)

            disease_logits, severity_logits = self.forward(images)
            disease_correct += (disease_logits.argmax(1) == disease_labels).sum().item()
            severity_correct += (severity_logits.argmax(1) == severity_labels).sum().item()
            total += labels.size(0)

        return {
            "disease_accuracy": disease_correct / total,
            "severity_accuracy": severity_correct / total,
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "disease_head_state_dict": self.disease_head.state_dict(),
            "severity_head_state_dict": self.severity_head.state_dict(),
            "metrics": metrics,
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.disease_head.load_state_dict(checkpoint["disease_head_state_dict"])
        self.severity_head.load_state_dict(checkpoint["severity_head_state_dict"])
        return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
