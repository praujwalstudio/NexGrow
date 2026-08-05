import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional


class KnowledgeDistiller:
    """Teacher-Student knowledge distillation for cotton disease diagnosis.

    Distills an ensemble of Swin Transformer + ConvNeXt (teacher) into a
    lightweight student (e.g. MobileNetV3) for edge deployment. Supports
    multi-task students that additionally predict disease severity.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        t_max: int = 50,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.teacher = teacher.to(self.device).eval()
        self.student = student.to(self.device)
        self.temperature = temperature
        self.alpha = alpha
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t_max
        )

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
        kd_loss = kd_loss * (self.temperature ** 2)

        ce_loss = F.cross_entropy(student_logits, labels)
        return self.alpha * kd_loss + (1 - self.alpha) * ce_loss

    def _student_outputs(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        output = self.student(images)
        if isinstance(output, (tuple, list)):
            return output[0], output[1] if len(output) > 1 else None
        return output, None

    def train_epoch(self, dataloader, epoch: int) -> dict:
        self.student.train()
        total_loss = 0.0
        disease_correct = 0
        severity_correct = 0
        total = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["disease"].to(self.device)

            with torch.no_grad():
                teacher_logits = self.teacher(images)

            student_logits, severity_logits = self._student_outputs(images)
            loss = self.distillation_loss(student_logits, teacher_logits, labels)

            if severity_logits is not None and "severity" in batch:
                severity_labels = batch["severity"].to(self.device)
                loss = loss + 0.5 * F.cross_entropy(severity_logits, severity_labels)
                severity_correct += (
                    severity_logits.argmax(1) == severity_labels
                ).sum().item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            disease_correct += (student_logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        self.scheduler.step()

        metrics = {
            "loss": total_loss / len(dataloader),
            "disease_accuracy": disease_correct / total,
        }
        if severity_correct:
            metrics["severity_accuracy"] = severity_correct / total
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        self.student.eval()
        disease_correct = 0
        severity_correct = 0
        total = 0
        total_loss = 0.0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["disease"].to(self.device)

            student_logits, severity_logits = self._student_outputs(images)
            loss = F.cross_entropy(student_logits, labels)
            total_loss += loss.item()
            disease_correct += (student_logits.argmax(dim=1) == labels).sum().item()

            if severity_logits is not None and "severity" in batch:
                severity_labels = batch["severity"].to(self.device)
                loss = loss + 0.5 * F.cross_entropy(severity_logits, severity_labels)
                severity_correct += (
                    severity_logits.argmax(1) == severity_labels
                ).sum().item()

            total += labels.size(0)

        metrics = {
            "loss": total_loss / len(dataloader),
            "disease_accuracy": disease_correct / total,
        }
        if severity_correct:
            metrics["severity_accuracy"] = severity_correct / total
        return metrics

    def save_student(self, path: str, epoch: Optional[int] = None, metrics: Optional[dict] = None):
        if hasattr(self.student, "save_checkpoint"):
            self.student.save_checkpoint(path, epoch=epoch, metrics=metrics)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.student.state_dict(), path)

    def load_student(self, path: str):
        if hasattr(self.student, "load_checkpoint"):
            self.student.load_checkpoint(path)
        else:
            self.student.load_state_dict(torch.load(path, map_location=self.device))
