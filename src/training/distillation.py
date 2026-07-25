import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path


class KnowledgeDistiller:
    """Teacher-Student knowledge distillation for cotton disease diagnosis.

    Distills an ensemble of Swin Transformer + ConvNeXt (teacher) into a
    lightweight MobileNetV4 or ShuffleNetV2 (student) for edge deployment.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
        lr: float = 1e-4,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.teacher = teacher.to(self.device).eval()
        self.student = student.to(self.device)
        self.temperature = temperature
        self.alpha = alpha
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)

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

    def train_epoch(self, dataloader, epoch: int) -> dict:
        self.student.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["disease"].to(self.device)

            with torch.no_grad():
                teacher_logits = self.teacher(images)

            student_logits = self.student(images)
            loss = self.distillation_loss(student_logits, teacher_logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = student_logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        self.scheduler.step()
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total,
        }

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        self.student.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["disease"].to(self.device)

            logits = self.student(images)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        return {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total,
        }

    def save_student(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.student.state_dict(), path)

    def load_student(self, path: str):
        self.student.load_state_dict(torch.load(path, map_location=self.device))
