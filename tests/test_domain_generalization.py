"""Tests for the domain generalization trainer."""

import os
import sys
import tempfile

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.config import N_DISEASES


class _SyntheticDataset(Dataset):
    def __init__(self, n: int = 16, image_size: int = 64):
        self.samples = [
            {
                "image": torch.randn(3, image_size, image_size),
                "disease": torch.tensor(i % N_DISEASES, dtype=torch.long),
                "severity": torch.tensor(i % 3, dtype=torch.long),
            }
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def _make_trainer():
    from src.training.domain_generalization import DomainGeneralizationTrainer

    return DomainGeneralizationTrainer(
        backbone_name="mobilenet_v3_large",
        pretrained=False,
        lr=1e-3,
        device="cpu",
    )


class TestDomainGeneralizationTrainer:
    def test_train_epoch_runs(self):
        trainer = _make_trainer()
        loader = DataLoader(_SyntheticDataset(), batch_size=8)

        metrics = trainer.train_epoch(loader, epoch=0)

        for key in ("disease_loss", "severity_loss", "disease_accuracy", "severity_accuracy"):
            assert key in metrics
        assert metrics["disease_loss"] >= 0
        assert 0.0 <= metrics["disease_accuracy"] <= 1.0

    def test_evaluate_runs(self):
        trainer = _make_trainer()
        loader = DataLoader(_SyntheticDataset(), batch_size=8)

        metrics = trainer.evaluate(loader)

        assert "disease_accuracy" in metrics
        assert "severity_accuracy" in metrics
        assert 0.0 <= metrics["disease_accuracy"] <= 1.0

    def test_checkpoint_round_trip(self):
        trainer = _make_trainer()
        metrics = {"disease_accuracy": 0.5, "severity_accuracy": 0.5}
        tmp = os.path.join(tempfile.gettempdir(), "nexgrow_dg_test.pth")

        trainer.save_checkpoint(tmp, epoch=2, metrics=metrics)
        assert os.path.exists(tmp)

        epoch, loaded_metrics = trainer.load_checkpoint(tmp)
        assert epoch == 2
        assert loaded_metrics["disease_accuracy"] == 0.5

    def test_checkpoint_loadable_by_multitask_model(self):
        from src.models.factory import MultiTaskModel

        trainer = _make_trainer()
        tmp = os.path.join(tempfile.gettempdir(), "nexgrow_dg_test.pth")
        trainer.save_checkpoint(tmp, epoch=1, metrics={})

        model = MultiTaskModel.from_checkpoint(tmp)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        disease_logits, severity_logits = model(x)
        assert disease_logits.shape == (2, N_DISEASES)  # 14 until Bacterial_Light data is collected
        assert severity_logits.shape == (2, 3)
