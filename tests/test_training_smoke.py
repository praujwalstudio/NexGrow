"""Smoke tests proving the training pipeline runs end-to-end.

Uses tiny synthetic on-disk data and random-init models so the suite
runs quickly on CPU without downloading pretrained weights.
"""

import os

from helpers import make_leaf_dataset


class TestTrainingSmoke:
    def test_dataloaders_build_from_disk(self, tmp_path):
        from src.training.dataset import create_dataloaders

        train_dir = make_leaf_dataset(tmp_path / "train")
        val_dir = make_leaf_dataset(tmp_path / "val")

        train_loader, val_loader = create_dataloaders(
            str(train_dir), str(val_dir), batch_size=4, image_size=112, num_workers=0
        )

        batch = next(iter(train_loader))
        assert set(batch.keys()) == {"image", "disease", "severity", "path"}
        assert batch["image"].shape == (4, 3, 112, 112)
        assert batch["disease"].max() < 4
        assert batch["severity"].max() < 3

    def test_distillation_smoke(self, tmp_path):
        from src.models.factory import MultiTaskModel, build_student
        from src.training.dataset import create_dataloaders
        from src.training.distillation import KnowledgeDistiller

        train_dir = make_leaf_dataset(tmp_path / "train", per_class=1)
        val_dir = make_leaf_dataset(tmp_path / "val", per_class=1)
        train_loader, val_loader = create_dataloaders(
            str(train_dir), str(val_dir), batch_size=4, image_size=112, num_workers=0
        )

        teacher = build_student(4, backbone_name="mobilenet_v3_small", pretrained=False)
        student = MultiTaskModel(backbone_name="mobilenet_v3_small", n_diseases=4, pretrained=False)
        distiller = KnowledgeDistiller(teacher, student, lr=1e-3, t_max=1, device="cpu")

        train_metrics = distiller.train_epoch(train_loader, 0)
        val_metrics = distiller.evaluate(val_loader)

        assert "loss" in train_metrics
        assert 0.0 <= val_metrics["disease_accuracy"] <= 1.0

        checkpoint = str(tmp_path / "student.pth")
        distiller.save_student(checkpoint, epoch=1, metrics=val_metrics)
        assert os.path.exists(checkpoint)

    def test_dg_trainer_smoke(self, tmp_path):
        from src.training.dataset import create_dataloaders
        from src.training.domain_generalization import DomainGeneralizationTrainer

        train_dir = make_leaf_dataset(tmp_path / "train", per_class=1)
        val_dir = make_leaf_dataset(tmp_path / "val", per_class=1)
        train_loader, val_loader = create_dataloaders(
            str(train_dir), str(val_dir), batch_size=4, image_size=112, num_workers=0
        )

        trainer = DomainGeneralizationTrainer(
            backbone_name="mobilenet_v3_small",
            n_diseases=4,
            pretrained=False,
            lr=1e-3,
            device="cpu",
        )

        train_metrics = trainer.train_epoch(train_loader, 0)
        val_metrics = trainer.evaluate(val_loader)

        assert "disease_accuracy" in train_metrics
        assert 0.0 <= val_metrics["severity_accuracy"] <= 1.0

        checkpoint = str(tmp_path / "dg.pth")
        trainer.save_checkpoint(checkpoint, epoch=1, metrics=val_metrics)
        assert os.path.exists(checkpoint)

    def test_apply_adasyn_balances_imbalanced(self, tmp_path):
        from src.training.dataset import CottonLeafDataset, apply_adasyn

        train_dir = make_leaf_dataset(tmp_path / "train", diseases=2, severities=1, per_class=3)
        dataset = CottonLeafDataset(str(train_dir))

        original = len(dataset)
        resampled = apply_adasyn(dataset)

        assert len(resampled.samples) >= original

    def test_apply_adasyn_single_class_returns_unchanged(self, tmp_path):
        from src.training.dataset import CottonLeafDataset, apply_adasyn

        train_dir = make_leaf_dataset(tmp_path / "train", diseases=1, severities=2, per_class=3)
        dataset = CottonLeafDataset(str(train_dir))

        resampled = apply_adasyn(dataset)

        assert len(resampled.samples) == len(dataset.samples)
