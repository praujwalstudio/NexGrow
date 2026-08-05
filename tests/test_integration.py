"""End-to-end integration test: synthetic data -> train -> diagnose -> API."""

import cv2
import numpy as np

from tests.test_training_smoke import _make_leaf_dataset


def _train_tiny_dg_model(data_root, checkpoint, epochs=1):
    from src.training.dataset import create_dataloaders
    from src.training.domain_generalization import DomainGeneralizationTrainer

    train_loader, val_loader = create_dataloaders(
        str(data_root / "train"),
        str(data_root / "val"),
        batch_size=4,
        image_size=112,
        num_workers=0,
    )
    trainer = DomainGeneralizationTrainer(
        backbone_name="mobilenet_v3_small",
        n_diseases=4,
        pretrained=False,
        lr=1e-3,
        device="cpu",
    )
    for epoch in range(epochs):
        trainer.train_epoch(train_loader, epoch)
    val_metrics = trainer.evaluate(val_loader)
    trainer.save_checkpoint(checkpoint, epoch=epochs, metrics=val_metrics)
    return val_metrics


def _make_leaf_image(size=224):
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 60), (170, 190), (120, 180, 60), -1)
    return image


def _setup(tmp_path):
    _make_leaf_dataset(tmp_path / "train", per_class=1)
    _make_leaf_dataset(tmp_path / "val", per_class=1)
    checkpoint = str(tmp_path / "dg.pth")
    val_metrics = _train_tiny_dg_model(tmp_path, checkpoint)
    image_path = str(tmp_path / "leaf.jpg")
    cv2.imwrite(image_path, _make_leaf_image())
    return checkpoint, image_path, val_metrics


def test_full_pipeline_end_to_end(tmp_path):
    checkpoint, image_path, val_metrics = _setup(tmp_path)
    assert 0.0 <= val_metrics["disease_accuracy"] <= 1.0

    from src.inference.pipeline import DiagnosticPipeline

    pipeline = DiagnosticPipeline(checkpoint, device="cpu", image_size=112)
    result = pipeline.run_file(image_path)

    assert isinstance(result.passed, bool)
    assert len(result.leaves) >= 1
    for leaf in result.leaves:
        assert leaf.disease_idx < 4
        assert 0 <= leaf.severity_idx < 3
        assert 0.0 <= leaf.confidence <= 1.0
        assert leaf.isolated_image.size > 0


def test_api_serves_trained_model_end_to_end(tmp_path, monkeypatch):
    import server.main as api

    checkpoint, image_path, _ = _setup(tmp_path)

    monkeypatch.setattr(api, "MODEL_PATH", checkpoint)
    monkeypatch.setattr(api, "DETECTOR_PATH", None)
    api._pipeline = None

    from fastapi.testclient import TestClient

    client = TestClient(api.app)
    with open(image_path, "rb") as handle:
        resp = client.post(
            "/diagnose",
            files={"image": ("leaf.jpg", handle, "image/jpeg")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["passed"] is True
    assert len(body["leaves"]) >= 1
    leaf = body["leaves"][0]
    assert leaf["disease_idx"] < 4
    assert 0.0 <= leaf["confidence"] <= 1.0
