"""Main training entry point for NexGrow cotton disease diagnosis.

Supports teacher-student distillation, domain generalization training,
and INT8 quantization for edge deployment. All settings are driven by
`configs/train.yaml` with CLI overrides.
"""

import os
import sys

import click
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.factory import MultiTaskModel, build_teacher
from src.training.dataset import create_dataloaders
from src.training.distillation import KnowledgeDistiller
from src.training.domain_generalization import DomainGeneralizationTrainer
from src.utils.config import N_DISEASES, TrainConfig, resolve_device
from src.utils.quantization import Quantizer


def _override(value, default):
    return value if value is not None else default


def _resolve_device(requested: str) -> str:
    return resolve_device(requested or "auto")


@click.group()
def cli():
    """NexGrow Training CLI."""


@cli.command()
@click.option("--config", default=None, help="Path to config YAML (default: configs/train.yaml).")
@click.option("--train-dir", default=None, help="Path to training data directory.")
@click.option("--val-dir", default=None, help="Path to validation data directory.")
@click.option("--epochs", type=int, default=None, help="Number of training epochs.")
@click.option("--batch-size", type=int, default=None, help="Batch size.")
@click.option("--lr", type=float, default=None, help="Learning rate.")
@click.option("--weight-decay", type=float, default=None, help="Weight decay.")
@click.option("--temperature", type=float, default=None, help="Distillation temperature.")
@click.option("--alpha", type=float, default=None, help="KD vs CE weighting.")
@click.option("--teacher-path", default=None, help="Path to pre-trained teacher weights.")
@click.option("--student-backbone", default=None, help="Student backbone name.")
@click.option("--student-out", default=None, help="Output path for student model.")
@click.option("--resume", default=None, help="Checkpoint to resume training from.")
@click.option("--device", default=None, help="Device (cuda, cpu, or auto).")
def distill(
    config,
    train_dir,
    val_dir,
    epochs,
    batch_size,
    lr,
    weight_decay,
    temperature,
    alpha,
    teacher_path,
    student_backbone,
    student_out,
    resume,
    device,
):
    """Run teacher-student knowledge distillation."""
    cfg = TrainConfig.load(config)
    device = _resolve_device(_override(device, cfg.training.device))
    train_dir = _override(train_dir, cfg.data.train_dir)
    val_dir = _override(val_dir, cfg.data.val_dir)
    epochs = _override(epochs, cfg.training.epochs)
    batch_size = _override(batch_size, cfg.training.batch_size)
    lr = _override(lr, cfg.training.lr)
    weight_decay = _override(weight_decay, cfg.training.weight_decay)
    temperature = _override(temperature, cfg.training.temperature)
    alpha = _override(alpha, cfg.training.alpha)
    student_backbone = _override(student_backbone, cfg.model.student_backbone)
    student_out = _override(student_out, cfg.paths.student_out)

    click.echo(f"Initializing distillation (student={student_backbone}, device={device})...")

    train_loader, val_loader = create_dataloaders(
        train_dir,
        val_dir,
        batch_size=batch_size,
        image_size=cfg.data.image_size,
        num_workers=0,
    )

    teacher = build_teacher(
        N_DISEASES,
        backbone_names=cfg.model.teacher_backbones,
        pretrained=cfg.model.teacher_pretrained,
    )
    student = MultiTaskModel(
        backbone_name=student_backbone,
        pretrained=cfg.model.student_pretrained,
    )

    if teacher_path and os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path, map_location=device), strict=False)

    start_epoch = 0
    if resume and os.path.exists(resume):
        student.load_checkpoint(resume)
        start_epoch = int(torch.load(resume, map_location="cpu").get("epoch", 0))
        click.echo(f"  Resumed from {resume} (epoch {start_epoch})")

    distiller = KnowledgeDistiller(
        teacher,
        student,
        temperature=temperature,
        alpha=alpha,
        lr=lr,
        weight_decay=weight_decay,
        t_max=epochs,
        device=device,
    )

    best_acc = 0.0
    for epoch in range(start_epoch, epochs):
        train_metrics = distiller.train_epoch(train_loader, epoch)
        val_metrics = distiller.evaluate(val_loader)

        click.echo(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Disease Acc: {val_metrics['disease_accuracy']:.4f}"
        )

        if val_metrics["disease_accuracy"] > best_acc:
            best_acc = val_metrics["disease_accuracy"]
            distiller.save_student(student_out, epoch=epoch + 1, metrics=val_metrics)
            click.echo(f"  -> Saved best model (acc={best_acc:.4f})")

    click.echo(f"Training complete. Best disease accuracy: {best_acc:.4f}")


@cli.command()
@click.option("--config", default=None, help="Path to config YAML (default: configs/train.yaml).")
@click.option("--train-dir", default=None, help="Path to training data directory.")
@click.option("--val-dir", default=None, help="Path to validation data directory.")
@click.option("--epochs", type=int, default=None, help="Number of training epochs.")
@click.option("--batch-size", type=int, default=None, help="Batch size.")
@click.option("--lr", type=float, default=None, help="Learning rate.")
@click.option("--weight-decay", type=float, default=None, help="Weight decay.")
@click.option("--checkpoint", default="models/teacher_ensemble/teacher_best.pth", help="Checkpoint output path.")
@click.option("--resume", default=None, help="Checkpoint to resume training from.")
@click.option("--device", default=None, help="Device (cuda, cpu, or auto).")
def teacher_train(
    config,
    train_dir,
    val_dir,
    epochs,
    batch_size,
    lr,
    weight_decay,
    checkpoint,
    resume,
    device,
):
    """Train Swin + ConvNeXt Teacher Ensemble individually."""
    cfg = TrainConfig.load(config)
    device = _resolve_device(_override(device, cfg.training.device))
    train_dir = _override(train_dir, cfg.data.train_dir)
    val_dir = _override(val_dir, cfg.data.val_dir)
    epochs = _override(epochs, cfg.training.epochs)
    batch_size = _override(batch_size, cfg.training.batch_size)
    lr = _override(lr, cfg.training.lr)
    weight_decay = _override(weight_decay, cfg.training.weight_decay)

    click.echo(f"Initializing Teacher Ensemble training (backbones={cfg.model.teacher_backbones}, device={device})...")

    train_loader, val_loader = create_dataloaders(
        train_dir,
        val_dir,
        batch_size=batch_size,
        image_size=cfg.data.image_size,
        num_workers=0,
    )

    teacher = build_teacher(
        N_DISEASES,
        backbone_names=cfg.model.teacher_backbones,
        pretrained=cfg.model.teacher_pretrained,
    ).to(device)

    optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    start_epoch = 0
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        teacher.load_state_dict(ckpt.get("model", ckpt))
        start_epoch = int(ckpt.get("epoch", 0))
        click.echo(f"  Resumed teacher from {resume} (epoch {start_epoch})")

    best_acc = 0.0
    for epoch in range(start_epoch, epochs):
        teacher.train()
        train_loss = 0.0
        d_correct = total = 0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            d_labels = batch["disease"].to(device)
            s_labels = batch.get("severity", torch.zeros_like(d_labels)).to(device)

            d_logits, s_logits = teacher(imgs)
            loss = torch.nn.functional.cross_entropy(d_logits, d_labels) + 0.5 * torch.nn.functional.cross_entropy(s_logits, s_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            d_correct += (d_logits.argmax(1) == d_labels).sum().item()
            total += d_labels.size(0)

        scheduler.step()
        train_acc = d_correct / max(total, 1)

        # Validation
        teacher.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                d_labels = batch["disease"].to(device)
                d_logits, _ = teacher(imgs)
                val_correct += (d_logits.argmax(1) == d_labels).sum().item()
                val_total += d_labels.size(0)

        val_acc = val_correct / max(val_total, 1)

        click.echo(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
            torch.save({"model": teacher.state_dict(), "epoch": epoch + 1, "best_acc": best_acc}, checkpoint)
            click.echo(f"  -> Saved best teacher (val_acc={best_acc:.4f})")

    click.echo(f"Teacher training complete. Best accuracy: {best_acc:.4f}")


@cli.command()
@click.option("--config", default=None, help="Path to config YAML (default: configs/train.yaml).")
@click.option("--train-dir", default=None, help="Path to training data directory.")
@click.option("--val-dir", default=None, help="Path to validation data directory.")
@click.option("--epochs", type=int, default=None, help="Number of training epochs.")
@click.option("--batch-size", type=int, default=None, help="Batch size.")
@click.option("--lr", type=float, default=None, help="Learning rate.")
@click.option("--weight-decay", type=float, default=None, help="Weight decay.")
@click.option("--backbone", default=None, help="Backbone name for feature extraction.")
@click.option("--checkpoint", default=None, help="Checkpoint output path.")
@click.option("--resume", default=None, help="Checkpoint to resume training from.")
@click.option("--device", default=None, help="Device (cuda, cpu, or auto).")
def dg_train(
    config,
    train_dir,
    val_dir,
    epochs,
    batch_size,
    lr,
    weight_decay,
    backbone,
    checkpoint,
    resume,
    device,
):
    """Train with Domain Generalization (multi-task feature learning)."""
    cfg = TrainConfig.load(config)
    device = _resolve_device(_override(device, cfg.training.device))
    train_dir = _override(train_dir, cfg.data.train_dir)
    val_dir = _override(val_dir, cfg.data.val_dir)
    epochs = _override(epochs, cfg.training.epochs)
    batch_size = _override(batch_size, cfg.training.batch_size)
    lr = _override(lr, cfg.training.lr)
    weight_decay = _override(weight_decay, cfg.training.weight_decay)
    backbone = _override(backbone, cfg.model.student_backbone)
    checkpoint = _override(checkpoint, cfg.paths.dg_checkpoint)

    click.echo(
        f"Initializing domain generalization training (backbone={backbone}, device={device})..."
    )

    train_loader, val_loader = create_dataloaders(
        train_dir,
        val_dir,
        batch_size=batch_size,
        image_size=cfg.data.image_size,
        num_workers=0,
    )

    trainer = DomainGeneralizationTrainer(
        backbone_name=backbone,
        pretrained=cfg.model.student_pretrained,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    start_epoch = 0
    if resume and os.path.exists(resume):
        start_epoch, _ = trainer.load_checkpoint(resume)
        click.echo(f"  Resumed from {resume} (epoch {start_epoch})")

    best_acc = 0.0
    for epoch in range(start_epoch, epochs):
        trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.evaluate(val_loader)

        click.echo(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Disease Acc: {val_metrics['disease_accuracy']:.4f} | "
            f"Severity Acc: {val_metrics['severity_accuracy']:.4f}"
        )

        avg_acc = (val_metrics["disease_accuracy"] + val_metrics["severity_accuracy"]) / 2
        if avg_acc > best_acc:
            best_acc = avg_acc
            trainer.save_checkpoint(checkpoint, epoch=epoch + 1, metrics=val_metrics)
            click.echo(f"  -> Saved checkpoint (avg_acc={best_acc:.4f})")

    click.echo(f"DG training complete. Best avg accuracy: {best_acc:.4f}")


@cli.command()
@click.option("--model-path", required=True, help="Path to trained NexGrow checkpoint.")
@click.option(
    "--output-dir", default="models/quantized", help="Output directory for quantized models."
)
@click.option("--input-size", default=224, help="Input image size.")
def quantize(model_path, output_dir, input_size):
    """Quantize a trained model to INT8 ONNX for edge deployment."""
    click.echo(f"Loading model from {model_path}...")

    model = MultiTaskModel.from_checkpoint(model_path)
    model.eval()

    quantizer = Quantizer(model, input_shape=(1, 3, input_size, input_size))

    onnx_path = os.path.join(output_dir, "nexgrow.onnx")
    quantized_path = os.path.join(output_dir, "nexgrow_int8.onnx")

    click.echo("Exporting to ONNX...")
    quantizer.export_onnx(onnx_path, output_names=["disease", "severity"])

    click.echo("Applying INT8 quantization...")
    quantizer.quantize_onnx(onnx_path, quantized_path)

    size = quantizer.get_size_mb(quantized_path)
    click.echo(f"Quantized model saved to {quantized_path} ({size:.2f} MB)")

    click.echo("Benchmarking...")
    bench = quantizer.benchmark_onnx(quantized_path)
    click.echo(f"  Mean latency: {bench['mean_ms']:.2f} ms")
    click.echo(f"  P95 latency: {bench['p95_ms']:.2f} ms")


if __name__ == "__main__":
    cli()
