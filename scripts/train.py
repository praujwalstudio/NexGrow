"""Main training entry point for NexGrow cotton disease diagnosis.

Supports teacher-student distillation, domain generalization training,
and multi-task disease + severity prediction.
"""

import sys
import os
import click
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.dataset import create_dataloaders, apply_adasyn
from src.training.distillation import KnowledgeDistiller
from src.training.domain_generalization import DomainGeneralizationTrainer
from src.utils.quantization import Quantizer


@click.group()
def cli():
    """NexGrow Training CLI."""
    pass


@cli.command()
@click.option("--train-dir", required=True, help="Path to training data directory.")
@click.option("--val-dir", required=True, help="Path to validation data directory.")
@click.option("--epochs", default=50, help="Number of training epochs.")
@click.option("--batch-size", default=32, help="Batch size.")
@click.option("--lr", default=1e-4, help="Learning rate.")
@click.option("--teacher-path", default=None, help="Path to pre-trained teacher weights.")
@click.option("--student-out", default="models/student_model/student.pth", help="Output path for student model.")
@click.option("--device", default="cuda", help="Device (cuda or cpu).")
def distill(train_dir, val_dir, epochs, batch_size, lr, teacher_path, student_out, device):
    """Run teacher-student knowledge distillation."""
    click.echo("Initializing knowledge distillation...")

    train_loader, val_loader = create_dataloaders(
        train_dir, val_dir, batch_size=batch_size
    )

    try:
        from torchvision.models import swin_v2_b, ConvNeXt_Base
        teacher = swin_v2_b(pretrained=True)
    except ImportError:
        click.echo("Warning: Using placeholder teacher model.")
        teacher = torch.nn.Linear(224 * 224 * 3, 15)

    try:
        from torchvision.models import mobilenet_v4
        student = mobilenet_v4(pretrained=True)
    except ImportError:
        click.echo("Warning: Using placeholder student model.")
        student = torch.nn.Linear(224 * 224 * 3, 15)

    if teacher_path and os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))

    distiller = KnowledgeDistiller(teacher, student, lr=lr, device=device)

    best_acc = 0.0
    for epoch in range(epochs):
        train_metrics = distiller.train_epoch(train_loader, epoch)
        val_metrics = distiller.evaluate(val_loader)

        click.echo(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            distiller.save_student(student_out)
            click.echo(f"  -> Saved best model (acc={best_acc:.4f})")

    click.echo(f"Training complete. Best accuracy: {best_acc:.4f}")


@cli.command()
@click.option("--train-dir", required=True, help="Path to training data directory.")
@click.option("--val-dir", required=True, help="Path to validation data directory.")
@click.option("--epochs", default=50, help="Number of training epochs.")
@click.option("--batch-size", default=32, help="Batch size.")
@click.option("--checkpoint", default="models/student_model/dg_checkpoint.pth", help="Checkpoint output path.")
@click.option("--device", default="cuda", help="Device (cuda or cpu).")
def dg_train(train_dir, val_dir, epochs, batch_size, checkpoint, device):
    """Train with Domain Generalization (MixStyle)."""
    click.echo("Initializing domain generalization training...")

    train_loader, val_loader = create_dataloaders(
        train_dir, val_dir, batch_size=batch_size
    )

    try:
        from torchvision.models import mobilenet_v4
        model = mobilenet_v4(pretrained=True)
    except ImportError:
        model = torch.nn.Linear(224 * 224 * 3, 15)

    trainer = DomainGeneralizationTrainer(model=model, device=device)

    best_acc = 0.0
    for epoch in range(epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.evaluate(val_loader)

        click.echo(
            f"Epoch {epoch+1}/{epochs} | "
            f"Disease Acc: {val_metrics['disease_accuracy']:.4f} | "
            f"Severity Acc: {val_metrics['severity_accuracy']:.4f}"
        )

        avg_acc = (val_metrics["disease_accuracy"] + val_metrics["severity_accuracy"]) / 2
        if avg_acc > best_acc:
            best_acc = avg_acc
            trainer.save_checkpoint(checkpoint, epoch, val_metrics)
            click.echo(f"  -> Saved checkpoint (avg_acc={best_acc:.4f})")

    click.echo(f"DG training complete. Best avg accuracy: {best_acc:.4f}")


@cli.command()
@click.option("--model-path", required=True, help="Path to trained PyTorch model.")
@click.option("--output-dir", default="models/quantized", help="Output directory for quantized models.")
@click.option("--input-size", default=224, help="Input image size.")
def quantize(model_path, output_dir, input_size):
    """Quantize a trained model to INT8 ONNX for edge deployment."""
    click.echo(f"Loading model from {model_path}...")

    model = torch.load(model_path, map_location="cpu")
    if isinstance(model, dict) and "model_state_dict" in model:
        click.echo("Loaded checkpoint, extracting model state dict.")

    from torchvision.models import mobilenet_v4
    net = mobilenet_v4(pretrained=False)
    if isinstance(model, dict) and "model_state_dict" in model:
        net.load_state_dict(model["model_state_dict"], strict=False)

    quantizer = Quantizer(net, input_shape=(1, 3, input_size, input_size))

    onnx_path = os.path.join(output_dir, "nexgrow.onnx")
    quantized_path = os.path.join(output_dir, "nexgrow_int8.onnx")

    click.echo("Exporting to ONNX...")
    quantizer.export_onnx(onnx_path)

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
