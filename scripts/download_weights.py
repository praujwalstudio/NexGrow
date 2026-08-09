"""Download pre-trained weights for NexGrow models."""

import os
import sys
from pathlib import Path

import click

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


AVAILABLE_MODELS = {
    "teacher_ensemble": {
        "description": "Swin Transformer + ConvNeXt teacher ensemble",
        "size": "~350 MB",
        "path": "models/teacher_ensemble/",
    },
    "student_model": {
        "description": "Distilled MobileNetV4 student model",
        "size": "~15 MB",
        "path": "models/student_model/",
    },
    "yolo11_leaf": {
        "description": "YOLO11 cotton leaf detection model",
        "size": "~25 MB",
        "path": "models/detection/",
    },
    "sam2_hiera": {
        "description": "SAM2 Hiera Large segmentation model",
        "size": "~900 MB",
        "path": "models/detection/",
    },
    "quantized_int8": {
        "description": "INT8 quantized student model for edge deployment",
        "size": "~1.5 MB",
        "path": "models/quantized/",
    },
}


@click.command()
@click.option("--model", default="all", help="Model to download (or 'all' for all models).")
@click.option("--output-dir", default="models/", help="Output directory.")
def download(model, output_dir):
    """Download pre-trained model weights."""
    if model == "all":
        models_to_download = list(AVAILABLE_MODELS.keys())
    elif model in AVAILABLE_MODELS:
        models_to_download = [model]
    else:
        click.echo(f"Unknown model: {model}")
        click.echo(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        sys.exit(1)

    for model_name in models_to_download:
        info = AVAILABLE_MODELS[model_name]
        target_dir = os.path.join(output_dir, info["path"].replace("models/", ""))
        Path(target_dir).mkdir(parents=True, exist_ok=True)

        click.echo(f"Preparing {model_name}...")
        click.echo(f"  Description: {info['description']}")
        click.echo(f"  Size: {info['size']}")
        click.echo(f"  Target: {target_dir}")

        click.echo(f"  Directory prepared: {target_dir}")

    click.echo("\nSetup complete.")
    click.echo("Note: Actual weights must be obtained from the training pipeline")
    click.echo("or transferred from a model registry (WandB, HuggingFace Hub, etc.).")


if __name__ == "__main__":
    download()
