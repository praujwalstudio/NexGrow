"""Prepare raw cotton leaf images for training.

Pipeline: quality filter -> leaf detection (YOLO11, optional) -> SAM isolation
(optional) -> organized output at data/processed/<disease>/<severity>/.

Input layouts supported:
    data/raw/<disease>/<severity>/*.jpg   (severity-preserving)
    data/raw/<disease>/*.jpg              (severity defaults to Low, with warning)
"""

import os
import sys
from pathlib import Path

import click
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.quality_assessment import QualityAssessor
from src.preprocessing.sam_isolation import SAMIsolator
from src.utils.config import DISEASE_CLASSES, SEVERITY_LEVELS

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _load_detector(path):
    if not path:
        return None
    try:
        from ultralytics import YOLO

        if os.path.exists(path):
            return YOLO(path)
        click.echo(f"  WARNING: detector not found at {path}; using full-image crops.")
    except ImportError:
        click.echo("  WARNING: ultralytics not installed; using full-image crops.")
    return None


def _detect_leaves(detector, image, min_confidence):
    if detector is None:
        h, w = image.shape[:2]
        return [((0, 0, w, h), 1.0)]
    try:
        results = detector(image, conf=min_confidence, verbose=False)[0]
        boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                boxes.append(((x1, y1, x2, y2), float(box.conf[0])))
        if boxes:
            return boxes
    except Exception:
        pass
    h, w = image.shape[:2]
    return [((0, 0, w, h), 1.0)]


def _infer_labels(image_path: Path, input_root: Path) -> tuple:
    parts = image_path.relative_to(input_root).parts
    disease = parts[0] if parts else None
    severity = "Low"
    warning = None
    if len(parts) > 1 and parts[1] in SEVERITY_LEVELS:
        severity = parts[1]
    elif len(parts) > 1:
        warning = f"  NOTE: '{parts[1]}' is not a valid severity; using 'Low'"
    if disease not in DISEASE_CLASSES:
        return None, None, f"  SKIP: unknown disease folder '{disease}'"
    return disease, severity, warning


@click.command()
@click.option("--input", "input_dir", required=True, help="Raw images root (data/raw).")
@click.option("--output", "output_dir", default="data/processed", help="Output root.")
@click.option("--detector", default=None, help="Path to YOLO11 weights (optional).")
@click.option("--min-confidence", default=0.25, help="Detection confidence threshold.")
@click.option("--dry-run", is_flag=True, help="Report what would be processed without writing.")
def prepare(input_dir, output_dir, detector, min_confidence, dry_run):
    """Filter, detect, isolate, and organize raw leaf images."""
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    if not input_root.exists():
        click.echo(f"ERROR: input directory not found: {input_root}")
        sys.exit(1)

    assessor = QualityAssessor()
    detector_model = _load_detector(detector)
    isolator = SAMIsolator()

    images = sorted(p for ext in IMAGE_EXTS for p in input_root.rglob(f"*{ext}"))
    if not images:
        click.echo("No images found.")
        sys.exit(1)

    click.echo(f"Found {len(images)} image(s) under {input_root}")
    click.echo(f"Output root: {output_root}")
    if dry_run:
        click.echo("DRY RUN: no files will be written.")

    stats = {"processed": 0, "rejected": 0, "leaves": 0, "isolated": 0}
    for image_path in images:
        disease, severity, warning = _infer_labels(image_path, input_root)
        if warning:
            click.echo(warning)
        if disease is None:
            stats["rejected"] += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            click.echo(f"  SKIP: cannot read {image_path}")
            stats["rejected"] += 1
            continue

        quality = assessor.assess(image, path=str(image_path))
        if not quality.passed:
            click.echo(
                f"  REJECT {image_path.name} (blur={quality.is_blurry}, "
                f"brightness={quality.mean_brightness:.0f})"
            )
            stats["rejected"] += 1
            continue

        boxes = _detect_leaves(detector_model, image, min_confidence)
        target_dir = output_root / disease / severity
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)

        for i, ((x1, y1, x2, y2), _conf) in enumerate(boxes):
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                isolated = isolator.isolate(crop)
                stats["isolated"] += 1
            except Exception:
                isolated = crop
            out_path = target_dir / f"{image_path.stem}_leaf{i + 1}.jpg"
            if not dry_run:
                cv2.imwrite(str(out_path), isolated)
            stats["leaves"] += 1

        stats["processed"] += 1

    click.echo("\nSummary:")
    click.echo(f"  Images processed:    {stats['processed']}")
    click.echo(f"  Images rejected:     {stats['rejected']}")
    click.echo(f"  Leaves detected:     {stats['leaves']}")
    click.echo(f"  Leaves isolated:     {stats['isolated']}")
    if dry_run:
        click.echo("(dry run — nothing written)")
    else:
        click.echo(f"Output written to {output_root}")


if __name__ == "__main__":
    prepare()
