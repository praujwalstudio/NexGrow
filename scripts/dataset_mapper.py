"""Dataset Mapper — NexGrow

Renames and restructures downloaded Kaggle/Mendeley cotton disease datasets
into the exact folder layout expected by CottonLeafDataset:

    output_root/<disease_name>/<severity>/image.jpg

Usage
-----
    python scripts/dataset_mapper.py \\
        --source data/raw_kaggle_dataset \\
        --output data/raw \\
        --mapping configs/dataset_map.yaml   # optional custom override

Supports all 4 recommended source datasets out of the box via built-in
alias tables. Unknown folders are listed in a SKIPPED report so nothing is
silently dropped.

Severity Assignment
-------------------
Because no public cotton dataset ships severity labels, this script assigns
them via an OpenCV lesion-coverage proxy:

    lesion_ratio = n_dark_pixels / n_leaf_pixels

    < 0.10  →  Low
    0.10–0.30  →  Moderate
    > 0.30  →  High

Pass --no-severity-proxy to skip this and place everything under Low/
(safe for disease-only training).
"""

import os
import shutil
import sys
from pathlib import Path

import click
import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import DISEASE_CLASSES, SEVERITY_LEVELS

# ---------------------------------------------------------------------------
# Built-in alias map  (source folder name → NexGrow canonical name)
# Add/edit here if a new dataset uses different names.
# ---------------------------------------------------------------------------
_BUILTIN_ALIASES: dict[str, str] = {
    # ── Bacterial Blight variants ──────────────────────────────────────────
    "bacterial blight": "Bacterial_Blight",
    "bacterialblight": "Bacterial_Blight",
    "bacterial_blight": "Bacterial_Blight",
    "blight": "Bacterial_Blight",
    # ── CLCuD / Curl Virus ─────────────────────────────────────────────────
    "clcud": "CLCuD",
    "curl virus": "CLCuD",
    "curl_virus": "CLCuD",
    "cotton leaf curl": "CLCuD",
    "cotton_leaf_curl": "CLCuD",
    "leaf curl disease": "CLCuD",
    # ── Leaf Curl (morphological, distinct from CLCuD virus) ───────────────
    "leaf curl": "Leaf_Curl",
    "leaf_curl": "Leaf_Curl",
    "leafcurl": "Leaf_Curl",
    # ── Alternaria ─────────────────────────────────────────────────────────
    "alternaria": "Alternaria_Leaf_Spot",
    "alternaria leaf spot": "Alternaria_Leaf_Spot",
    "alternaria_leaf_spot": "Alternaria_Leaf_Spot",
    "alternaria spot": "Alternaria_Leaf_Spot",
    # ── Grey Mildew / Areolate Mildew ──────────────────────────────────────
    "grey mildew": "Grey_Mildew",
    "grey_mildew": "Grey_Mildew",
    "gray mildew": "Grey_Mildew",
    "areolate mildew": "Grey_Mildew",
    "powdery mildew": "Grey_Mildew",
    "powdery_mildew": "Grey_Mildew",
    # ── Leaf Spot (generic) ────────────────────────────────────────────────
    "leaf spot": "Leaf_Spot",
    "leaf_spot": "Leaf_Spot",
    "leafspot": "Leaf_Spot",
    # ── Red Rust ───────────────────────────────────────────────────────────
    "red rust": "Red_Rust",
    "red_rust": "Red_Rust",
    "rust": "Red_Rust",
    # ── Target Spot ────────────────────────────────────────────────────────
    "target spot": "Target_Spot",
    "target_spot": "Target_Spot",
    "targetspot": "Target_Spot",
    # ── Bacterial Light ────────────────────────────────────────────────────
    "bacterial light": "Bacterial_Light",
    "bacterial_light": "Bacterial_Light",
    "bacteriallight": "Bacterial_Light",
    # ── Fusarium Wilt ──────────────────────────────────────────────────────
    "fusarium wilt": "Fusarium_Wilt",
    "fusarium_wilt": "Fusarium_Wilt",
    "fusarium": "Fusarium_Wilt",
    # ── Verticillium Wilt ──────────────────────────────────────────────────
    "verticillium wilt": "Verticillium_Wilt",
    "verticillium_wilt": "Verticillium_Wilt",
    "verticillium": "Verticillium_Wilt",
    # ── Anthracnose ────────────────────────────────────────────────────────
    "anthracnose": "Anthracnose",
    "anthracnose on cotton": "Anthracnose",
    # ── Cercospora ─────────────────────────────────────────────────────────
    "cercospora": "Cercospora_Leaf_Spot",
    "cercospora leaf spot": "Cercospora_Leaf_Spot",
    "cercospora_leaf_spot": "Cercospora_Leaf_Spot",
    # ── Root Rot ───────────────────────────────────────────────────────────
    "root rot": "Root_Rot",
    "root_rot": "Root_Rot",
    # ── Healthy ────────────────────────────────────────────────────────────
    "healthy": "Healthy",
    "normal": "Healthy",
    "disease free": "Healthy",
    "disease_free": "Healthy",
    "no disease": "Healthy",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Severity proxy
# ---------------------------------------------------------------------------

def _assign_severity(image_path: Path) -> str:
    """Assign Low/Moderate/High based on lesion coverage ratio via OpenCV."""
    img = cv2.imread(str(image_path))
    if img is None:
        return "Low"

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Green leaf mask
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_green, upper_green)

    leaf_pixels = int(leaf_mask.sum() / 255)
    if leaf_pixels == 0:
        return "Low"

    # Lesion = dark-brown / yellow pixels NOT in the green range
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Non-leaf, non-background (very dark) pixels are lesion candidates
    non_leaf_mask = cv2.bitwise_not(leaf_mask)
    _, bg_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY_INV)
    lesion_mask = cv2.bitwise_and(non_leaf_mask, cv2.bitwise_not(bg_mask))
    lesion_pixels = int(lesion_mask.sum() / 255)

    ratio = lesion_pixels / (leaf_pixels + lesion_pixels)

    if ratio < 0.10:
        return "Low"
    elif ratio < 0.30:
        return "Moderate"
    else:
        return "High"


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------

def _build_alias_map(custom_yaml: str | None) -> dict[str, str]:
    aliases = dict(_BUILTIN_ALIASES)
    if custom_yaml and Path(custom_yaml).exists():
        with open(custom_yaml, encoding="utf-8") as f:
            custom = yaml.safe_load(f) or {}
        for src, tgt in custom.items():
            aliases[src.lower().strip()] = tgt
        click.echo(f"  Loaded {len(custom)} custom alias(es) from {custom_yaml}")
    return aliases


def _resolve_disease(folder_name: str, alias_map: dict[str, str]) -> str | None:
    key = folder_name.lower().strip()
    # Exact alias match
    if key in alias_map:
        candidate = alias_map[key]
        if candidate in DISEASE_CLASSES:
            return candidate
    # Direct canonical match (case-insensitive)
    for cls in DISEASE_CLASSES:
        if cls.lower() == key:
            return cls
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--source", required=True,
    help="Root of the downloaded dataset (can be nested).",
)
@click.option(
    "--output", default="data/raw",
    help="Output root for organised images (default: data/raw).",
)
@click.option(
    "--mapping", default=None,
    help="Optional YAML file with extra alias overrides.",
)
@click.option(
    "--no-severity-proxy", "skip_severity", is_flag=True,
    help="Skip OpenCV severity proxy and put all images under Low/.",
)
@click.option(
    "--dry-run", is_flag=True,
    help="Show what would be moved without copying files.",
)
@click.option(
    "--copy/--move", "do_copy", default=True,
    help="Copy files (default) or move them.",
)
def map_dataset(source, output, mapping, skip_severity, dry_run, do_copy):
    """Rename and restructure a raw Kaggle/Mendeley dataset into NexGrow layout."""
    source_root = Path(source)
    output_root = Path(output)

    if not source_root.exists():
        click.echo(f"ERROR: source directory not found: {source_root}")
        sys.exit(1)

    alias_map = _build_alias_map(mapping)

    # Collect every image file under the source root
    all_images = [
        p for p in source_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    if not all_images:
        click.echo("No images found under source directory.")
        sys.exit(1)

    click.echo(f"\nFound {len(all_images)} image(s) under: {source_root}")
    click.echo(f"Output root          : {output_root}")
    click.echo(f"Severity proxy       : {'DISABLED (all → Low)' if skip_severity else 'ENABLED (OpenCV lesion coverage)'}")
    click.echo(f"Mode                 : {'DRY RUN' if dry_run else ('COPY' if do_copy else 'MOVE')}\n")

    stats = {
        "mapped": 0,
        "skipped": 0,
        "severity": {s: 0 for s in SEVERITY_LEVELS},
        "disease": {d: 0 for d in DISEASE_CLASSES},
    }
    skipped_folders: set[str] = set()

    for img_path in all_images:
        # The disease folder is the first meaningful directory under source_root
        parts = img_path.relative_to(source_root).parts
        if not parts:
            skipped_folders.add("<root>")
            stats["skipped"] += 1
            continue

        raw_disease_folder = parts[0]
        disease = _resolve_disease(raw_disease_folder, alias_map)

        if disease is None:
            skipped_folders.add(raw_disease_folder)
            stats["skipped"] += 1
            continue

        # Severity
        if skip_severity:
            severity = "Low"
        else:
            severity = _assign_severity(img_path)

        dest_dir = output_root / disease / severity
        dest_path = dest_dir / img_path.name

        # Resolve filename collision
        if not dry_run and dest_path.exists():
            stem = img_path.stem
            suffix = img_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            if do_copy:
                shutil.copy2(img_path, dest_path)
            else:
                shutil.move(str(img_path), str(dest_path))
        else:
            click.echo(f"  [DRY] {img_path.relative_to(source_root)} → {disease}/{severity}/{img_path.name}")

        stats["mapped"] += 1
        stats["severity"][severity] += 1
        stats["disease"][disease] += 1

    # ── Summary ──────────────────────────────────────────────────────────
    click.echo("\n" + "=" * 60)
    click.echo("  DATASET MAPPING SUMMARY")
    click.echo("=" * 60)
    click.echo(f"  Images mapped  : {stats['mapped']}")
    click.echo(f"  Images skipped : {stats['skipped']}")

    if skipped_folders:
        click.echo("\n  SKIPPED folders (unknown disease names):")
        for f in sorted(skipped_folders):
            click.echo(f"    - '{f}'  ← add to alias map if this is a known disease")

    click.echo("\n  Disease class breakdown:")
    for disease in DISEASE_CLASSES:
        count = stats["disease"][disease]
        bar = "█" * min(count // 10, 40)
        status = "✓" if count > 0 else "✗ EMPTY"
        click.echo(f"    {disease:<30} {count:>5}  {bar}  {status}")

    click.echo("\n  Severity breakdown:")
    for sev in SEVERITY_LEVELS:
        click.echo(f"    {sev:<10} : {stats['severity'][sev]}")

    # Warn about empty classes
    empty = [d for d in DISEASE_CLASSES if stats["disease"][d] == 0]
    if empty:
        click.echo(f"\n  ⚠ WARNING: {len(empty)} class(es) have 0 images:")
        for d in empty:
            click.echo(f"    - {d}")
        click.echo("  These classes will be SKIPPED by the dataset loader.")
        click.echo("  Add more source datasets or remove them from DISEASE_CLASSES in config.py.")

    if dry_run:
        click.echo("\n  (dry run — nothing written)")
    else:
        click.echo(f"\n  Output written to: {output_root}")
    click.echo("=" * 60)


if __name__ == "__main__":
    map_dataset()
