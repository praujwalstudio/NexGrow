"""Download and map NexGrow Cotton Disease Datasets.

Automates downloading and restructuring the 4 primary cotton disease datasets:
  1. Mendeley / Shamim Ripon Dataset (Primary Core)
  2. CLCuD Dataset (Cotton Leaf Curl Virus - Janmejay Bhoi)
  3. Cotton Crop Disease Detection Dataset (Paridhi Jain)
  4. COT-AD (Cotton Analysis Dataset)

Usage:
------
    python scripts/download_datasets.py --dataset all --auto-map
    python scripts/download_datasets.py --dataset clcud --auto-map
"""

import os
import subprocess
import sys
from pathlib import Path

import click
import kagglehub

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATASET_REGISTRY = {
    "mendeley_ripon": {
        "title": "Mendeley / Shamim Ripon Dataset (2025)",
        "kaggle_handle": "shamimripon/cotton-leaf-image-dataset",
        "mendeley_url": "https://data.mendeley.com/datasets/t9hgvk2h9p/1",
        "classes": ["Bacterial_Blight", "Alternaria_Leaf_Spot", "Fusarium_Wilt", "Verticillium_Wilt", "Healthy"],
        "description": "Primary core dataset covering 4 key fungal/bacterial diseases and healthy leaves.",
    },
    "clcud": {
        "title": "CLCuD Dataset (Janmejay Bhoi)",
        "kaggle_handle": "janmejaybhoi/cotton-disease-dataset",
        "classes": ["CLCuD", "Leaf_Curl", "Bacterial_Blight", "Healthy"],
        "description": "Cotton leaf curl virus and bacterial blight disease samples.",
    },
    "paridhi_jain": {
        "title": "Cotton Crop Disease Detection (Paridhi Jain)",
        "kaggle_handle": "paridhijain/cotton-crop-disease-detection",
        "classes": ["Target_Spot", "Anthracnose", "Grey_Mildew"],
        "description": "Target spot, Anthracnose, and Grey Mildew disease samples.",
    },
    "cot_ad": {
        "title": "COT-AD (Cotton Analysis Dataset)",
        "kaggle_handle": "cot-ad/cot-ad",
        "classes": ["Red_Rust", "Cercospora_Leaf_Spot", "Root_Rot"],
        "description": "Red Rust, Cercospora Leaf Spot, and Root Rot disease samples.",
    },
}


@click.command()
@click.option(
    "--dataset",
    default="all",
    type=click.Choice(list(DATASET_REGISTRY.keys()) + ["all"]),
    help="Dataset to download (or 'all').",
)
@click.option(
    "--download-dir",
    default="data/raw_downloads",
    help="Directory to save downloaded dataset archives.",
)
@click.option(
    "--auto-map/--no-auto-map",
    default=True,
    help="Automatically run dataset_mapper.py after downloading to restructure into data/raw/.",
)
@click.option(
    "--auto-prepare",
    is_flag=True,
    help="Automatically run prepare_data.py after mapping to generate background-isolated leaves in data/processed/.",
)
def download_datasets(dataset, download_dir, auto_map, auto_prepare):
    """Download and structure cotton disease datasets for NexGrow."""
    targets = list(DATASET_REGISTRY.keys()) if dataset == "all" else [dataset]
    download_root = Path(download_dir)
    download_root.mkdir(parents=True, exist_ok=True)

    click.echo("\n============================================================")
    click.echo("  NEXGROW DATASET COLLECTION & MAPPING UTILITY")
    click.echo("============================================================")
    click.echo(f"  Target Datasets : {', '.join(targets)}")
    click.echo(f"  Download Root   : {download_root.resolve()}")
    click.echo(f"  Auto-map        : {'ENABLED (data/raw)' if auto_map else 'DISABLED'}")
    click.echo(f"  Auto-prepare    : {'ENABLED (data/processed)' if auto_prepare else 'DISABLED'}\n")

    downloaded_paths = []

    for ds_key in targets:
        info = DATASET_REGISTRY[ds_key]
        handle = info["kaggle_handle"]
        title = info["title"]

        click.echo(f"\n---> Fetching: {title}")
        click.echo(f"     Classes  : {', '.join(info['classes'])}")
        click.echo(f"     Handle   : {handle}")

        try:
            download_path = kagglehub.dataset_download(handle)
            click.echo(f"     [OK] Downloaded to: {download_path}")
            downloaded_paths.append((ds_key, download_path))

            if auto_map:
                click.echo(f"     Running dataset_mapper.py on {ds_key}...")
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/dataset_mapper.py",
                        "--source",
                        str(download_path),
                        "--output",
                        "data/raw",
                        "--copy",
                    ],
                    check=True,
                )

        except Exception as err:
            click.echo(f"     [WARN] Could not download {handle} directly: {err}")
            click.echo(f"     To manually add {title}:")
            if "mendeley_url" in info:
                click.echo(f"       1. Download from Mendeley: {info['mendeley_url']}")
            click.echo(f"       2. Place zip/extracted folder in: {download_root / ds_key}")
            click.echo(f"       3. Run: python scripts/dataset_mapper.py --source data/raw_downloads/{ds_key} --output data/raw")

    if auto_prepare and downloaded_paths:
        click.echo("\n---> Running prepare_data.py for Quality Assessment and SAM Background Removal...")
        subprocess.run(
            [
                sys.executable,
                "scripts/prepare_data.py",
                "--input-dir",
                "data/raw",
                "--output-dir",
                "data/processed",
            ],
            check=True,
        )

    click.echo("\n============================================================")
    click.echo("  DATASET COLLECTION STEP COMPLETE")
    click.echo("============================================================")


if __name__ == "__main__":
    download_datasets()
