"""End-to-end inference CLI for NexGrow cotton disease diagnosis.

Ingests imagery -> Quality check -> YOLO detection -> SAM isolation ->
Multi-task classification -> Explainability -> Weather-aware treatment advice.
"""

import os
import sys

import click
import cv2
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.explainability.gradcam import GradCAMPlusPlus
from src.inference.pipeline import DiagnosticPipeline, DiseaseHeadModel, find_target_layer
from src.utils.weather import WeatherAdvisor


def _to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@click.command()
@click.option("--image", required=True, help="Path to input image.")
@click.option("--model-path", required=True, help="Path to trained NexGrow checkpoint.")
@click.option("--detector", default=None, help="Path to YOLO11 weights (optional; falls back to full image).")
@click.option("--output-dir", default="outputs/", help="Directory for output visualizations.")
@click.option("--weather", is_flag=True, help="Enable weather-aware treatment advice.")
@click.option("--lat", default=20.5, help="Latitude for weather data.")
@click.option("--lon", default=78.9, help="Longitude for weather data.")
@click.option("--api-key", default=None, help="OpenWeatherMap API key (or OPENWEATHER_API_KEY env).")
@click.option("--device", default="auto", help="Device (cuda, cpu, or auto).")
def diagnose(image, model_path, detector, output_dir, weather, lat, lon, api_key, device):
    """Run full diagnostic pipeline on a cotton leaf image."""
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(model_path):
        click.echo(f"ERROR: checkpoint not found: {model_path}")
        click.echo("Train a model first via: python scripts/train.py distill|dg-train")
        sys.exit(1)

    click.echo(f"Processing: {image}")

    pipeline = DiagnosticPipeline(
        model_path=model_path,
        detector_path=detector,
        device=device,
    )

    # Step 1-4: quality, detection, isolation, classification
    result = pipeline.run_file(image)
    quality = result.quality
    if not result.passed:
        click.echo(
            f"  WARNING: Image failed quality check (blur={quality.is_blurry}, "
            f"brightness={quality.mean_brightness:.1f})"
        )
    else:
        click.echo(
            f"  Quality: PASSED (laplacian={quality.laplacian_var:.1f}, "
            f"brightness={quality.mean_brightness:.1f})"
        )

    click.echo(f"  Detected {len(result.leaves)} leaf region(s)")

    if not result.leaves:
        click.echo("ERROR: No leaf regions detected.")
        sys.exit(1)

    # Step 5: Explainability heatmaps
    target_layer = find_target_layer(pipeline.model.model.backbone)
    disease_wrapper = DiseaseHeadModel(pipeline.model).to(pipeline.device)
    if target_layer is not None:
        gradcam = GradCAMPlusPlus(disease_wrapper, target_layer)

    treatments = []
    for i, leaf in enumerate(result.leaves):
        click.echo(f"  --- Leaf {i + 1} ---")
        click.echo(f"    BBox: {leaf.bbox} (det_conf={leaf.det_conf:.2f})")
        click.echo(f"    Disease: {leaf.disease} ({leaf.disease_idx})")
        click.echo(f"    Severity: {leaf.severity}")
        click.echo(f"    Confidence: {leaf.confidence:.2%}")

        isolated_path = output_path / f"isolated_leaf_{i + 1}.jpg"
        cv2.imwrite(str(isolated_path), leaf.isolated_image)

        if target_layer is not None:
            input_tensor = pipeline.transform(leaf.isolated_image).unsqueeze(0).to(pipeline.device)
            try:
                cam, _ = gradcam.generate(input_tensor, target_class=leaf.disease_idx)
                gradcam.save_heatmap(
                    _to_rgb(leaf.isolated_image),
                    cam,
                    str(output_path / f"gradcam_heatmap_{i + 1}.jpg"),
                )
                click.echo(f"    Heatmap saved.")
            except Exception as exc:
                click.echo(f"    Grad-CAM failed ({exc}). Skipping.")

        if weather:
            advisor = WeatherAdvisor(api_key=api_key or os.getenv("OPENWEATHER_API_KEY"))
            weather_data = advisor.fetch_weather(lat, lon)
            if weather_data:
                advice = advisor.get_treatment(leaf.disease, weather_data)
                click.echo(f"    Treatment: {advice['treatment']}")
            else:
                click.echo("    Could not fetch weather data.")

    click.echo(f"\nDiagnostic report saved to {output_path}")
    click.echo("Pipeline complete.")


if __name__ == "__main__":
    diagnose()
