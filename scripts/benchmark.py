"""Benchmark model inference latency for edge deployment planning.

Benchmarks a trained checkpoint as FP32 ONNX and optionally INT8, printing
mean/p50/p95 latencies and model sizes for hardware targeting.
"""

import os
import sys

import click

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.factory import MultiTaskModel
from src.utils.quantization import Quantizer

OUTPUT_NAMES = ["disease", "severity"]


def _benchmark_report(quantizer: Quantizer, onnx_path: str, n_runs: int) -> dict:
    bench = quantizer.benchmark_onnx(onnx_path, n_runs=n_runs)
    click.echo(f"    Size:      {bench['model_size_mb']:.2f} MB")
    click.echo(f"    Mean:      {bench['mean_ms']:.2f} ms")
    click.echo(f"    P50:       {bench['p50_ms']:.2f} ms")
    click.echo(f"    P95:       {bench['p95_ms']:.2f} ms")
    return bench


@click.command()
@click.option("--model-path", required=True, help="Trained NexGrow checkpoint (.pth).")
@click.option("--onnx-path", default=None, help="Pre-exported ONNX model (skips export/quantize).")
@click.option("--output-dir", default="models/benchmark", help="Where to write ONNX artifacts.")
@click.option("--input-size", default=224, help="Input image size.")
@click.option("--n-runs", default=200, help="Number of timed runs.")
@click.option("--quantize", is_flag=True, help="Also benchmark INT8 quantized model.")
def benchmark(model_path, onnx_path, output_dir, input_size, n_runs, quantize):
    """Benchmark FP32 (and optionally INT8) ONNX inference latency."""
    os.makedirs(output_dir, exist_ok=True)

    if onnx_path:
        if not os.path.exists(onnx_path):
            click.echo(f"ERROR: ONNX model not found: {onnx_path}")
            sys.exit(1)
        quantizer = Quantizer(None, input_shape=(1, 3, input_size, input_size))
        click.echo(f"Benchmarking pre-exported ONNX: {onnx_path}")
        _benchmark_report(quantizer, onnx_path, n_runs)
        return

    click.echo(f"Loading checkpoint: {model_path}")
    model = MultiTaskModel.from_checkpoint(model_path)
    model.eval()
    quantizer = Quantizer(model, input_shape=(1, 3, input_size, input_size))

    fp32_path = os.path.join(output_dir, "nexgrow_fp32.onnx")
    click.echo("Exporting FP32 ONNX...")
    quantizer.export_onnx(fp32_path, output_names=OUTPUT_NAMES)
    click.echo("FP32 benchmark:")
    fp32 = _benchmark_report(quantizer, fp32_path, n_runs)

    if quantize:
        int8_path = os.path.join(output_dir, "nexgrow_int8.onnx")
        click.echo("Applying INT8 quantization...")
        quantizer.quantize_onnx(fp32_path, int8_path)
        click.echo("INT8 benchmark:")
        int8 = _benchmark_report(quantizer, int8_path, n_runs)
        click.echo(
            f"\nSpeedup: {fp32['mean_ms'] / int8['mean_ms']:.2f}x "
            f"({fp32['model_size_mb']:.2f} MB -> {int8['model_size_mb']:.2f} MB)"
        )


if __name__ == "__main__":
    benchmark()
