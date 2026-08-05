import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional


class Quantizer:
    """Post-training INT8 quantization for edge deployment.

    Compresses PyTorch models via ONNX export and dynamic quantization
    to achieve sub-millisecond inference on mobile devices.
    """

    def __init__(self, model: Optional[nn.Module], input_shape: tuple = (1, 3, 224, 224)):
        self.model = model.eval() if model is not None else None
        self.input_shape = input_shape

    def export_onnx(
        self,
        output_path: str,
        opset_version: int = 14,
        output_names: Optional[list] = None,
    ) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        dummy_input = torch.randn(*self.input_shape)
        output_names = output_names or ["output"]
        dynamic_axes = {"input": {0: "batch_size"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
        return output_path

    def quantize_onnx(
        self,
        onnx_path: str,
        output_path: Optional[str] = None,
        weight_type: str = "QInt8",
    ) -> str:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        if output_path is None:
            output_path = onnx_path.replace(".onnx", "_int8.onnx")

        quant_type = QuantType.QInt8 if weight_type == "QInt8" else QuantType.QUInt8
        quantize_dynamic(
            model_input=onnx_path,
            model_output=output_path,
            weight_type=quant_type,
        )
        return output_path

    def get_size_mb(self, path: str) -> float:
        return Path(path).stat().st_size / (1024 * 1024)

    def benchmark_onnx(self, onnx_path: str, n_runs: int = 100) -> dict:
        import onnxruntime as ort
        import time

        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(*self.input_shape).astype(np.float32)

        for _ in range(10):
            session.run(None, {input_name: dummy})

        latencies = []
        for _ in range(n_runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy})
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "model_size_mb": self.get_size_mb(onnx_path),
        }
