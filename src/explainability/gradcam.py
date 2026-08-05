import contextlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAMPlusPlus:
    """Grad-CAM++ implementation for cotton leaf disease visualization.

    Generates heatmaps highlighting specific lesion regions that contributed
    most to the model's diagnosis, enabling expert validation and building
    farmer trust.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        device: str = "cuda",
    ) -> tuple[np.ndarray, int]:
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients[0]
        activations = self.activations[0]

        grads_power_2 = grads**2
        grads_power_3 = grads**3
        activations_power_4 = activations**4

        a = grads_power_2
        b = 2 * grads_power_3 * activations
        c = grads_power_2 * activations_power_4

        a = a.flatten(1).sum(dim=1)
        b = b.flatten(1).sum(dim=1)
        c = c.flatten(1).sum(dim=1)

        weights = a / (2 * a + b + c + 1e-7)
        weights = weights.view(weights.size(0), 1, 1)

        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)

        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))

        return cam, target_class

    def overlay(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        if image.shape[:2] != cam.shape[:2]:
            cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        cam_uint8 = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return overlay

    def save_heatmap(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        output_path: str,
        alpha: float = 0.5,
    ):
        overlay = self.overlay(image, cam, alpha)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


with contextlib.suppress(ImportError):
    from torch.nn import functional as F
