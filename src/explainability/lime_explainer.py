import numpy as np
import cv2
from pathlib import Path
from typing import Callable, Optional

try:
    from lime import lime_image
    from skimage.segmentation import slic
except ImportError:
    lime_image = None
    slic = None


class LIMEExplainer:
    """LIME-based local interpretable explanations for cotton leaf diagnosis.

    Generates superpixel-level explanations showing which image regions
    contributed most to a specific disease prediction.
    """

    def __init__(self, predict_fn: Callable, num_samples: int = 1000):
        self.predict_fn = predict_fn
        self.num_samples = num_samples
        self.explainer = None
        if lime_image is not None:
            self.explainer = lime_image.LimeImageExplainer()

    def explain(
        self,
        image: np.ndarray,
        top_labels: int = 3,
        hide_color: int = 0,
        num_features: int = 10,
        segmentation_fn: Optional[Callable] = None,
    ) -> dict:
        if self.explainer is None:
            raise ImportError("lime not installed. Run: pip install lime scikit-image")

        if segmentation_fn is None and slic is not None:
            def segmentation_fn(img):
                return slic(img, n_segments=50, compactness=10)

        explanation = self.explainer.explain_instance(
            image,
            self.predict_fn,
            top_labels=top_labels,
            hide_color=hide_color,
            num_features=num_features,
            segmentation_fn=segmentation_fn,
            num_samples=self.num_samples,
        )

        return {
            "segments": explanation.segments,
            "local_exp": explanation.local_exp,
            "score": explanation.score,
            "top_labels": explanation.top_labels,
        }

    def get_visualization(
        self,
        image: np.ndarray,
        explanation: dict,
        label: int,
        num_features: int = 10,
    ) -> np.ndarray:
        vis = self.explainer.get_image_and_mask(
            label,
            positive_only=True,
            num_features=num_features,
            hide_rest=False,
        )
        return (vis[0] * 255).astype(np.uint8) if vis[0].max() <= 1.0 else vis[0].astype(np.uint8)

    def save_explanation(
        self,
        image: np.ndarray,
        explanation: dict,
        output_dir: str,
        label: int,
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        vis = self.get_visualization(image, explanation, label)
        cv2.imwrite(str(output_path / f"lime_label_{label}.png"), vis)

        segments = explanation["segments"]
        seg_vis = (segments.astype(float) / segments.max() * 255).astype(np.uint8)
        cv2.imwrite(str(output_path / "segments.png"), seg_vis)

    def explain_batch(
        self,
        images: list[np.ndarray],
        top_labels: int = 3,
    ) -> list[dict]:
        results = []
        for img in images:
            try:
                results.append(self.explain(img, top_labels=top_labels))
            except Exception:
                results.append(None)
        return results
