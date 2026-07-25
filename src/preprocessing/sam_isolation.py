import cv2
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from segment_anything import SamPredictor, sam_model_registry, build_sam2
except ImportError:
    SamPredictor = None
    sam_model_registry = None
    build_sam2 = None


class SAMIsolator:
    """Isolates cotton leaves from complex backgrounds using SAM/SAM2.

    Generates binary masks and produces background-free leaf images
    for downstream pathological analysis.
    """

    def __init__(
        self,
        model_type: str = "sam2_hiera_large",
        checkpoint: str = "models/detection/sam2_hiera_large.pth",
        device: str = "cuda",
    ):
        self.device = device
        self.predictor = None
        self._load_model(model_type, checkpoint)

    def _load_model(self, model_type: str, checkpoint: str):
        if SamPredictor is None:
            return
        try:
            if "sam2" in model_type:
                sam = build_sam2(model_type, checkpoint, device=self.device)
            else:
                sam = sam_model_registry[model_type](checkpoint=checkpoint)
                sam.to(self.device)
            self.predictor = SamPredictor(sam)
        except Exception:
            self.predictor = None

    def set_image(self, image: np.ndarray):
        if self.predictor is not None:
            self.predictor.set_image(image)

    def predict_center_point(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        center_point = np.array([[w // 2, h // 2]])
        if self.predictor is None:
            mask = np.ones((h, w), dtype=np.uint8)
            return mask
        masks, scores, _ = self.predictor.predict(
            point_coords=center_point,
            point_labels=np.array([1]),
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        return masks[best_idx].astype(np.uint8)

    def predict_bounding_box(
        self, image: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        if self.predictor is None:
            h, w = image.shape[:2]
            return np.ones((h, w), dtype=np.uint8)
        box_array = np.array(bbox)
        masks, scores, _ = self.predictor.predict(
            box=box_array, multimask_output=True
        )
        best_idx = np.argmax(scores)
        return masks[best_idx].astype(np.uint8)

    def isolate(self, image: np.ndarray, bbox: Optional[tuple] = None) -> np.ndarray:
        self.set_image(image)
        if bbox is not None:
            mask = self.predict_bounding_box(image, bbox)
        else:
            mask = self.predict_center_point(image)
        isolated = image.copy()
        isolated[mask == 0] = [0, 0, 0]
        return isolated

    def isolate_to_transparent(self, image: np.ndarray, bbox: Optional[tuple] = None) -> np.ndarray:
        self.set_image(image)
        if bbox is not None:
            mask = self.predict_bounding_box(image, bbox)
        else:
            mask = self.predict_center_point(image)
        h, w = image.shape[:2]
        result = np.zeros((h, w, 4), dtype=np.uint8)
        result[:, :, :3] = image
        result[:, :, 3] = mask * 255
        return result

    def isolate_file(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        bbox: Optional[tuple] = None,
        transparent: bool = False,
    ) -> str:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        if transparent:
            result = self.isolate_to_transparent(image, bbox)
            ext = ".png"
        else:
            result = self.isolate(image, bbox)
            ext = ".jpg"
        if output_path is None:
            p = Path(image_path)
            output_path = str(p.parent / f"{p.stem}_isolated{ext}")
        cv2.imwrite(output_path, result)
        return output_path
