from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.config import DISEASE_CLASSES, SEVERITY_LEVELS


class CottonLeafDataset(Dataset):
    """PyTorch dataset for SAM-isolated cotton leaf images.

    Expected directory layout:
        root/
            <disease_class>/
                <severity>/
                    *.png or *.jpg
    """

    def __init__(
        self,
        root: str,
        transform: transforms.Compose | None = None,
        multi_task: bool = True,
    ):
        self.root = Path(root)
        self.transform = transform
        self.multi_task = multi_task
        self.samples = []
        self._build_index()

    def _build_index(self):
        for disease_idx, disease_name in enumerate(DISEASE_CLASSES):
            for severity_idx, severity in enumerate(SEVERITY_LEVELS):
                disease_dir = self.root / disease_name / severity
                if not disease_dir.exists():
                    continue
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                    for img_path in disease_dir.rglob(ext):
                        self.samples.append(
                            {
                                "path": str(img_path),
                                "disease_idx": disease_idx,
                                "severity_idx": severity_idx,
                            }
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_transform(self, image: np.ndarray) -> torch.Tensor:
        if self.transform is None:
            image = transforms.ToTensor()(image)
            return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        module = type(self.transform).__module__
        if module.startswith("albumentations"):
            return self.transform(image=image)["image"]
        return self.transform(image)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = cv2.imread(sample["path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self._apply_transform(image)

        result = {
            "image": image,
            "disease": torch.tensor(sample["disease_idx"], dtype=torch.long),
            "path": sample["path"],
        }

        if self.multi_task:
            result["severity"] = torch.tensor(sample["severity_idx"], dtype=torch.long)

        return result

    def get_class_weights(self) -> torch.Tensor:
        counts = Counter(s["disease_idx"] for s in self.samples)
        total = sum(counts.values())
        weights = torch.zeros(len(DISEASE_CLASSES))
        for idx in range(len(DISEASE_CLASSES)):
            weights[idx] = total / (len(DISEASE_CLASSES) * counts.get(idx, 1))
        return weights


def apply_adasyn(dataset: CottonLeafDataset) -> CottonLeafDataset:
    """Apply ADASYN oversampling to balance disease class distribution.

    Falls back to random oversampling when a class is too small for ADASYN's
    k-neighbor sampling, and returns the dataset unchanged when it cannot
    oversample safely (empty or single-class data).
    """
    from imblearn.over_sampling import ADASYN, RandomOverSampler

    if len(dataset.samples) == 0:
        return dataset

    labels = [s["disease_idx"] for s in dataset.samples]
    if len(set(labels)) < 2:
        return dataset

    X_dummy = np.zeros((len(labels), 1))
    label_to_samples = {}
    for s in dataset.samples:
        label_to_samples.setdefault(s["disease_idx"], []).append(s)

    try:
        _, y_res = ADASYN(random_state=42).fit_resample(X_dummy, labels)
    except Exception:
        _, y_res = RandomOverSampler(random_state=42).fit_resample(X_dummy, labels)

    resampled = []
    for y in y_res:
        candidates = label_to_samples.get(y, [])
        if candidates:
            resampled.append(np.random.choice(candidates))

    if len(resampled) < len(dataset.samples):
        return dataset
    dataset.samples = resampled
    return dataset


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = CottonLeafDataset(train_dir, transform=train_transform)
    val_dataset = CottonLeafDataset(val_dir, transform=val_transform)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
