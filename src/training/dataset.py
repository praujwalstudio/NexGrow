import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from pathlib import Path
from typing import Optional
import numpy as np
from collections import Counter
import cv2


DISEASE_CLASSES = [
    "Bacterial_Blight",
    "CLCuD",
    "Alternaria_Leaf_Spot",
    "Grey_Mildew",
    "Leaf_Curl",
    "Leaf_Spot",
    "Red_Rust",
    "Target_Spot",
    "Bacterial_Light",
    "Healthy",
    "Fusarium_Wilt",
    "Verticillium_Wilt",
    "Anthracnose",
    "Cercospora_Leaf_Spot",
    "Root_Rot",
]

SEVERITY_LEVELS = ["Low", "Moderate", "High"]


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
        transform: Optional[transforms.Compose] = None,
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
                        self.samples.append({
                            "path": str(img_path),
                            "disease_idx": disease_idx,
                            "severity_idx": severity_idx,
                        })

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_transform(self, image: np.ndarray) -> torch.Tensor:
        if self.transform is None:
            image = transforms.ToTensor()(image)
            return transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )(image)
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
    """Apply ADASYN oversampling to balance disease class distribution."""
    from imblearn.over_sampling import ADASYN

    labels = [s["disease_idx"] for s in dataset.samples]
    X_dummy = np.zeros((len(labels), 1))

    adasyn = ADASYN(random_state=42)
    X_res, y_res = adasyn.fit_resample(X_dummy, labels)

    resampled = []
    label_to_paths = {}
    for s in dataset.samples:
        label_to_paths.setdefault(s["disease_idx"], []).append(s)

    for y in y_res:
        candidates = label_to_paths.get(y, [])
        if candidates:
            resampled.append(np.random.choice(candidates))

    dataset.samples = resampled
    return dataset


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = CottonLeafDataset(train_dir, transform=train_transform)
    val_dataset = CottonLeafDataset(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
