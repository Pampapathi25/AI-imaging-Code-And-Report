from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .config import IMAGE_SIZE


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_grayscale(path: Path, size=IMAGE_SIZE) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(path: Path, size=IMAGE_SIZE) -> np.ndarray:
    mask = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    return (np.asarray(mask) > 127).astype(np.float32)


def image_paths(data_dir: Path, split: str):
    return sorted((data_dir / split / "images").glob("*.png"))


class NucleiDataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        self.images = image_paths(data_dir, split)
        self.mask_dir = data_dir / split / "masks"
        if not self.images:
            raise FileNotFoundError(f"No images found for split={split} in {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.mask_dir / image_path.name
        image = load_grayscale(image_path)
        mask = load_mask(mask_path)
        return (
            torch.from_numpy(image).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
            image_path.stem,
        )
