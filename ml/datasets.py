"""
Dataset classes for digit recognition training.

Supports:
- Synthetic data (generated printed digits)
- Real data (extracted from sudoku photos)
- Combined datasets for training
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler


def preprocess_cell(img: np.ndarray) -> np.ndarray:
    """Apply consistent preprocessing to cell images.

    This preprocessing must match what's applied during inference:
    1. CLAHE contrast enhancement
    2. Adaptive thresholding
    3. Invert to white-on-black
    """
    # Ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))

    # CLAHE: Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img = clahe.apply(img)

    # Adaptive thresholding for cleaner binary-like image
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Invert: white digit on black background
    img = 255 - img

    return img


class SyntheticDataset(Dataset):
    """Dataset for synthetic printed digits.

    Directory structure:
        root/
            train/
                0/, 1/, 2/, ..., 9/
            val/
                0/, 1/, 2/, ..., 9/
    """

    def __init__(self, root: Path, split: str = "train", transform=None):
        self.root = Path(root) / split
        self.transform = transform
        self.samples = []

        # Load all samples
        for label in range(10):
            class_dir = self.root / str(label)
            if class_dir.exists():
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((img_path, label))

        print(f"SyntheticDataset ({split}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load grayscale image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load {img_path}")

        # Apply consistent preprocessing (CLAHE + threshold + invert)
        img = preprocess_cell(img)

        # Convert to tensor: (1, 28, 28), values in [0, 1]
        tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label


class RealDataset(Dataset):
    """Dataset for real extracted sudoku cells.

    Directory structure:
        root/
            sample_3/
                cell_0_0.png, cell_0_1.png, ...
            sample_4/
                cell_0_0.png, cell_0_1.png, ...
        labels_sample_3.csv
        labels_sample_4.csv
    """

    def __init__(self, root: Path, samples: list[str] | None = None, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        # Find all label files
        if samples is None:
            label_files = list(self.root.glob("labels_*.csv"))
        else:
            label_files = [self.root / f"labels_{s}.csv" for s in samples]

        for label_file in label_files:
            if not label_file.exists():
                continue

            # Extract sample name from filename (e.g., labels_sample_3.csv -> sample_3)
            sample_name = label_file.stem.replace("labels_", "")
            sample_dir = self.root / sample_name

            if not sample_dir.exists():
                print(f"Warning: {sample_dir} not found, skipping")
                continue

            # Load labels
            with open(label_file, "r") as f:
                lines = f.readlines()[1:]  # Skip header
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        filename, label = parts[0], int(parts[1])
                        img_path = sample_dir / filename
                        if img_path.exists():
                            self.samples.append((img_path, label))

        print(f"RealDataset: {len(self.samples)} samples from {len(label_files)} sources")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load grayscale image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load {img_path}")

        # Apply consistent preprocessing (CLAHE + threshold + invert)
        img = preprocess_cell(img)

        # Convert to tensor: (1, 28, 28), values in [0, 1]
        tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label


def get_class_weights(dataset: Dataset, num_classes: int = 10) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets.

    Returns weights inversely proportional to class frequency.
    """
    counts = [0] * num_classes
    for _, label in dataset:
        counts[label] += 1

    total = sum(counts)
    weights = []
    for c in counts:
        if c > 0:
            weights.append(total / (num_classes * c))
        else:
            weights.append(0.0)

    return torch.tensor(weights, dtype=torch.float32)


def get_balanced_sampler(dataset: Dataset, num_classes: int = 10) -> WeightedRandomSampler:
    """Create a sampler that balances classes during training."""
    class_weights = get_class_weights(dataset, num_classes)

    # Assign weight to each sample based on its class
    sample_weights = []
    for _, label in dataset:
        sample_weights.append(class_weights[label].item())

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


def create_combined_dataset(
    synthetic_dir: Path,
    real_dir: Path,
    split: str = "train",
    transform=None,
    real_weight: float = 1.0,
) -> Dataset:
    """Create a combined dataset from synthetic and real data.

    Args:
        synthetic_dir: Path to synthetic data
        real_dir: Path to real data
        split: 'train' or 'val'
        transform: Optional transforms
        real_weight: How many times to repeat real data (for oversampling)

    Returns:
        Combined dataset
    """
    datasets = []

    # Synthetic data
    syn_path = Path(synthetic_dir)
    if syn_path.exists():
        datasets.append(SyntheticDataset(syn_path, split=split, transform=transform))

    # Real data (only for training, or if we have enough)
    if split == "train":
        real_path = Path(real_dir)
        if real_path.exists():
            real_ds = RealDataset(real_path, transform=transform)
            # Oversample real data
            for _ in range(int(real_weight)):
                datasets.append(real_ds)

    if not datasets:
        raise ValueError("No datasets found")

    return ConcatDataset(datasets)


if __name__ == "__main__":
    # Test datasets
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    data_dir = Path(__file__).parent.parent / "data"

    # Test synthetic
    print("\n=== Testing SyntheticDataset ===")
    syn_ds = SyntheticDataset(data_dir / "synthetic", split="train")
    print(f"Length: {len(syn_ds)}")
    img, label = syn_ds[0]
    print(f"Sample shape: {img.shape}, label: {label}")

    # Test real
    print("\n=== Testing RealDataset ===")
    real_ds = RealDataset(data_dir / "real")
    print(f"Length: {len(real_ds)}")
    if len(real_ds) > 0:
        img, label = real_ds[0]
        print(f"Sample shape: {img.shape}, label: {label}")

    # Test combined
    print("\n=== Testing Combined ===")
    combined = create_combined_dataset(
        data_dir / "synthetic",
        data_dir / "real",
        split="train",
    )
    print(f"Combined length: {len(combined)}")
