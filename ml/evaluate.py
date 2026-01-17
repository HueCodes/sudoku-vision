#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluates the digit recognition model on:
1. MNIST test set (baseline)
2. Real extracted sudoku cells (if available)

Usage:
    python evaluate.py [--model PATH] [--real-data PATH] [--labels PATH]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from model import DigitCNN


class RealCellDataset(Dataset):
    """Dataset for real extracted sudoku cells."""

    def __init__(self, cells_dir: Path, labels_file: Path = None, transform=None):
        self.transform = transform
        self.samples = []

        # If cells_dir is actually the real/ directory, find all label files
        if labels_file is None:
            # Look for all labels_*.csv files
            label_files = list(cells_dir.glob("labels_*.csv"))
            for lf in label_files:
                sample_name = lf.stem.replace("labels_", "")
                sample_dir = cells_dir / sample_name
                if sample_dir.exists():
                    self._load_labels(lf, sample_dir)
        else:
            # Single label file mode (backward compatibility)
            self._load_labels(labels_file, cells_dir)

        print(f"Loaded {len(self.samples)} labeled samples")

    def _load_labels(self, labels_file: Path, cells_dir: Path):
        """Load labels from a CSV file."""
        with open(labels_file, "r") as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    filename, label = parts[0], int(parts[1])
                    filepath = cells_dir / filename
                    if filepath.exists():
                        self.samples.append((filepath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        # Load image
        img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load {filepath}")

        # Resize to 28x28 if needed
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28))

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        img = clahe.apply(img)

        # Apply adaptive thresholding for cleaner binary-like image
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Invert: real images are dark-on-light, we need white-on-black
        img = 255 - img

        # Convert to tensor and normalize
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # Add channel dim

        if self.transform:
            img = self.transform(img)

        return img, label


def evaluate_mnist(model: torch.nn.Module, device: torch.device) -> dict:
    """Evaluate on MNIST test set."""
    print("\n=== MNIST Test Set Evaluation ===")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_dataset = datasets.MNIST(
        root=Path(__file__).parent.parent / "data",
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    per_digit_correct = [0] * 10
    per_digit_total = [0] * 10
    confusion = np.zeros((10, 10), dtype=int)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            for p, t in zip(pred.cpu().numpy(), target.cpu().numpy()):
                confusion[t][p] += 1
                per_digit_total[t] += 1
                if p == t:
                    correct += 1
                    per_digit_correct[t] += 1
                total += 1

    accuracy = correct / total

    print(f"\nOverall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print("\nPer-Digit Accuracy:")
    for d in range(10):
        if per_digit_total[d] > 0:
            acc = per_digit_correct[d] / per_digit_total[d]
            print(f"  Digit {d}: {acc:.4f} ({per_digit_correct[d]}/{per_digit_total[d]})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_digit_accuracy": [
            per_digit_correct[d] / per_digit_total[d] if per_digit_total[d] > 0 else 0
            for d in range(10)
        ],
        "confusion_matrix": confusion,
    }


def evaluate_real(
    model: torch.nn.Module,
    device: torch.device,
    cells_dir: Path,
    labels_file: Path = None,
) -> dict:
    """Evaluate on real extracted cells."""
    print("\n=== Real Cells Evaluation ===")

    # Normalize to [-1, 1] to match synthetic training
    normalize = transforms.Normalize((0.5,), (0.5,))

    dataset = RealCellDataset(cells_dir, labels_file, transform=normalize)
    if len(dataset) == 0:
        print("No samples found!")
        return {}

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    per_digit_correct = [0] * 10
    per_digit_total = [0] * 10
    confusion = np.zeros((10, 10), dtype=int)

    predictions = []
    confidences = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            conf = probs[0, pred[0]].item()

            t = target.item()
            p = pred.item()

            predictions.append((t, p, conf))
            confidences.append(conf)
            confusion[t][p] += 1
            per_digit_total[t] += 1

            if p == t:
                correct += 1
                per_digit_correct[t] += 1
            total += 1

    accuracy = correct / total if total > 0 else 0

    print(f"\nOverall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Average Confidence: {np.mean(confidences):.4f}")

    print("\nPer-Digit Accuracy:")
    for d in range(10):
        if per_digit_total[d] > 0:
            acc = per_digit_correct[d] / per_digit_total[d]
            print(f"  Digit {d}: {acc:.4f} ({per_digit_correct[d]}/{per_digit_total[d]})")

    # Show confusion matrix
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print("     ", end="")
    for d in range(10):
        print(f"{d:4d}", end="")
    print()
    for t in range(10):
        if per_digit_total[t] > 0:
            print(f"  {t}: ", end="")
            for p in range(10):
                if confusion[t][p] > 0:
                    print(f"{confusion[t][p]:4d}", end="")
                else:
                    print("   .", end="")
            print()

    # Show misclassifications
    print("\nMisclassifications:")
    for i, (t, p, conf) in enumerate(predictions):
        if t != p:
            print(f"  Sample {i}: true={t}, pred={p}, conf={conf:.3f}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_digit_accuracy": [
            per_digit_correct[d] / per_digit_total[d] if per_digit_total[d] > 0 else 0
            for d in range(10)
        ],
        "confusion_matrix": confusion,
        "avg_confidence": np.mean(confidences),
    }


def print_summary(mnist_results: dict, real_results: dict) -> None:
    """Print summary comparison."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nMNIST Test Accuracy:  {mnist_results['accuracy']:.4f}")
    if real_results:
        print(f"Real Cells Accuracy:  {real_results['accuracy']:.4f}")
        print(f"Real Cells Confidence: {real_results['avg_confidence']:.4f}")

        # Identify problem digits
        print("\nProblem Digits (real accuracy < 80%):")
        for d in range(10):
            if real_results["per_digit_accuracy"][d] < 0.8:
                mnist_acc = mnist_results["per_digit_accuracy"][d]
                real_acc = real_results["per_digit_accuracy"][d]
                print(f"  Digit {d}: MNIST={mnist_acc:.2f}, Real={real_acc:.2f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate digit recognition model")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent / "digit_cnn.pt",
        help="Path to model weights",
    )
    parser.add_argument(
        "--real-data",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "real",
        help="Path to real data directory (contains sample_* dirs and labels_*.csv)",
    )
    parser.add_argument(
        "--skip-mnist",
        action="store_true",
        help="Skip MNIST evaluation",
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model}")
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    # Evaluate MNIST
    mnist_results = {}
    if not args.skip_mnist:
        mnist_results = evaluate_mnist(model, device)

    # Evaluate real data
    real_results = {}
    if args.real_data.exists():
        real_results = evaluate_real(model, device, args.real_data)
    else:
        print(f"\nNo real data found at {args.real_data}")

    # Summary
    if mnist_results:
        print_summary(mnist_results, real_results)


if __name__ == "__main__":
    main()
