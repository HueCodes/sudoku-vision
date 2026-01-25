"""
Improved training script with augmentation and better practices.

Improvements over v1:
- Training-time augmentation with torchvision/albumentations
- Learning rate warmup and cosine annealing
- Label smoothing
- Mixup augmentation
- Early stopping
- Better checkpointing
- Comprehensive logging

Usage:
    python train_v2.py --model v3 --dataset combined --epochs 30
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from model import DigitCNN, count_parameters
from model_v3 import DigitCNNv3, DigitCNNv3Light, calibrate_temperature


# =============================================================================
# Augmentation
# =============================================================================

class TrainingAugmentation:
    """Training-time augmentation pipeline."""

    def __init__(self, intensity: str = "medium"):
        if intensity == "light":
            self.transform = transforms.Compose([
                transforms.RandomRotation(5),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
            ])
        elif intensity == "medium":
            self.transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=5,
                ),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                ], p=0.3),
                transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ])
        else:  # heavy
            self.transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.15, 0.15),
                    scale=(0.85, 1.15),
                    shear=10,
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                ], p=0.5),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            ])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)


class NormalizeTransform:
    """Normalize to [-1, 1] range."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - 0.5) / 0.5


# =============================================================================
# Mixup augmentation
# =============================================================================

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# Label smoothing
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        log_probs = torch.log_softmax(pred, dim=-1)

        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(pred)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        loss = (-smooth_targets * log_probs).sum(dim=-1).mean()
        return loss


# =============================================================================
# Learning rate scheduling
# =============================================================================

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            factor = self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1 + np.cos(np.pi * progress))

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = max(self.min_lr, self.base_lrs[i] * factor)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


# =============================================================================
# Early stopping
# =============================================================================

class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.should_stop


# =============================================================================
# Dataset with augmentation
# =============================================================================

class AugmentedDataset(Dataset):
    """Wrapper dataset that applies augmentation."""

    def __init__(
        self,
        base_dataset: Dataset,
        augmentation: Optional[TrainingAugmentation] = None,
        normalize: bool = True,
    ):
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.normalize = NormalizeTransform() if normalize else None

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]

        # Apply augmentation
        if self.augmentation is not None:
            x = self.augmentation(x)

        # Normalize
        if self.normalize is not None:
            x = self.normalize(x)

        return x, y


# =============================================================================
# Training functions
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        if use_mixup:
            data, target_a, target_b, lam = mixup_data(data, target, mixup_alpha)
            output = model(data)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            output = model(data)
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, dict]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = [0] * 10
    per_class_total = [0] * 10

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)

            for p, t in zip(pred.cpu().numpy(), target.cpu().numpy()):
                per_class_total[t] += 1
                if p == t:
                    correct += 1
                    per_class_correct[t] += 1
                total += 1

    per_class_acc = {
        i: per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0
        for i in range(10)
    }

    return total_loss / len(loader), correct / total if total > 0 else 0, per_class_acc


# =============================================================================
# Data loading
# =============================================================================

def get_mnist_loaders(
    batch_size: int,
    data_dir: Path,
    augmentation_intensity: str = "medium",
) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST with augmentation."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # Wrap with augmentation
    augmentation = TrainingAugmentation(augmentation_intensity)
    train_dataset = AugmentedDataset(train_dataset, augmentation)
    test_dataset = AugmentedDataset(test_dataset, None)  # No aug for validation

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


# =============================================================================
# Main training
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train digit recognition model (v2)")

    # Model
    parser.add_argument("--model", choices=["v1", "v3", "v3light"], default="v3")

    # Data
    parser.add_argument("--dataset", choices=["mnist", "synthetic", "combined", "real"], default="mnist")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent.parent / "data")
    parser.add_argument("--augmentation", choices=["light", "medium", "heavy"], default="medium")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)

    # Techniques
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--no-mixup", action="store_true")

    # Early stopping
    parser.add_argument("--patience", type=int, default=10)

    # Output
    parser.add_argument("--output", type=Path, default=Path("digit_cnn_v3.pt"))
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibrate", action="store_true", help="Calibrate temperature after training")

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "mnist":
        train_loader, val_loader = get_mnist_loaders(
            args.batch_size, args.data_dir, args.augmentation
        )
    else:
        # Import from original train.py for other datasets
        from train import get_synthetic_loaders, get_combined_loaders, get_real_loaders
        if args.dataset == "synthetic":
            train_loader, val_loader = get_synthetic_loaders(args.batch_size, args.data_dir)
        elif args.dataset == "combined":
            train_loader, val_loader = get_combined_loaders(args.batch_size, args.data_dir)
        else:
            train_loader, val_loader = get_real_loaders(args.batch_size, args.data_dir)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Model
    if args.model == "v1":
        model = DigitCNN()
    elif args.model == "v3":
        model = DigitCNNv3()
    else:
        model = DigitCNNv3Light()

    model = model.to(device)

    # Load pretrained
    if args.pretrained and args.pretrained.exists():
        print(f"Loading pretrained: {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device, weights_only=True))

    print(f"Model: {args.model}, Parameters: {count_parameters(model):,}")

    # Loss and optimizer
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    # Logging
    args.log_dir.mkdir(exist_ok=True)
    log_file = args.log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    history = {
        "args": vars(args),
        "epochs": [],
    }

    # Training loop
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_mixup=not args.no_mixup,
            mixup_alpha=args.mixup_alpha,
        )

        # Evaluate
        val_loss, val_acc, per_class = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_lr(),
            "per_class_acc": per_class,
        }
        history["epochs"].append(epoch_data)

        # Print
        empty_acc = per_class.get(0, 0)
        digit_acc = np.mean([per_class.get(d, 0) for d in range(1, 10)])

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f}, {val_acc:.4f} | "
              f"Empty: {empty_acc:.3f}, Digits: {digit_acc:.3f} | "
              f"LR: {scheduler.get_lr():.2e}")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  -> Saved best model (acc={val_acc:.4f})")

        # Early stopping
        if early_stopping(val_acc):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(torch.load(args.output, map_location=device, weights_only=True))

    # Calibrate temperature
    if args.calibrate and hasattr(model, 'set_temperature'):
        print("\nCalibrating temperature...")
        temp = calibrate_temperature(model, val_loader, device)
        model.set_temperature(temp)
        torch.save(model.state_dict(), args.output)
        history["calibrated_temperature"] = temp

    # Save log
    history["best_acc"] = best_acc
    with open(log_file, "w") as f:
        # Convert Path objects to strings and numpy to lists
        def convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            return obj

        json.dump(history, f, indent=2, default=convert)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Model saved to: {args.output}")
    print(f"Log saved to: {log_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
