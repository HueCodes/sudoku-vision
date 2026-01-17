"""
Training script for digit recognition CNN.

Supports training on:
- MNIST (handwritten digits, baseline)
- Synthetic (generated printed digits)
- Combined (synthetic + real extracted cells)

Usage:
    # Train on MNIST (baseline)
    python train.py --dataset mnist --epochs 10

    # Train on synthetic data
    python train.py --dataset synthetic --epochs 10

    # Train on combined data (synthetic + real, fine-tuned)
    python train.py --dataset combined --epochs 15 --output digit_cnn_v2.pt

    # Fine-tune existing model on real data
    python train.py --dataset real --pretrained digit_cnn.pt --lr 0.0001 --epochs 5
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitCNN, count_parameters
from datasets import SyntheticDataset, RealDataset, create_combined_dataset, get_balanced_sampler


def get_mnist_loaders(batch_size: int, data_dir: Path) -> tuple[DataLoader, DataLoader]:
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_synthetic_loaders(
    batch_size: int,
    data_dir: Path,
    use_balanced_sampler: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Load synthetic dataset."""
    # Normalize to [-1, 1] to match training range
    transform = transforms.Normalize((0.5,), (0.5,))

    train_dataset = SyntheticDataset(data_dir / "synthetic", split="train", transform=transform)
    val_dataset = SyntheticDataset(data_dir / "synthetic", split="val", transform=transform)

    if use_balanced_sampler:
        sampler = get_balanced_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_combined_loaders(
    batch_size: int,
    data_dir: Path,
    real_weight: float = 5.0,
) -> tuple[DataLoader, DataLoader]:
    """Load combined synthetic + real dataset."""
    transform = transforms.Normalize((0.5,), (0.5,))

    train_dataset = create_combined_dataset(
        data_dir / "synthetic",
        data_dir / "real",
        split="train",
        transform=transform,
        real_weight=real_weight,
    )

    # Use synthetic validation set
    val_dataset = SyntheticDataset(data_dir / "synthetic", split="val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_real_loaders(
    batch_size: int,
    data_dir: Path,
) -> tuple[DataLoader, DataLoader]:
    """Load real dataset only (for fine-tuning)."""
    transform = transforms.Normalize((0.5,), (0.5,))

    dataset = RealDataset(data_dir / "real", transform=transform)

    # Split 80/20 for train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
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
) -> tuple[float, float, dict]:
    """Evaluate model. Returns (loss, accuracy, per_class_acc)."""
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

    return total_loss / len(loader), correct / total, per_class_acc


def evaluate_on_real(
    model: nn.Module,
    data_dir: Path,
    device: torch.device,
) -> tuple[float, dict]:
    """Evaluate model on real data specifically."""
    transform = transforms.Normalize((0.5,), (0.5,))
    dataset = RealDataset(data_dir / "real", transform=transform)

    if len(dataset) == 0:
        return 0.0, {}

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    per_class_correct = [0] * 10
    per_class_total = [0] * 10

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
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

    return correct / total if total > 0 else 0.0, per_class_acc


def main():
    parser = argparse.ArgumentParser(description="Train digit recognition CNN")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "synthetic", "combined", "real"],
        default="synthetic",
        help="Dataset to train on",
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent.parent / "data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=Path("digit_cnn_v2.pt"))
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Path to pretrained model for fine-tuning",
    )
    parser.add_argument(
        "--real-weight",
        type=float,
        default=5.0,
        help="How many times to oversample real data in combined mode",
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")

    # Data loaders
    if args.dataset == "mnist":
        train_loader, val_loader = get_mnist_loaders(args.batch_size, args.data_dir)
    elif args.dataset == "synthetic":
        train_loader, val_loader = get_synthetic_loaders(args.batch_size, args.data_dir)
    elif args.dataset == "combined":
        train_loader, val_loader = get_combined_loaders(
            args.batch_size, args.data_dir, real_weight=args.real_weight
        )
    else:  # real
        train_loader, val_loader = get_real_loaders(args.batch_size, args.data_dir)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Model
    model = DigitCNN().to(device)

    # Load pretrained weights if specified
    if args.pretrained and args.pretrained.exists():
        print(f"Loading pretrained weights from {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device, weights_only=True))

    print(f"Model parameters: {count_parameters(model):,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # Training loop
    best_acc = 0.0
    best_real_acc = 0.0

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_per_class = evaluate(model, val_loader, criterion, device)

        # Also evaluate on real data
        real_acc, real_per_class = evaluate_on_real(model, args.data_dir, device)

        # Update scheduler
        scheduler.step(val_acc)

        # Print progress
        print(f"\nEpoch {epoch:3d}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}, Val acc: {val_acc:.4f}")
        print(f"  Real acc:   {real_acc:.4f}")

        # Per-class accuracy on real data
        if real_per_class:
            empty_acc = real_per_class.get(0, 0)
            digit_accs = [real_per_class.get(d, 0) for d in range(1, 10)]
            avg_digit_acc = sum(digit_accs) / len(digit_accs) if digit_accs else 0
            print(f"  Real empty acc: {empty_acc:.4f}, Real digit acc: {avg_digit_acc:.4f}")

        # Save best model based on real accuracy
        if real_acc > best_real_acc:
            best_real_acc = real_acc
            torch.save(model.state_dict(), args.output)
            print(f"  -> Saved best model (real_acc={real_acc:.4f})")
        elif val_acc > best_acc and real_acc >= best_real_acc * 0.9:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Best real data accuracy: {best_real_acc:.4f}")
    print(f"Model saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
