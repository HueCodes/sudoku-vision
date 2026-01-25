#!/usr/bin/env python3
"""
Comprehensive model evaluation suite.

Features:
- Multi-dataset evaluation (MNIST, synthetic, real)
- Per-class metrics (precision, recall, F1)
- Confusion matrix visualization
- Reliability diagram (calibration plot)
- Failure case analysis
- Confidence distribution analysis

Usage:
    python evaluate_v2.py --model digit_cnn_v3.pt --data-dir data/
    python evaluate_v2.py --model digit_cnn_v3.pt --save-failures failures/
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_model(model_path: Path, device: torch.device):
    """Load model from checkpoint."""
    # Try different model versions
    from model import DigitCNN
    from model_v3 import DigitCNNv3, DigitCNNv3Light

    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Infer model type from state dict
    if 'layer1.conv1.weight' in state_dict:
        if state_dict['layer1.conv1.weight'].shape[0] == 32:
            model = DigitCNNv3()
        else:
            model = DigitCNNv3Light()
    else:
        model = DigitCNN()

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def evaluate_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset_name: str = "test",
) -> Dict:
    """Evaluate model on a dataset.

    Returns dictionary with all metrics.
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_correct = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_correct.extend((pred == target).cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_correct = np.array(all_correct)

    # Overall accuracy
    accuracy = np.mean(all_correct)

    # Per-class metrics
    per_class = {}
    for cls in range(10):
        mask = all_labels == cls
        if mask.sum() == 0:
            continue

        true_pos = ((all_preds == cls) & (all_labels == cls)).sum()
        false_pos = ((all_preds == cls) & (all_labels != cls)).sum()
        false_neg = ((all_preds != cls) & (all_labels == cls)).sum()

        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class[cls] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(mask.sum()),
            'accuracy': float(all_correct[mask].mean()),
        }

    # Confidence statistics
    max_probs = all_probs.max(axis=1)
    correct_conf = max_probs[all_correct].mean() if all_correct.sum() > 0 else 0
    incorrect_conf = max_probs[~all_correct].mean() if (~all_correct).sum() > 0 else 0

    # Calibration data (for reliability diagram)
    calibration_data = compute_calibration(all_probs, all_labels, n_bins=10)

    return {
        'dataset': dataset_name,
        'samples': len(all_labels),
        'accuracy': float(accuracy),
        'per_class': per_class,
        'mean_confidence': float(max_probs.mean()),
        'correct_confidence': float(correct_conf),
        'incorrect_confidence': float(incorrect_conf),
        'calibration': calibration_data,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
    }


def compute_calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict:
    """Compute calibration metrics for reliability diagram."""
    max_probs = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (max_probs > bin_boundaries[i]) & (max_probs <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_confidences.append(max_probs[in_bin].mean())
            bin_counts.append(int(in_bin.sum()))

    # Expected Calibration Error
    ece = 0
    for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
        ece += abs(acc - conf) * count / len(labels)

    return {
        'bin_centers': bin_centers,
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
        'ece': float(ece),
    }


def find_failures(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_failures: int = 100,
) -> List[Dict]:
    """Find misclassified samples for analysis."""
    model.eval()
    failures = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)

            # Find failures in this batch
            wrong = pred != target
            for i in wrong.nonzero().flatten():
                if len(failures) >= max_failures:
                    return failures

                idx = i.item()
                failures.append({
                    'batch_idx': batch_idx,
                    'sample_idx': idx,
                    'true_label': int(target[idx].item()),
                    'predicted': int(pred[idx].item()),
                    'confidence': float(probs[idx, pred[idx]].item()),
                    'true_prob': float(probs[idx, target[idx]].item()),
                    'top3_preds': probs[idx].topk(3).indices.cpu().numpy().tolist(),
                    'top3_probs': probs[idx].topk(3).values.cpu().numpy().tolist(),
                    'image': data[idx].cpu().numpy(),
                })

    return failures


def plot_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, output_path: Path):
    """Plot and save confusion matrix."""
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        print("Skipping confusion matrix (matplotlib/sklearn not available)")
        return

    cm = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix')
    fig.colorbar(im, ax=ax)

    classes = ['empty'] + [str(i) for i in range(1, 10)]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_reliability_diagram(calibration_data: Dict, output_path: Path):
    """Plot reliability diagram (calibration plot)."""
    if not HAS_MATPLOTLIB:
        print("Skipping reliability diagram (matplotlib not available)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.bar(calibration_data['bin_centers'], calibration_data['bin_accuracies'],
           width=0.08, alpha=0.7, label='Model')
    ax1.set_xlabel('Mean Predicted Confidence')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(f"Reliability Diagram (ECE={calibration_data['ece']:.3f})")
    ax1.legend()
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Confidence histogram
    ax2.bar(calibration_data['bin_centers'], calibration_data['bin_counts'],
           width=0.08, alpha=0.7)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved reliability diagram to {output_path}")


def plot_failures(failures: List[Dict], output_path: Path, n_show: int = 25):
    """Plot grid of failure cases."""
    if not HAS_MATPLOTLIB:
        print("Skipping failure plot (matplotlib not available)")
        return

    n_show = min(n_show, len(failures))
    if n_show == 0:
        print("No failures to plot")
        return

    grid_size = int(np.ceil(np.sqrt(n_show)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i, failure in enumerate(failures[:n_show]):
        ax = axes[i]
        img = failure['image'].squeeze()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"T:{failure['true_label']} P:{failure['predicted']}\n"
                    f"conf:{failure['confidence']:.2f}", fontsize=8)
        ax.axis('off')

    # Hide unused axes
    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Failure Cases (True label vs Predicted)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved failure cases to {output_path}")


def print_report(results: Dict):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT: {results['dataset']}")
    print("=" * 60)

    print(f"\nSamples: {results['samples']}")
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {results['mean_confidence']:.4f}")
    print(f"  Correct samples: {results['correct_confidence']:.4f}")
    print(f"  Incorrect samples: {results['incorrect_confidence']:.4f}")
    print(f"  ECE (calibration error): {results['calibration']['ece']:.4f}")

    print("\nPer-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':>6} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)

    for cls in range(10):
        if cls in results['per_class']:
            m = results['per_class'][cls]
            label = 'empty' if cls == 0 else str(cls)
            print(f"{label:>6} {m['precision']:>10.3f} {m['recall']:>10.3f} "
                  f"{m['f1']:>10.3f} {m['support']:>10}")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation")
    parser.add_argument("--model", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--save-failures", action="store_true", help="Save failure cases")
    parser.add_argument("--max-failures", type=int, default=100)
    parser.add_argument("--skip-mnist", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.output_dir.mkdir(exist_ok=True)

    # Load model
    print(f"Loading model from {args.model}")
    model = load_model(args.model, device)

    all_results = {}

    # Evaluate on MNIST
    if not args.skip_mnist:
        print("\nEvaluating on MNIST test set...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        mnist_test = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
        mnist_loader = DataLoader(mnist_test, batch_size=args.batch_size, shuffle=False)

        results = evaluate_dataset(model, mnist_loader, device, "MNIST")
        print_report(results)
        all_results['mnist'] = results

        # Plot confusion matrix
        plot_confusion_matrix(
            results['labels'],
            results['predictions'],
            args.output_dir / 'confusion_mnist.png'
        )

        # Plot reliability diagram
        plot_reliability_diagram(
            results['calibration'],
            args.output_dir / 'reliability_mnist.png'
        )

        # Find and plot failures
        if args.save_failures:
            failures = find_failures(model, mnist_loader, device, args.max_failures)
            plot_failures(failures, args.output_dir / 'failures_mnist.png')

    # Evaluate on synthetic (if available)
    if not args.skip_synthetic and (args.data_dir / "synthetic" / "val").exists():
        print("\nEvaluating on synthetic validation set...")
        try:
            from datasets import SyntheticDataset
            transform = transforms.Normalize((0.5,), (0.5,))
            synthetic_test = SyntheticDataset(args.data_dir / "synthetic", split="val", transform=transform)
            synthetic_loader = DataLoader(synthetic_test, batch_size=args.batch_size, shuffle=False)

            results = evaluate_dataset(model, synthetic_loader, device, "Synthetic")
            print_report(results)
            all_results['synthetic'] = results

            plot_confusion_matrix(
                results['labels'],
                results['predictions'],
                args.output_dir / 'confusion_synthetic.png'
            )
        except ImportError:
            print("Could not import SyntheticDataset, skipping synthetic evaluation")

    # Evaluate on real data (if available)
    real_data_dir = args.data_dir / "labeled" / "test"
    if real_data_dir.exists():
        print(f"\nEvaluating on real test set ({real_data_dir})...")
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        real_test = datasets.ImageFolder(real_data_dir, transform=transform)
        real_loader = DataLoader(real_test, batch_size=args.batch_size, shuffle=False)

        results = evaluate_dataset(model, real_loader, device, "Real")
        print_report(results)
        all_results['real'] = results

        plot_confusion_matrix(
            results['labels'],
            results['predictions'],
            args.output_dir / 'confusion_real.png'
        )

        plot_reliability_diagram(
            results['calibration'],
            args.output_dir / 'reliability_real.png'
        )

        if args.save_failures:
            failures = find_failures(model, real_loader, device, args.max_failures)
            plot_failures(failures, args.output_dir / 'failures_real.png')

    # Save summary
    summary = {
        'model': str(args.model),
        'results': {
            name: {
                'accuracy': r['accuracy'],
                'ece': r['calibration']['ece'],
                'samples': r['samples'],
            }
            for name, r in all_results.items()
        }
    }

    with open(args.output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")
    print("\nSummary:")
    for name, r in summary['results'].items():
        print(f"  {name}: acc={r['accuracy']:.4f}, ECE={r['ece']:.4f}, n={r['samples']}")


if __name__ == "__main__":
    main()
