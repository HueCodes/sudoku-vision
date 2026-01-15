"""
Convert PyTorch digit classifier to CoreML format.

Usage:
    uv run python convert_coreml.py --input digit_cnn.pt --output DigitClassifier.mlpackage
"""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from model import DigitCNN


def convert_to_coreml(checkpoint_path: Path, output_path: Path) -> None:
    """
    Convert PyTorch model to CoreML format.

    The model expects:
    - Input: 1x1x28x28 float32 tensor (grayscale image, normalized [0,1])
    - Output: 1x10 float32 tensor (class logits, NOT softmax)

    In the iOS app:
    - Class 0-9 represent digits 0-9 in MNIST terms
    - However, sudoku doesn't use digit 0
    - Empty cells are detected separately via pixel variance
    - For filled cells, we use argmax of classes 1-9
    """
    print(f"Loading PyTorch model from: {checkpoint_path}")

    # Load PyTorch model
    model = DigitCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()

    # Create example input for tracing
    example_input = torch.randn(1, 1, 28, 28)

    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(model, example_input)

    # Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="input",
                shape=(1, 1, 28, 28),
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(name="output", dtype=np.float32)
        ],
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",  # Use ML Program format for iOS 17+
    )

    # Add metadata for documentation
    mlmodel.author = "Sudoku Vision"
    mlmodel.license = "MIT"
    mlmodel.short_description = "Classifies handwritten digits 0-9 from 28x28 grayscale images"
    mlmodel.version = "1.0.0"

    # Input/output descriptions
    mlmodel.input_description["input"] = (
        "28x28 grayscale image as [1, 1, 28, 28] tensor. "
        "Pixels normalized to [0, 1] range. "
        "Image should be inverted (white digit on black background, MNIST-style)."
    )
    mlmodel.output_description["output"] = (
        "10-element logits vector for digits 0-9. "
        "Apply softmax for probabilities, then argmax for prediction. "
        "Note: Sudoku uses 1-9 only; class 0 from MNIST is the digit zero."
    )

    # Save the model
    mlmodel.save(str(output_path))
    print(f"Saved CoreML model to: {output_path}")

    # Print model info
    print(f"\nModel info:")
    print(f"  Format: ML Program")
    print(f"  Minimum iOS: 17.0")


def verify_coreml_model(model_path: Path) -> None:
    """
    Verify the CoreML model loads and runs correctly.
    """
    print(f"\nVerifying CoreML model...")

    # Load model
    model = ct.models.MLModel(str(model_path))

    # Create test input (random noise)
    test_input = {"input": np.random.randn(1, 1, 28, 28).astype(np.float32)}

    # Run inference
    output = model.predict(test_input)

    # Check output shape
    logits = output["output"]
    print(f"  Output shape: {logits.shape}")
    print(f"  Output dtype: {logits.dtype}")
    print(f"  Sample logits: [{', '.join(f'{x:.2f}' for x in logits.flatten()[:5])}...]")

    # Verify softmax behavior
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    print(f"  After softmax, sum: {probs.sum():.4f} (should be 1.0)")
    print(f"  Predicted class: {np.argmax(logits)}")

    print("  Verification passed!")


def test_with_mnist_sample(model_path: Path) -> None:
    """
    Test the CoreML model with a real MNIST sample.
    """
    try:
        from torchvision import datasets, transforms
        print(f"\nTesting with MNIST sample...")

        # Load a test image
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(
            root="../data",
            train=False,
            download=True,
            transform=transform
        )

        # Load CoreML model
        model = ct.models.MLModel(str(model_path))

        # Test first 10 samples
        correct = 0
        for i in range(10):
            img, label = test_dataset[i]
            img_np = img.numpy().reshape(1, 1, 28, 28).astype(np.float32)

            output = model.predict({"input": img_np})
            pred = np.argmax(output["output"])

            status = "✓" if pred == label else "✗"
            print(f"  Sample {i}: label={label}, pred={pred} {status}")

            if pred == label:
                correct += 1

        print(f"  Accuracy: {correct}/10")

    except ImportError:
        print("  Skipping MNIST test (torchvision not available)")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to CoreML")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("digit_cnn.pt"),
        help="Input PyTorch checkpoint path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("DigitClassifier.mlpackage"),
        help="Output CoreML model path"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification step"
    )
    parser.add_argument(
        "--test-mnist",
        action="store_true",
        help="Test with MNIST samples after conversion"
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Convert
    convert_to_coreml(args.input, args.output)

    # Verify
    if not args.skip_verify:
        verify_coreml_model(args.output)

    # Optional MNIST test
    if args.test_mnist:
        test_with_mnist_sample(args.output)

    print(f"\nDone! Copy {args.output} to ios/SudokuVision/Resources/")
    return 0


if __name__ == "__main__":
    exit(main())
