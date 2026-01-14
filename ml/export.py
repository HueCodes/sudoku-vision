"""
Export trained model to ONNX format for web deployment.

Usage:
    uv run python export.py --checkpoint digit_cnn.pt --output digit_cnn.onnx
"""

import argparse
from pathlib import Path

import torch
import torch.onnx

from model import DigitCNN


def export_to_onnx(checkpoint_path: Path, output_path: Path) -> None:
    """Export PyTorch model to ONNX."""
    # Load model
    model = DigitCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # Dummy input for tracing
    dummy_input = torch.randn(1, 1, 28, 28)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"Exported to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


def verify_onnx(onnx_path: Path) -> None:
    """Verify ONNX model is valid."""
    import onnx

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX model verification passed")


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("digit_cnn.onnx"))
    parser.add_argument("--verify", action="store_true", help="Verify exported model")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return 1

    export_to_onnx(args.checkpoint, args.output)

    if args.verify:
        verify_onnx(args.output)

    return 0


if __name__ == "__main__":
    exit(main())
