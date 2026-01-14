"""
CNN model for sudoku digit recognition.

Architecture from ROADMAP.md:
  Input: 28x28 grayscale image
  Conv2D(32, 3x3, ReLU) -> MaxPool(2x2)
  Conv2D(64, 3x3, ReLU) -> MaxPool(2x2)
  Flatten -> Dense(128, ReLU) -> Dropout(0.5)
  Dense(10, Softmax) -> Output: digit 0-9

0 = empty cell, 1-9 = digits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """Simple CNN for digit classification (0-9, where 0 = empty)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # After 2 pools: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # -> (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (batch, 64, 7, 7)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # logits, use CrossEntropyLoss


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    model = DigitCNN()
    print(f"DigitCNN parameters: {count_parameters(model):,}")

    # Test forward pass
    dummy = torch.randn(4, 1, 28, 28)
    out = model(dummy)
    print(f"Input shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")  # should be (4, 10)
