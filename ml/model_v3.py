"""
Improved CNN model for sudoku digit recognition (v3).

Architecture improvements over v1:
- Batch normalization for faster convergence
- Residual connections for better gradient flow
- Spatial dropout for regularization
- Squeeze-and-excitation blocks for channel attention
- Temperature scaling for calibrated confidence

Designed to stay under 500K parameters for mobile deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """Residual block with batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SpatialDropout2d(nn.Module):
    """Spatial dropout that drops entire feature maps."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        # Shape: (batch, channels, h, w)
        mask = torch.bernoulli(torch.ones(x.shape[0], x.shape[1], 1, 1, device=x.device) * (1 - self.p))
        return x * mask / (1 - self.p)


class DigitCNNv3(nn.Module):
    """Improved CNN for digit classification with residual connections.

    Architecture:
        Input: 28x28 grayscale
        Conv 3x3, 32 filters -> BN -> ReLU
        ResBlock(32, 32)
        ResBlock(32, 64, stride=2) -> 14x14
        ResBlock(64, 64)
        ResBlock(64, 128, stride=2) -> 7x7
        ResBlock(128, 128)
        Global Average Pool -> 128
        Dropout(0.5)
        FC(128, 10)

    Parameters: ~280K
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.5,
        use_se: bool = True,
    ):
        super().__init__()

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.layer1 = ResidualBlock(32, 32, stride=1, use_se=use_se)
        self.layer2 = ResidualBlock(32, 64, stride=2, use_se=use_se)  # 14x14
        self.layer3 = ResidualBlock(64, 64, stride=1, use_se=use_se)
        self.layer4 = ResidualBlock(64, 128, stride=2, use_se=use_se)  # 7x7
        self.layer5 = ResidualBlock(128, 128, stride=1, use_se=use_se)

        # Spatial dropout
        self.spatial_dropout = SpatialDropout2d(p=0.1)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

        # Temperature for calibration (learned or set post-training)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.spatial_dropout(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.spatial_dropout(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Global average pool
        features = self.gap(x).flatten(1)  # (batch, 128)

        if return_features:
            return features

        # Classifier
        x = self.dropout(features)
        logits = self.fc(x)

        return logits

    def forward_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with MC Dropout for uncertainty estimation.

        Args:
            x: Input tensor
            n_samples: Number of forward passes with dropout

        Returns:
            Tuple of (mean_probs, std_probs, predicted_class)
        """
        self.train()  # Enable dropout

        probs_list = []
        for _ in range(n_samples):
            logits = self.forward(x)
            probs = F.softmax(logits / self.temperature, dim=1)
            probs_list.append(probs)

        probs_stack = torch.stack(probs_list, dim=0)  # (n_samples, batch, classes)
        mean_probs = probs_stack.mean(dim=0)
        std_probs = probs_stack.std(dim=0)
        predicted = mean_probs.argmax(dim=1)

        self.eval()
        return mean_probs, std_probs, predicted

    def get_confidence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with calibrated confidence scores.

        Returns:
            Tuple of (predicted_class, confidence)
        """
        logits = self.forward(x)
        probs = F.softmax(logits / self.temperature, dim=1)
        confidence, predicted = probs.max(dim=1)
        return predicted, confidence

    def set_temperature(self, temperature: float):
        """Set temperature for probability calibration."""
        self.temperature.data.fill_(temperature)


class DigitCNNv3Light(nn.Module):
    """Lighter version for faster inference (~150K parameters).

    Architecture:
        Input: 28x28 grayscale
        Conv 3x3, 24 filters -> BN -> ReLU -> MaxPool
        Conv 3x3, 48 filters -> BN -> ReLU -> MaxPool
        Conv 3x3, 96 filters -> BN -> ReLU
        Global Average Pool -> 96
        Dropout(0.5)
        FC(96, 10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 14x14 -> 7x7
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 7x7 -> 7x7
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(96, num_classes)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def get_confidence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(x)
        probs = F.softmax(logits / self.temperature, dim=1)
        confidence, predicted = probs.max(dim=1)
        return predicted, confidence


class EmptyClassifier(nn.Module):
    """Simple binary classifier for empty vs non-empty cells.

    Much simpler than full digit recognition - can be used as
    a first stage to filter out empty cells.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def is_empty(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns boolean tensor indicating if cells are empty."""
        logits = self.forward(x)
        return torch.sigmoid(logits) < threshold


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calibrate_temperature(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    lr: float = 0.01,
    max_iter: int = 50,
) -> float:
    """Learn optimal temperature for probability calibration.

    Uses the validation set to find a temperature that minimizes
    negative log likelihood on held-out data.
    """
    model.eval()

    # Collect all logits and labels
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            logits_list.append(logits)
            labels_list.append(target)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    # Optimize temperature
    temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    def eval_loss():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)

    optimal_temp = temperature.item()
    print(f"Calibrated temperature: {optimal_temp:.4f}")

    return optimal_temp


if __name__ == "__main__":
    # Test all models
    print("Testing model architectures...")
    print()

    models = [
        ("DigitCNNv3", DigitCNNv3()),
        ("DigitCNNv3Light", DigitCNNv3Light()),
        ("EmptyClassifier", EmptyClassifier()),
    ]

    dummy = torch.randn(4, 1, 28, 28)

    for name, model in models:
        params = count_parameters(model)
        out = model(dummy)
        print(f"{name}:")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {out.shape}")
        print()

    # Test uncertainty estimation
    print("Testing MC Dropout uncertainty...")
    model = DigitCNNv3()
    mean_probs, std_probs, predicted = model.forward_with_uncertainty(dummy, n_samples=5)
    print(f"  Mean probs shape: {mean_probs.shape}")
    print(f"  Std probs shape: {std_probs.shape}")
    print(f"  Predicted: {predicted}")
