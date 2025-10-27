"""
Neural network models for training experiments.

Includes basic CNN architectures for image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SimpleCNN(nn.Module):
    """
    Simple CNN for MNIST classification.

    Architecture:
    - Conv1: 1 -> 32 channels, 3x3 kernel
    - Conv2: 32 -> 64 channels, 3x3 kernel
    - FC1: 9216 -> 128
    - FC2: 128 -> 10
    """

    def __init__(self, num_classes: int = 10):
        """
        Initialize SimpleCNN.

        Args:
            num_classes: Number of output classes.
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CIFAR10CNN(nn.Module):
    """
    CNN for CIFAR-10 classification.

    Architecture:
    - Conv1: 3 -> 64 channels, 3x3 kernel
    - Conv2: 64 -> 128 channels, 3x3 kernel
    - Conv3: 128 -> 256 channels, 3x3 kernel
    - FC1: 4096 -> 512
    - FC2: 512 -> 10
    """

    def __init__(self, num_classes: int = 10):
        """
        Initialize CIFAR10CNN.

        Args:
            num_classes: Number of output classes.
        """
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class ResNetBlock(nn.Module):
    """Residual block for ResNet."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """Initialize ResNet block."""
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SimpleResNet(nn.Module):
    """
    Simplified ResNet for CIFAR-10.

    Good for demonstrating edge AI optimization techniques.
    """

    def __init__(self, num_classes: int = 10):
        """Initialize SimpleResNet."""
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        """Create a layer with two residual blocks."""
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        layers.append(ResNetBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    Factory function to get model by name.

    Args:
        model_name: Name of the model ('simple_cnn', 'cifar10_cnn', 'simple_resnet').
        num_classes: Number of output classes.

    Returns:
        PyTorch model.

    Raises:
        ValueError: If model name is not recognized.
    """
    models = {
        "simple_cnn": SimpleCNN,
        "cifar10_cnn": CIFAR10CNN,
        "simple_resnet": SimpleResNet,
    }

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(models.keys())}"
        )

    return models[model_name](num_classes=num_classes)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Tuple of (total_params, trainable_params).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
