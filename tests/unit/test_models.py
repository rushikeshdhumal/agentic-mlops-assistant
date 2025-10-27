"""
Unit tests for neural network models.
"""

import pytest
import torch
from athena.mlops.models import (
    SimpleCNN,
    CIFAR10CNN,
    SimpleResNet,
    get_model,
    count_parameters,
)


class TestModels:
    """Tests for model architectures."""

    def test_simple_cnn_forward(self):
        """Test SimpleCNN forward pass."""
        model = SimpleCNN(num_classes=10)
        x = torch.randn(2, 1, 28, 28)  # Batch of 2 MNIST images
        output = model(x)
        assert output.shape == (2, 10)

    def test_cifar10_cnn_forward(self):
        """Test CIFAR10CNN forward pass."""
        model = CIFAR10CNN(num_classes=10)
        x = torch.randn(2, 3, 32, 32)  # Batch of 2 CIFAR-10 images
        output = model(x)
        assert output.shape == (2, 10)

    def test_simple_resnet_forward(self):
        """Test SimpleResNet forward pass."""
        model = SimpleResNet(num_classes=10)
        x = torch.randn(2, 3, 32, 32)  # Batch of 2 CIFAR-10 images
        output = model(x)
        assert output.shape == (2, 10)

    def test_get_model_factory(self):
        """Test model factory function."""
        model = get_model("simple_cnn", num_classes=10)
        assert isinstance(model, SimpleCNN)

        model = get_model("cifar10_cnn", num_classes=10)
        assert isinstance(model, CIFAR10CNN)

        model = get_model("simple_resnet", num_classes=10)
        assert isinstance(model, SimpleResNet)

    def test_get_model_invalid_name(self):
        """Test model factory with invalid name."""
        with pytest.raises(ValueError):
            get_model("invalid_model")

    def test_count_parameters(self):
        """Test parameter counting."""
        model = SimpleCNN(num_classes=10)
        total, trainable = count_parameters(model)
        assert total > 0
        assert trainable > 0
        assert trainable == total  # All params should be trainable

    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = SimpleCNN(num_classes=10)
        x = torch.randn(2, 1, 28, 28, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None
