"""
Training script for MNIST dataset.

Demonstrates MLflow integration with experiment tracking.
"""

import argparse
import logging
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from athena.mlops.models import SimpleCNN, count_parameters
from athena.mlops.mlflow_client import initialize_mlflow
from athena.config.storage import get_storage_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%"
            )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, device, test_loader, criterion):
    """Validate the model."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100.0 * correct / total
    return test_loss, test_acc


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CNN on MNIST")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--experiment", type=str, default="mnist_baseline", help="Experiment name")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA")
    args = parser.parse_args()

    # Initialize storage and MLflow
    storage_config = get_storage_config()
    storage_config.initialize_storage()

    mlflow_client = initialize_mlflow(default_experiment=args.experiment)
    experiment_id = mlflow_client.get_or_create_experiment(args.experiment)

    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"Using device: {device}")

    # Data loading
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    data_path = storage_config.get_storage_path("datasets") / "mnist"
    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = SimpleCNN().to(device)
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id):
        # Log parameters
        mlflow.log_param("model", "SimpleCNN")
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("total_params", total_params)
        mlflow.log_param("trainable_params", trainable_params)
        mlflow.log_param("device", str(device))

        # Training loop
        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(
                model, device, train_loader, optimizer, criterion, epoch
            )
            test_loss, test_acc = validate(model, device, test_loader, criterion)

            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_acc", test_acc, step=epoch)

            logger.info(
                f"Epoch {epoch}/{args.epochs} - "
                f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% - "
                f"Test Loss: {test_loss:.4f} Test Acc: {test_acc:.2f}%"
            )

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                checkpoint_path = storage_config.get_storage_path("model_checkpoints")
                checkpoint_path = checkpoint_path / "mnist_best.pth"
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved best model with accuracy: {best_acc:.2f}%")

        # Log final metrics
        mlflow.log_metric("best_test_acc", best_acc)

        # Log model
        mlflow.pytorch.log_model(model, "model")

        logger.info(f"Training completed. Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
