"""
Quick test script to verify MLflow configuration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from athena.mlops.mlflow_client import initialize_mlflow
from athena.config.storage import get_storage_config
from athena.utils.logging import setup_logging

logger = setup_logging(level="INFO")


def main():
    """Test MLflow configuration."""
    logger.info("Testing MLflow configuration...")

    # Initialize storage
    storage_config = get_storage_config()
    storage_config.initialize_storage()

    # Initialize MLflow
    mlflow_client = initialize_mlflow(default_experiment="test_experiment")

    # Get configuration info
    info = mlflow_client.get_experiment_info()

    logger.info("MLflow Configuration:")
    logger.info(f"  Tracking URI: {info['tracking_uri']}")
    logger.info(f"  Artifact Location: {info['artifact_location']}")
    logger.info(f"  External Storage Available: {info['external_storage_available']}")
    if info['external_storage_path']:
        logger.info(f"  External Storage Path: {info['external_storage_path']}")

    # Test creating an experiment
    exp_id = mlflow_client.get_or_create_experiment("test_experiment")
    logger.info(f"  Test Experiment ID: {exp_id}")

    logger.info("")
    logger.info("Configuration test completed successfully!")
    logger.info("")
    logger.info("Expected artifact URI format should be:")
    logger.info("  file:///D:/athena/mlflow_artifacts")
    logger.info("  (or similar file:// URI)")


if __name__ == "__main__":
    main()
