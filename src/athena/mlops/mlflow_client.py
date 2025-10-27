"""
MLflow client for experiment tracking and model management.

Integrates MLflow with external storage configuration.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import mlflow
from mlflow.tracking import MlflowClient as _MlflowClient

from athena.config.storage import get_storage_config

logger = logging.getLogger(__name__)


class MLflowClient:
    """
    Wrapper around MLflow client with external storage integration.

    Handles automatic configuration of tracking URI and artifact storage
    based on external drive availability.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize MLflow client.

        Args:
            tracking_uri: MLflow tracking URI. If None, uses environment variable
                or auto-configures based on storage.
            artifact_location: Artifact storage location. If None, uses external
                storage if available.
        """
        self.storage_config = get_storage_config()
        self.tracking_uri = tracking_uri or self._get_tracking_uri()
        self.artifact_location = artifact_location or self._get_artifact_location()

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = _MlflowClient(tracking_uri=self.tracking_uri)

        logger.info(f"MLflow tracking URI: {self.tracking_uri}")
        logger.info(f"MLflow artifact location: {self.artifact_location}")

    def _get_tracking_uri(self) -> str:
        """
        Get MLflow tracking URI.

        Returns:
            Tracking URI string.
        """
        # Check environment variable
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            return tracking_uri

        # Use SQLite database in local storage
        db_path = self.storage_config.project_root / "local_storage" / "sqlite_db"
        db_path.mkdir(parents=True, exist_ok=True)

        # Convert to proper URI format with forward slashes
        db_file = (db_path / "mlflow.db").as_posix()
        return f"sqlite:///{db_file}"

    def _get_artifact_location(self) -> str:
        """
        Get artifact storage location.

        Uses external storage if available, otherwise local storage.

        Returns:
            Artifact location URI (file:// format).
        """
        # Check environment variable
        artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT")
        if artifact_root:
            return artifact_root

        # Use external storage for artifacts
        artifact_path = self.storage_config.get_storage_path("mlflow_artifacts")

        # Convert to file:// URI format for MLflow
        from pathlib import Path
        artifact_uri = Path(artifact_path).as_posix()

        # Ensure proper file:// URI format
        if not artifact_uri.startswith("file://"):
            if artifact_uri.startswith("/"):
                artifact_uri = f"file://{artifact_uri}"
            else:
                artifact_uri = f"file:///{artifact_uri}"

        return artifact_uri

    def create_experiment(
        self,
        name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new MLflow experiment.

        Args:
            name: Experiment name.
            artifact_location: Artifact location for this experiment.
                If None, uses default artifact location.
            tags: Optional tags for the experiment.

        Returns:
            Experiment ID.
        """
        artifact_loc = artifact_location or self.artifact_location

        try:
            experiment_id = self.client.create_experiment(
                name=name,
                artifact_location=artifact_loc,
                tags=tags,
            )
            logger.info(f"Created experiment '{name}' with ID: {experiment_id}")
            return experiment_id
        except Exception as e:
            # Experiment might already exist
            logger.debug(f"Could not create experiment '{name}': {e}")
            experiment = self.client.get_experiment_by_name(name)
            if experiment:
                return experiment.experiment_id
            raise

    def get_or_create_experiment(self, name: str) -> str:
        """
        Get existing experiment or create new one.

        Args:
            name: Experiment name.

        Returns:
            Experiment ID.
        """
        experiment = self.client.get_experiment_by_name(name)
        if experiment:
            return experiment.experiment_id
        return self.create_experiment(name)

    def log_model_artifacts(
        self,
        run_id: str,
        model_path: Path,
        artifact_path: str = "model",
    ) -> None:
        """
        Log model artifacts to MLflow.

        Args:
            run_id: MLflow run ID.
            model_path: Path to model file or directory.
            artifact_path: Artifact path within the run.
        """
        self.client.log_artifact(run_id, str(model_path), artifact_path)
        logger.info(f"Logged model artifacts for run {run_id}")

    def get_experiment_info(self) -> Dict[str, Any]:
        """
        Get information about MLflow configuration.

        Returns:
            Dictionary with configuration details.
        """
        return {
            "tracking_uri": self.tracking_uri,
            "artifact_location": self.artifact_location,
            "external_storage_available": self.storage_config.external_storage_available,
            "external_storage_path": str(self.storage_config.external_storage_path)
            if self.storage_config.external_storage_available
            else None,
        }


def initialize_mlflow(
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None,
    default_experiment: str = "default",
) -> MLflowClient:
    """
    Initialize MLflow with external storage configuration.

    Args:
        tracking_uri: MLflow tracking URI.
        artifact_location: Artifact storage location.
        default_experiment: Name of default experiment to create.

    Returns:
        Configured MLflowClient instance.
    """
    client = MLflowClient(tracking_uri=tracking_uri, artifact_location=artifact_location)

    # Set environment variable for default artifact location
    # This ensures MLflow uses the correct artifact root for all operations
    if not os.getenv("MLFLOW_ARTIFACT_ROOT"):
        os.environ["MLFLOW_ARTIFACT_ROOT"] = client.artifact_location
        logger.info(f"Set MLFLOW_ARTIFACT_ROOT to: {client.artifact_location}")

    # Create default experiment with explicit artifact location
    client.get_or_create_experiment(default_experiment)

    logger.info("MLflow initialized successfully")
    return client
