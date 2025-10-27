"""
External storage detection and configuration for ATHENA MLOps Platform.

This module handles automatic detection of the DHUMAL external drive and configures
paths for heavy components (models, datasets, artifacts) to be stored externally
while keeping code and lightweight components local.
"""

import os
import platform
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class StorageConfig:
    """
    Manages storage configuration with automatic external drive detection.

    The DHUMAL drive (exFAT filesystem) is used for storing large files:
    - Ollama models (~4.7 GB)
    - Training datasets (~2 GB)
    - MLflow artifacts (~2 GB)
    - Model checkpoints (~3 GB)
    - Cache files (~2 GB)

    Local storage (<10 GB) is used for:
    - Code and dependencies
    - SQLite databases
    - Vector store (Chroma)
    - Configuration files
    """

    DRIVE_NAME = "DHUMAL"
    MOUNT_PATHS = {
        "Darwin": "/Volumes/DHUMAL",  # macOS
        "Linux": "/mnt/dhumal",  # Linux
        "Windows": ["D:/", "E:/", "F:/", "G:/"],  # Windows - check multiple drive letters
    }

    # Components to externalize
    EXTERNAL_COMPONENTS = [
        "ollama_models",
        "datasets",
        "mlflow_artifacts",
        "model_checkpoints",
        "cache",
    ]

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize storage configuration.

        Args:
            project_root: Root directory of the project. If None, auto-detected.
        """
        self.project_root = project_root or self._detect_project_root()
        self.system = platform.system()
        self.external_storage_path = self._detect_external_storage()
        self.external_storage_available = self.external_storage_path is not None

        logger.info(f"Project root: {self.project_root}")
        logger.info(f"External storage available: {self.external_storage_available}")
        if self.external_storage_available:
            logger.info(f"External storage path: {self.external_storage_path}")

    def _detect_project_root(self) -> Path:
        """Detect the project root directory."""
        # Start from current file and go up to find project root
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / "src" / "athena").exists() or (parent / ".git").exists():
                return parent
        # Fallback to current working directory
        return Path.cwd()

    def _detect_external_storage(self) -> Optional[Path]:
        """
        Detect the DHUMAL external drive.

        Returns:
            Path to external storage if found, None otherwise.
        """
        system = self.system

        if system == "Windows":
            # Check multiple drive letters on Windows
            for drive_path in self.MOUNT_PATHS["Windows"]:
                path = Path(drive_path)
                if self._is_dhumal_drive(path):
                    return path
        else:
            # macOS or Linux
            mount_path = self.MOUNT_PATHS.get(system)
            if mount_path:
                path = Path(mount_path)
                if self._is_dhumal_drive(path):
                    return path

        logger.warning(
            f"DHUMAL drive not detected. Checked paths: {self._get_checked_paths()}"
        )
        return None

    def _is_dhumal_drive(self, path: Path) -> bool:
        """
        Check if the given path is the DHUMAL drive.

        Args:
            path: Path to check.

        Returns:
            True if this is the DHUMAL drive, False otherwise.
        """
        if not path.exists():
            return False

        # Check for marker file or directory structure
        athena_marker = path / ".athena_storage"
        if athena_marker.exists():
            return True

        # Check volume name on Windows
        if self.system == "Windows":
            try:
                import win32api
                volume_name = win32api.GetVolumeInformation(str(path))[0]
                if volume_name == self.DRIVE_NAME:
                    return True
            except ImportError:
                # win32api not available, use alternative method
                pass
            except Exception as e:
                logger.debug(f"Error checking volume name for {path}: {e}")

        # For macOS/Linux, check if the path contains DHUMAL in the name
        if self.DRIVE_NAME in str(path):
            return True

        return False

    def _get_checked_paths(self) -> list:
        """Get list of paths that were checked for external storage."""
        if self.system == "Windows":
            return self.MOUNT_PATHS["Windows"]
        else:
            return [self.MOUNT_PATHS.get(self.system, "")]

    def get_storage_path(self, component: str, ensure_exists: bool = True) -> Path:
        """
        Get storage path for a specific component.

        Args:
            component: Component name (e.g., 'ollama_models', 'datasets').
            ensure_exists: Create directory if it doesn't exist.

        Returns:
            Path for the component storage.
        """
        if component in self.EXTERNAL_COMPONENTS and self.external_storage_available:
            # Store on external drive
            path = self.external_storage_path / "athena" / component
        else:
            # Store locally
            path = self.project_root / "local_storage" / component

        if ensure_exists:
            path.mkdir(parents=True, exist_ok=True)

        return path

    def get_all_paths(self) -> Dict[str, Path]:
        """
        Get all configured storage paths.

        Returns:
            Dictionary mapping component names to their storage paths.
        """
        paths = {}
        for component in self.EXTERNAL_COMPONENTS:
            paths[component] = self.get_storage_path(component, ensure_exists=False)

        # Add local storage paths
        paths["logs"] = self.project_root / "logs"
        paths["configs"] = self.project_root / "configs"
        paths["vector_db"] = self.project_root / "local_storage" / "vector_db"
        paths["sqlite_db"] = self.project_root / "local_storage" / "sqlite_db"

        return paths

    def create_symlinks(self) -> None:
        """
        Create symlinks in project root to external storage locations.

        This makes external storage paths accessible from the project directory
        for convenience during development.
        """
        if not self.external_storage_available:
            logger.warning("External storage not available. Skipping symlink creation.")
            return

        symlinks = {
            "data": self.get_storage_path("datasets"),
            "mlruns": self.get_storage_path("mlflow_artifacts"),
            "models": self.get_storage_path("model_checkpoints"),
        }

        for link_name, target in symlinks.items():
            link_path = self.project_root / link_name

            # Skip if symlink already exists and points to correct target
            if link_path.exists() or link_path.is_symlink():
                if link_path.is_symlink() and link_path.resolve() == target.resolve():
                    logger.debug(f"Symlink {link_name} already exists and is correct")
                    continue
                else:
                    logger.warning(
                        f"Path {link_path} already exists but is not a valid symlink. Skipping."
                    )
                    continue

            try:
                # Create symlink
                if self.system == "Windows":
                    # Windows requires admin privileges for symlinks, use junction instead
                    import subprocess
                    subprocess.run(
                        ["mklink", "/J", str(link_path), str(target)],
                        shell=True,
                        check=True,
                    )
                else:
                    link_path.symlink_to(target)

                logger.info(f"Created symlink: {link_name} -> {target}")
            except Exception as e:
                logger.error(f"Failed to create symlink {link_name}: {e}")

    def initialize_storage(self) -> None:
        """
        Initialize storage structure by creating all necessary directories.

        Also creates a marker file on external storage for future detection.
        """
        # Create all storage paths
        for component in self.EXTERNAL_COMPONENTS:
            path = self.get_storage_path(component, ensure_exists=True)
            logger.info(f"Initialized storage for {component}: {path}")

        # Create local storage paths
        for path_name in ["logs", "configs", "vector_db", "sqlite_db"]:
            path = self.get_all_paths()[path_name]
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized local storage for {path_name}: {path}")

        # Create marker file on external storage
        if self.external_storage_available:
            marker_file = self.external_storage_path / ".athena_storage"
            if not marker_file.exists():
                marker_file.write_text(
                    "This drive is configured for ATHENA MLOps Platform storage.\n"
                    f"Project root: {self.project_root}\n"
                )
                logger.info(f"Created storage marker file: {marker_file}")

        # Create symlinks
        self.create_symlinks()

    def get_storage_info(self) -> Dict[str, any]:
        """
        Get storage configuration information.

        Returns:
            Dictionary with storage configuration details.
        """
        info = {
            "project_root": str(self.project_root),
            "system": self.system,
            "external_storage_available": self.external_storage_available,
            "external_storage_path": str(self.external_storage_path) if self.external_storage_available else None,
            "storage_paths": {k: str(v) for k, v in self.get_all_paths().items()},
        }
        return info


# Singleton instance
_storage_config_instance: Optional[StorageConfig] = None


def get_storage_config() -> StorageConfig:
    """
    Get the singleton StorageConfig instance.

    Returns:
        StorageConfig instance.
    """
    global _storage_config_instance
    if _storage_config_instance is None:
        _storage_config_instance = StorageConfig()
    return _storage_config_instance
