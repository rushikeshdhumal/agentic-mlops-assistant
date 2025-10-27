"""
Unit tests for storage configuration.
"""

import pytest
from pathlib import Path
from athena.config.storage import StorageConfig


class TestStorageConfig:
    """Tests for StorageConfig class."""

    def test_initialization(self, temp_dir):
        """Test StorageConfig initialization."""
        config = StorageConfig(project_root=temp_dir)
        assert config.project_root == temp_dir
        assert config.system in ["Windows", "Darwin", "Linux"]

    def test_get_storage_path(self, temp_dir):
        """Test getting storage path for a component."""
        config = StorageConfig(project_root=temp_dir)
        path = config.get_storage_path("datasets", ensure_exists=True)
        assert path.exists()
        assert "datasets" in str(path)

    def test_get_all_paths(self, temp_dir):
        """Test getting all configured paths."""
        config = StorageConfig(project_root=temp_dir)
        paths = config.get_all_paths()

        assert "datasets" in paths
        assert "mlflow_artifacts" in paths
        assert "logs" in paths
        assert "vector_db" in paths
        assert isinstance(paths["datasets"], Path)

    def test_initialize_storage(self, temp_dir):
        """Test storage initialization."""
        config = StorageConfig(project_root=temp_dir)
        config.initialize_storage()

        # Check that key directories were created
        paths = config.get_all_paths()
        assert paths["logs"].exists()
        assert paths["vector_db"].exists()

    def test_storage_info(self, temp_dir):
        """Test getting storage information."""
        config = StorageConfig(project_root=temp_dir)
        info = config.get_storage_info()

        assert "project_root" in info
        assert "system" in info
        assert "external_storage_available" in info
        assert "storage_paths" in info
