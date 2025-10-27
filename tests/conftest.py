"""
Pytest configuration and shared fixtures.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_storage_config(temp_dir, monkeypatch):
    """Mock storage configuration for testing."""
    from athena.config.storage import StorageConfig

    # Create mock storage instance with temp directory
    config = StorageConfig(project_root=temp_dir)
    config.external_storage_path = temp_dir / "external"
    config.external_storage_available = True

    # Patch the singleton
    monkeypatch.setattr("athena.config.storage._storage_config_instance", config)

    return config
