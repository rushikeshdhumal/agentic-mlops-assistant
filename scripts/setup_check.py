"""
Setup verification script for ATHENA MLOps Platform.

Checks that all components are properly configured.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from athena.config.storage import StorageConfig
from athena.utils.logging import setup_logging

logger = setup_logging(level="INFO")


def check_python_version():
    """Check Python version."""
    logger.info("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        logger.info(f"Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        logger.error(f"Python {version.major}.{version.minor} - Requires Python 3.9+")
        return False


def check_imports():
    """Check that key packages can be imported."""
    logger.info("Checking package imports...")
    packages = [
        ("torch", "PyTorch"),
        ("mlflow", "MLflow"),
        ("langchain", "LangChain"),
        ("chromadb", "ChromaDB"),
        ("streamlit", "Streamlit"),
        ("fastapi", "FastAPI"),
    ]

    success = True
    for package, name in packages:
        try:
            __import__(package)
            logger.info(f"{name} - OK")
        except ImportError:
            logger.error(f"{name} - NOT FOUND")
            success = False

    return success


def check_storage():
    """Check storage configuration."""
    logger.info("Checking storage configuration...")

    try:
        storage_config = StorageConfig()
        logger.info(f"Project root: {storage_config.project_root}")
        logger.info(f"System: {storage_config.system}")

        if storage_config.external_storage_available:
            logger.info(f"External storage: {storage_config.external_storage_path}")
        else:
            logger.warning("External storage (DHUMAL drive) not detected")
            logger.warning("  Will use local storage (may fill up quickly)")

        # Check paths
        paths = storage_config.get_all_paths()
        logger.info(f"Configured {len(paths)} storage paths")

        return True
    except Exception as e:
        logger.error(f"Storage configuration failed: {e}")
        return False


def check_ollama():
    """Check if Ollama is running."""
    logger.info("Checking Ollama availability...")

    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            logger.info(f"Ollama is running with {len(models)} models")
            for model in models:
                logger.info(f"  - {model.get('name', 'unknown')}")
            return True
        else:
            logger.warning("Ollama API returned unexpected status")
            return False
    except Exception as e:
        logger.warning(f"Ollama not available: {e}")
        logger.warning("  Install from https://ollama.ai and run: ollama pull llama3.1:8b")
        return False


def check_directory_structure():
    """Check that directory structure is correct."""
    logger.info("Checking directory structure...")

    project_root = Path(__file__).parent.parent
    required_dirs = [
        "src/athena",
        "tests",
        "scripts",
        "docs",
        "configs",
    ]

    success = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            logger.info(f"{dir_path}")
        else:
            logger.error(f"{dir_path} - NOT FOUND")
            success = False

    return success


def main():
    """Run all checks."""
    logger.info("="*60)
    logger.info("ATHENA MLOps Platform - Setup Verification")
    logger.info("="*60)

    checks = [
        ("Python Version", check_python_version),
        ("Package Imports", check_imports),
        ("Storage Configuration", check_storage),
        ("Directory Structure", check_directory_structure),
        ("Ollama", check_ollama),
    ]

    results = []
    for name, check_func in checks:
        logger.info("")
        logger.info("-"*60)
        result = check_func()
        results.append((name, result))

    # Summary
    logger.info("")
    logger.info("="*60)
    logger.info("Summary")
    logger.info("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{status:10} | {name}")

    logger.info("="*60)
    logger.info(f"Result: {passed}/{total} checks passed")

    if passed == total:
        logger.info("All checks passed! ATHENA is ready to use.")
        return 0
    else:
        logger.warning(f"{total - passed} check(s) failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
