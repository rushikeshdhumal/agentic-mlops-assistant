"""
CLI entry point for ATHENA MLOps Platform.
"""

import argparse
import sys
from pathlib import Path

from athena.config.storage import StorageConfig
from athena.utils.logging import setup_logging


def main():
    """Main entry point for ATHENA CLI."""
    parser = argparse.ArgumentParser(
        description="ATHENA MLOps Platform - Agentic assistant for ML workflows"
    )
    parser.add_argument(
        "--mode",
        choices=["ui", "api", "cli"],
        default="ui",
        help="Launch mode: ui (Streamlit), api (FastAPI), or cli (interactive)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--check-storage", action="store_true", help="Check external storage configuration"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logging(level=log_level)

    logger.info(f"Starting ATHENA MLOps Platform v{__import__('athena').__version__}")

    # Check storage if requested
    if args.check_storage:
        storage_config = StorageConfig()
        logger.info(f"External storage detected: {storage_config.external_storage_available}")
        if storage_config.external_storage_available:
            logger.info(f"External storage path: {storage_config.external_storage_path}")
            logger.info(f"Storage paths: {storage_config.get_all_paths()}")
        else:
            logger.warning("External storage not detected. Using local storage only.")
        return 0

    # Launch appropriate mode
    if args.mode == "ui":
        logger.info("Launching Streamlit UI...")
        # TODO: Launch Streamlit app
        logger.error("UI mode not yet implemented")
        return 1
    elif args.mode == "api":
        logger.info("Launching FastAPI server...")
        # TODO: Launch FastAPI server
        logger.error("API mode not yet implemented")
        return 1
    elif args.mode == "cli":
        logger.info("Starting interactive CLI...")
        # TODO: Start CLI loop
        logger.error("CLI mode not yet implemented")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
