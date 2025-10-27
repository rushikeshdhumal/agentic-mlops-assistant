"""
Logging configuration for ATHENA MLOps Platform.

Provides structured logging with both console and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up structured logging for ATHENA.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. If None, uses project_root/logs.
        log_file: Log file name. If None, generates timestamp-based name.

    Returns:
        Root logger instance.
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # File handler
    if log_dir is None:
        # Try to detect project root
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / "src" / "athena").exists():
                log_dir = parent / "logs"
                break
        if log_dir is None:
            log_dir = Path.cwd() / "logs"

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"athena_{timestamp}.log"

    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Log initialization
    root_logger.info(f"Logging initialized at {level} level")
    root_logger.info(f"Log file: {log_dir / log_file}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
