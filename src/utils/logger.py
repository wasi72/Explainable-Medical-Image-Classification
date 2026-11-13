"""Logging configuration for the medical imaging framework."""

import logging
import os
import sys
from typing import Optional


def get_logger(name: str = "ChestXRayAI", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Creates or returns a standard structured logger.

    Args:
        name: Name of the logger.
        log_file: Optional path to save log outputs.
        level: Logging verbosity level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
