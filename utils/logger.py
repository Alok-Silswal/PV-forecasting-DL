"""
Logging utilities for short-term photovoltaic (PV) power forecasting.

This module configures the project logger used throughout model training.
Logs are written to both the console and a log file for reproducibility
and experiment tracking.
"""

import logging
from pathlib import Path

from configs import config


def get_logger(
    log_dir: str | Path = config.MODEL_EXPERIMENT_DIR,
    log_file: str = "training.log",
) -> logging.Logger:

    log_directory = Path(log_dir)
    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Derive a logger name from the experiment directory so that each
    # experiment gets its own independent,
    # cached logger instead of all runs sharing a single global
    # "PVForecasting" logger within the same Python kernel.
    logger_name = f"PVForecasting.{log_directory.resolve()}"

    logger = logging.getLogger(logger_name)

    # Prevent duplicate handlers for this specific experiment logger,
    # while still allowing different experiments to have different
    # loggers (and therefore different log files).
    if logger.handlers:
        return logger

    log_file_path = log_directory / log_file

    logger.setLevel(logging.INFO)

    # Avoid propagating to the root logger, which would otherwise cause
    # duplicate console output when multiple per-experiment loggers are
    # created within the same process.
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(
        log_file_path,
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger