"""Structured logging setup for the project."""

import logging
import os
import sys
from pathlib import Path


def setup_logging(
    level: str | None = None,
    log_file: Path | str | None = None,
    run_id: str | None = None,
) -> logging.Logger:
    """
    Configure root and project-level logging with console and optional file output.

    Args:
        level: Logging level (e.g., 'DEBUG', 'INFO', 'WARNING'). Defaults to env LOG_LEVEL or 'INFO'.
        log_file: Optional path to save log output.
        run_id: Optional run identifier to tag logs with.

    Returns:
        Configured root logger.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    log_level = getattr(logging, level, logging.INFO)

    prefix = f"[{run_id}] " if run_id else ""
    log_format = f"%(asctime)s - {prefix}%(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Reset existing handlers on root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        root_logger.addHandler(file_handler)

    return root_logger
