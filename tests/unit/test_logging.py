"""Unit tests for logging configuration."""

import logging
from pathlib import Path

from src.utils.logging import setup_logging


def test_setup_logging_console_and_file(tmp_path: Path) -> None:
    log_file = tmp_path / "test.log"
    logger = setup_logging(level="DEBUG", log_file=log_file, run_id="TEST_RUN_01")

    assert logger.level == logging.DEBUG

    test_logger = logging.getLogger("test_module")
    test_logger.info("Test message for logging verification")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "TEST_RUN_01" in content
    assert "Test message for logging verification" in content
