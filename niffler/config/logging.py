"""Unified logging configuration for all Niffler scripts and services.

This module lives inside the ``niffler`` package so that importing it never
depends on the repository root being on ``sys.path``.  ``config/logging.py`` at
the repository root is kept as a thin backwards-compatible shim.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

# Levels accepted by setup_logging. Deliberately a whitelist: a plain
# getattr(logging, level.upper()) happily returns unrelated module attributes
# and raises a confusing AttributeError on a typo.
VALID_LOG_LEVELS: Tuple[str, ...] = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')

DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DEFAULT_LOG_DIR = 'logs'
DEFAULT_LOG_FILENAME = 'niffler.log'


def resolve_log_level(level: str) -> int:
    """Translate a log level name into its numeric logging constant.

    Args:
        level: Log level name, case insensitive (e.g. "info", "DEBUG").

    Returns:
        The numeric logging level (e.g. ``logging.INFO``).

    Raises:
        TypeError: If level is not a string.
        ValueError: If level is not one of VALID_LOG_LEVELS.
    """
    if not isinstance(level, str):
        raise TypeError(
            f"Log level must be a string, got {type(level).__name__}. "
            f"Valid levels are: {', '.join(VALID_LOG_LEVELS)}"
        )

    normalized = level.strip().upper()
    if normalized not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level: {level!r}. "
            f"Valid levels are: {', '.join(VALID_LOG_LEVELS)}"
        )

    return getattr(logging, normalized)


def setup_logging(level: str = "INFO", log_to_file: bool = False,
                  log_file: Optional[str] = None) -> None:
    """Setup unified logging configuration for all Niffler scripts.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_file: Whether to also log to a file.
        log_file: Optional specific log file path. Only used when
            log_to_file is True; defaults to ``logs/niffler.log``.

    Raises:
        TypeError: If level is not a string.
        ValueError: If level is not a recognised logging level.
        OSError: If the log file or its directory cannot be created.
    """
    numeric_level = resolve_log_level(level)

    handlers: List[logging.Handler] = []

    if log_to_file:
        resolved_log_file = Path(log_file) if log_file else Path(DEFAULT_LOG_DIR) / DEFAULT_LOG_FILENAME
        if str(resolved_log_file.parent) not in ('', '.'):
            resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(resolved_log_file)))

    handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=numeric_level,
        format=DEFAULT_LOG_FORMAT,
        handlers=handlers
    )
