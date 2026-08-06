"""Production logging setup for the FreshSense pipeline.

Configures three log sinks:

  - Console          : human-readable ``INFO`` output to stdout.
  - ``training.log`` : rotating file handler with ``INFO`` level.
  - ``errors.log``   : rotating file handler with ``ERROR`` level.

The setup function is idempotent: calling :func:`setup_logging` multiple
times removes previously registered FreshSense handlers before re-adding them,
so repeated calls never create duplicate log lines.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

__all__ = ["setup_logging"]

# Names we own so we can remove them on re-initialisation.
_CONSOLE_HANDLER_NAME = "freshsense_console_handler"
_TRAINING_HANDLER_NAME = "freshsense_training_file_handler"
_ERRORS_HANDLER_NAME = "freshsense_errors_file_handler"

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default: 5 MB per file, 3 backups (5 MB worth of rotation each).
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 3


def _remove_existing_freshsense_handlers(logger: logging.Logger) -> None:
    """Remove handlers we previously installed so setup is idempotent."""
    for handler in list(logger.handlers):
        if handler.name in {
            _CONSOLE_HANDLER_NAME,
            _TRAINING_HANDLER_NAME,
            _ERRORS_HANDLER_NAME,
        }:
            handler.close()
            logger.removeHandler(handler)


def _add_console_handler(
    logger: logging.Logger, level: int = logging.INFO
) -> None:
    """Add a console handler at the given level."""
    handler: logging.Handler = logging.StreamHandler()
    handler.set_name(_CONSOLE_HANDLER_NAME)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)
    )
    logger.addHandler(handler)


def _add_file_handler(
    logger: logging.Logger,
    log_dir: Path,
    filename: str,
    handler_name: str,
    level: int,
) -> Path:
    """Add a rotating file handler and return the created file path."""
    path = log_dir / filename
    handler = RotatingFileHandler(
        path,
        maxBytes=_DEFAULT_MAX_BYTES,
        backupCount=_DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.set_name(handler_name)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    return path


def setup_logging(
    log_dir: str | Path,
    console_level: int = logging.INFO,
    file_level: int = logging.INFO,
) -> tuple[Path, Path]:
    """Configure console + file logging for the whole application.

    Args:
        log_dir: Directory where ``training.log`` and ``errors.log`` will live.
        console_level: Log level for the console handler.
        file_level: Log level for the ``training.log`` file handler.

    Returns:
        ``(training_log_path, errors_log_path)`` as resolved ``Path`` objects.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Lowest level; handlers filter.

    _remove_existing_freshsense_handlers(root_logger)
    _add_console_handler(root_logger, console_level)
    training_path = _add_file_handler(
        root_logger,
        log_dir,
        "training.log",
        _TRAINING_HANDLER_NAME,
        file_level,
    )
    errors_path = _add_file_handler(
        root_logger,
        log_dir,
        "errors.log",
        _ERRORS_HANDLER_NAME,
        logging.ERROR,
    )

    logging.getLogger(__name__).info(
        "Logging initialised: console=%s, training.log=%s, errors.log=%s",
        logging.getLevelName(console_level),
        training_path,
        errors_path,
    )
    return training_path, errors_path