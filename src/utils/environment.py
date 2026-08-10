"""FreshSense AI - Centralized environment loading.

This module provides a single, authoritative way to load environment variables
from .env files and access them safely throughout the project.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

__all__ = ["load_environment", "get_env", "require_env"]

# Keys whose values should never be logged verbatim.
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "secret",
        "password",
        "token",
        "private_key",
        "credentials",
        "auth",
    }
)


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(secret in lowered for secret in _SENSITIVE_KEYS)


def load_environment(
    env_file: str = ".env",
    override: bool = True,
    verbose: bool = False,
) -> None:
    """Load environment variables from .env file.

    This function is idempotent and safe to call multiple times. It uses
    python-dotenv to read the file and populate os.environ.
    """
    loaded = load_dotenv(dotenv_path=env_file, override=override)
    if loaded:
        logger.info("Loaded environment variables from %s", env_file)
        if verbose:
            for key in os.environ:
                if _is_sensitive(key):
                    logger.debug("%s is set", key)
                else:
                    logger.debug("%s = %s", key, os.environ[key])


def get_env(key: str, default: Optional[str] = None) -> str:
    """Read an environment variable, returning default if missing."""
    value = os.environ.get(key, default)
    if value is not None and _is_sensitive(key):
        logger.debug("%s is set", key)
    return value if value is not None else ""


def require_env(key: str) -> str:
    """Read a required environment variable or exit with an error."""
    value = os.environ.get(key)
    if value is None:
        logger.error("Required environment variable %s is not set.", key)
        raise SystemExit(f"ERROR: {key} is not set. Add it to your .env file.")
    return value

