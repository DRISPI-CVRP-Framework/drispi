"""Structured logging setup for experiments."""

from __future__ import annotations

import logging
from typing import Any


def configure_logging(level: int = logging.INFO, json_format: bool = False) -> None:
    """Configure root logger for CLI and batch runs."""
    # TODO: basicConfig or structured JSON handler
    raise NotImplementedError


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    # TODO: return logging.getLogger(name)
    raise NotImplementedError
