"""Shared logging helpers for chat / ingest / LLM demos."""

from __future__ import annotations

import logging
import os
import sys


_CONFIGURED = False


def setup_logging(name: str = "cad_agent") -> logging.Logger:
    """Configure root logging once; return a named logger.

    Level from CAD_LOG_LEVEL (DEBUG/INFO/WARNING/ERROR), default INFO.
    """
    global _CONFIGURED
    level_name = (os.environ.get("CAD_LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stderr,
            force=True,
        )
        _CONFIGURED = True
    else:
        logging.getLogger().setLevel(level)

    return logging.getLogger(name)
