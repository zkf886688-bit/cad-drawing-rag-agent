"""Load project-root .env into os.environ (never override existing vars)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_local_env(path: Path | None = None) -> Path | None:
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return None
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Treat a present-but-empty variable as unset. This is common on
        # Windows when a terminal/profile predeclares API key names.
        if key and not os.environ.get(key):
            os.environ[key] = value
    return env_path
