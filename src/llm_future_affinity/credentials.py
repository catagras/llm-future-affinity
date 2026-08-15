"""Credential lookup without exposing secrets through configuration objects."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def openrouter_api_key() -> str | None:
    """Return the project .env key, falling back to the process environment."""
    value = dotenv_values(PROJECT_ENV_FILE).get("OPENROUTER_API_KEY") if PROJECT_ENV_FILE.is_file() else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return os.environ.get("OPENROUTER_API_KEY")
