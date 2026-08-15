from __future__ import annotations

from pathlib import Path

import pytest

from llm_future_affinity import credentials


def test_openrouter_api_key_prefers_project_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('OPENROUTER_API_KEY="dotenv-key"\n', encoding="utf-8")
    monkeypatch.setattr(credentials, "PROJECT_ENV_FILE", env_file)
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-key")

    assert credentials.openrouter_api_key() == "dotenv-key"


def test_openrouter_api_key_falls_back_to_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials, "PROJECT_ENV_FILE", tmp_path / ".env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-key")

    assert credentials.openrouter_api_key() == "environment-key"
