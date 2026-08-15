from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_future_affinity.config import AppConfig, LoadedConfig
from llm_future_affinity.debug import DebugWriter
from llm_future_affinity.manifest import build_manifest, canonical_json, ensure_manifest, sha256_file, sha256_tree


def make_loaded(tmp_path: Path, app_config: AppConfig) -> LoadedConfig:
    return LoadedConfig(config=app_config, source_path=tmp_path / "config.yaml")


def test_manifest_is_stable_and_detects_changes(tmp_path: Path, app_config: AppConfig, games_csv: Path) -> None:
    loaded = make_loaded(tmp_path, app_config)
    first = build_manifest(loaded, "test-model", games_csv, 2, tmp_path / "output.csv")
    second = build_manifest(loaded, "test-model", games_csv, 2, tmp_path / "output.csv")
    assert first["experiment_fingerprint"] == second["experiment_fingerprint"]
    assert first["games_sha256"] == sha256_file(games_csv)
    app_config.game.initial_query_credits = 3
    changed = build_manifest(loaded, "test-model", games_csv, 2, tmp_path / "output.csv")
    assert changed["experiment_fingerprint"] != first["experiment_fingerprint"]


def test_ensure_manifest_create_update_and_reject(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    expected: dict[str, Any] = {"experiment_fingerprint": "abc", "latest_invocation_at": "old"}
    ensure_manifest(path, expected)
    created = json.loads(path.read_text(encoding="utf-8"))
    assert created["experiment_fingerprint"] == "abc"
    ensure_manifest(path, expected)
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["latest_invocation_at"] != "old"
    with pytest.raises(ValueError, match="fingerprint"):
        ensure_manifest(path, {"experiment_fingerprint": "different"})


def test_canonical_json_sorts_keys() -> None:
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_source_tree_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    source = package / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    first = sha256_tree(package)
    assert first == sha256_tree(package)
    source.write_text("value = 2\n", encoding="utf-8")
    assert sha256_tree(package) != first


async def test_debug_writer_groups_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "debug.jsonl"
    writer = DebugWriter(path)
    writer.start()
    await writer.write_conversation({"attempt_id": "one", "attempts": [{"response": "QUERY ABCD"}]})
    writer.close()
    writer.close()
    assert json.loads(path.read_text(encoding="utf-8"))["attempt_id"] == "one"


async def test_debug_writer_requires_start(tmp_path: Path) -> None:
    writer = DebugWriter(tmp_path / "debug.jsonl")
    with pytest.raises(RuntimeError, match="not started"):
        await writer.write_conversation({})
