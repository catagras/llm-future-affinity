"""Experiment manifest and compatibility fingerprint."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_future_affinity import __version__
from llm_future_affinity.config import LoadedConfig
from llm_future_affinity.prompting import render_all_prompts

SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_manifest(
    loaded: LoadedConfig,
    model_key: str,
    games_path: Path,
    game_count: int,
    output_path: Path,
) -> dict[str, Any]:
    config = loaded.config
    model = config.model_for(model_key)
    prompts = {track.value: rendered for track, rendered in render_all_prompts(config.game, config.prompt).items()}
    critical = {
        "schema_version": SCHEMA_VERSION,
        "game": config.game.model_dump(mode="json"),
        "prompts": prompts,
        "model": model.model_dump(mode="json"),
        "games_sha256": sha256_file(games_path),
        "scoring": "positions_correct/code_length",
        "protocol": {
            "invalid_response_limit": 3,
            "zero_credit_query_is_submit": True,
            "outer_whitespace_only": True,
        },
        "runner_version": __version__,
        "runner_source_sha256": sha256_tree(Path(__file__).resolve().parent),
    }
    fingerprint = hashlib.sha256(canonical_json(critical).encode("utf-8")).hexdigest()
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": config.experiment.name,
        "model_key": model_key,
        "model_family": model.model_family,
        "model_id": model.model_id,
        "routing": model.routing.model_dump(mode="json"),
        "inference": model.inference.model_dump(mode="json"),
        "resolved_prompts": prompts,
        "games_file": str(games_path),
        "games_sha256": critical["games_sha256"],
        "game_count": game_count,
        "output_file": str(output_path),
        "runner_version": __version__,
        "runner_source_sha256": critical["runner_source_sha256"],
        "git": git_metadata(loaded.base_dir),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "created_at": now,
        "latest_invocation_at": now,
        "experiment_fingerprint": fingerprint,
        "critical_configuration": critical,
    }


def ensure_manifest(path: Path, expected: dict[str, Any]) -> None:
    if path.exists():
        existing = validate_manifest(path, expected)
        existing["latest_invocation_at"] = utc_now()
        _write_atomic(path, existing)
        return
    _write_atomic(path, expected)


def validate_manifest(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    existing = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        raise ValueError("manifest must contain a JSON object")
    if existing.get("experiment_fingerprint") != expected["experiment_fingerprint"]:
        raise ValueError("experiment fingerprint does not match the existing manifest")
    return existing


def git_metadata(cwd: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except OSError, subprocess.CalledProcessError:
        return {"commit": None, "dirty": None}


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path.rglob("*.py")):
        digest.update(source.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
