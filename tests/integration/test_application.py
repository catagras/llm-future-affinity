from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import yaml
from typer.testing import CliRunner

from llm_future_affinity.application import create_plan, execute_plan
from llm_future_affinity.cli import app
from llm_future_affinity.config import load_config
from llm_future_affinity.persistence import read_output
from llm_future_affinity.telemetry import NullTelemetry


def write_project(tmp_path: Path, config_dict: dict[str, Any]) -> Path:
    raw = deepcopy(config_dict)
    raw["experiment"].update(games_file="games.csv", output_dir="outputs", debug_dir="debug")
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    (tmp_path / "games.csv").write_text("game_id,hidden_code\n1,ABCD\n2,AABC\n", encoding="utf-8")
    return config_path


def mock_openrouter() -> respx.Route:
    respx.get("https://openrouter.ai/api/v1/model/test/model").mock(
        return_value=httpx.Response(
            200,
            json={"data": {"supported_parameters": ["max_tokens", "temperature", "top_p", "reasoning"]}},
        )
    )
    respx.get("https://openrouter.ai/api/v1/models/test/model/endpoints").mock(
        return_value=httpx.Response(200, json={"data": {"endpoints": [{"provider_slug": "test-provider/exact"}]}})
    )
    return respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"X-OpenRouter-Cache-Status": "MISS"},
            json={
                "id": "gen-test",
                "provider": "test-provider",
                "provider_endpoint": "test-provider/exact",
                "choices": [{"message": {"role": "assistant", "content": "SUBMIT ABCD"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cost": 0.001},
            },
        )
    )


def test_dry_plan_does_not_mutate_files(tmp_path: Path, config_dict: dict[str, Any]) -> None:
    loaded = load_config(write_project(tmp_path, config_dict))
    plan = create_plan(loaded, "test-model", 1)
    assert [(game.game_id, [track.value for track in tracks]) for game, tracks in plan.selected] == [
        (1, ["A", "B", "C", "D"])
    ]
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "debug").exists()


@respx.mock
async def test_execute_writes_four_tracks_and_resume_moves_to_next_game(
    tmp_path: Path, config_dict: dict[str, Any]
) -> None:
    loaded = load_config(write_project(tmp_path, config_dict))
    plan = create_plan(loaded, "test-model", 1)
    route = mock_openrouter()
    result = await execute_plan(
        loaded,
        "test-model",
        plan,
        disable_otel=True,
        debug=True,
        show_progress=False,
        api_key="secret",
    )
    assert result.successful
    assert route.call_count == 4
    rows = read_output(tmp_path / "outputs" / "test-model.csv")
    assert len(rows) == 4
    assert {row.track.value for row in rows} == {"A", "B", "C", "D"}
    assert all(row.run_status.value == "completed" for row in rows)
    assert (tmp_path / "outputs" / "test-model.manifest.json").exists()
    assert len(list((tmp_path / "debug").glob("*_test-model.jsonl"))) == 1
    resumed = create_plan(loaded, "test-model", 1)
    assert [game.game_id for game, _ in resumed.selected] == [2]
    assert resumed.completed_count == 4


@respx.mock
async def test_execute_refreshes_stale_resume_plan_inside_lock(tmp_path: Path, config_dict: dict[str, Any]) -> None:
    loaded = load_config(write_project(tmp_path, config_dict))
    first_plan = create_plan(loaded, "test-model", 1)
    stale_plan = create_plan(loaded, "test-model", 1)
    route = mock_openrouter()
    first = await execute_plan(
        loaded,
        "test-model",
        first_plan,
        disable_otel=True,
        debug=False,
        show_progress=False,
        api_key="secret",
    )
    second = await execute_plan(
        loaded,
        "test-model",
        stale_plan,
        disable_otel=True,
        debug=False,
        show_progress=False,
        api_key="secret",
    )
    finished = await execute_plan(
        loaded,
        "test-model",
        stale_plan,
        disable_otel=False,
        debug=False,
        show_progress=False,
    )
    assert first.successful and second.successful
    assert finished.successful
    assert route.call_count == 8
    rows = read_output(first_plan.paths.output)
    assert len(rows) == 8
    assert {row.game_id for row in rows} == {1, 2}


def test_cli_print_prompts_without_model(tmp_path: Path, config_dict: dict[str, Any]) -> None:
    config_path = write_project(tmp_path, config_dict)
    result = CliRunner().invoke(app, ["run", "--config", str(config_path), "--print-prompts"])
    assert result.exit_code == 0
    assert "TRACK A" in result.stdout
    assert "comparable-capabilities" in result.stdout


def test_cli_dry_run_and_missing_model(tmp_path: Path, config_dict: dict[str, Any]) -> None:
    config_path = write_project(tmp_path, config_dict)
    runner = CliRunner()
    missing = runner.invoke(app, ["run", "--config", str(config_path)])
    assert missing.exit_code == 2
    assert "--model is required" in missing.output
    dry = runner.invoke(app, ["run", "--config", str(config_path), "--model", "test-model", "--batch-size", "1"])
    assert dry.exit_code == 0
    assert "Dry run only" in dry.stdout
    assert "Selected game IDs: 1" in dry.stdout
    assert not (tmp_path / "outputs").exists()


def test_fingerprint_change_blocks_resume(tmp_path: Path, config_dict: dict[str, Any]) -> None:
    loaded = load_config(write_project(tmp_path, config_dict))
    plan = create_plan(loaded, "test-model", 1)
    plan.paths.manifest.parent.mkdir()
    plan.paths.manifest.write_text('{"experiment_fingerprint":"different"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        create_plan(loaded, "test-model", 1)


@respx.mock
async def test_final_otel_flush_failure_makes_execution_unsuccessful(
    tmp_path: Path, config_dict: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailOnShutdownTelemetry(NullTelemetry):
        def __init__(self) -> None:
            self.failed = False

        async def shutdown(self) -> None:
            self.failed = True

    loaded = load_config(write_project(tmp_path, config_dict))
    plan = create_plan(loaded, "test-model", 1)
    mock_openrouter()
    telemetry = FailOnShutdownTelemetry()
    monkeypatch.setattr("llm_future_affinity.application.OtelTelemetry", lambda config: telemetry)
    result = await execute_plan(
        loaded,
        "test-model",
        plan,
        disable_otel=False,
        debug=False,
        show_progress=False,
        api_key="secret",
    )
    assert not result.successful
    assert result.reason == "otel_export_failure"
    assert len(read_output(plan.paths.output)) == 4
