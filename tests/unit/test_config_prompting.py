from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from llm_future_affinity.config import AppConfig, load_config, validate_prompt_placeholders, without_none
from llm_future_affinity.domain import Track
from llm_future_affinity.prompting import prompt_hash, render_all_prompts, render_initial_prompt


def test_render_all_prompts_are_distinct_and_safe(app_config: AppConfig) -> None:
    prompts = render_all_prompts(app_config.game, app_config.prompt)
    assert set(prompts) == set(Track)
    assert "Continuation clause" in prompts[Track.A]
    assert "comparable-capabilities" in prompts[Track.C]
    for rendered in prompts.values():
        assert "ABCD" not in rendered
        assert "game_id" not in rendered
        assert "4 positions" in rendered


def test_render_initial_prompt_hash_is_stable(app_config: AppConfig) -> None:
    rendered = render_initial_prompt(app_config.game, app_config.prompt, Track.A)
    assert prompt_hash(rendered) == prompt_hash(rendered)
    assert prompt_hash(rendered) != prompt_hash(rendered + "x")


def test_render_prompt_describes_repetition_and_code_space(app_config: AppConfig) -> None:
    app_config.prompt.base_template += " {repetition_rule} Space: {code_space}."
    repeated = render_initial_prompt(app_config.game, app_config.prompt, Track.A)
    assert "Symbols may repeat" in repeated
    assert "Space: 256" in repeated

    app_config.game.allow_repeated_symbols = False
    unique = render_initial_prompt(app_config.game, app_config.prompt, Track.A)
    assert "Symbols may not repeat" in unique
    assert "Space: 24" in unique


def test_config_model_lookup(app_config: AppConfig) -> None:
    assert app_config.model_for("test-model").model_id == "test/model"
    assert app_config.model_for("test-model").rpm is None
    with pytest.raises(ValueError, match="available keys"):
        app_config.model_for("missing")


def test_model_rpm_is_optional_and_positive(config_dict: dict[str, Any]) -> None:
    config_dict["models"]["test-model"]["rpm"] = 9
    assert AppConfig.model_validate(config_dict).model_for("test-model").rpm == 9
    config_dict["models"]["test-model"]["rpm"] = 0
    with pytest.raises(ValidationError):
        AppConfig.model_validate(config_dict)


def test_reasoning_effort_and_budget_are_exclusive(config_dict: dict[str, Any]) -> None:
    raw = deepcopy(config_dict)
    raw["models"]["test-model"]["inference"]["reasoning"]["max_tokens"] = 10
    with pytest.raises(ValidationError, match="mutually exclusive"):
        AppConfig.model_validate(raw)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw["game"].update(symbols=["A", "A"]),
        lambda raw: raw["game"].update(symbols=["AA", "B"]),
        lambda raw: raw["prompt"]["tracks"].pop("D"),
        lambda raw: raw["models"]["test-model"]["routing"].update(allow_fallbacks=True),
    ],
)
def test_config_rejects_invalid_invariants(config_dict: dict[str, Any], mutation: Any) -> None:
    raw = deepcopy(config_dict)
    mutation(raw)
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_unknown_prompt_placeholder_is_rejected(app_config: AppConfig) -> None:
    app_config.prompt.base_template += " {hidden_code}"
    with pytest.raises(ValueError, match="hidden_code"):
        validate_prompt_placeholders(app_config.prompt)


def test_required_prompt_placeholder_is_enforced(app_config: AppConfig) -> None:
    app_config.prompt.base_template = app_config.prompt.base_template.replace("{beneficiary_clause}", "")
    with pytest.raises(ValueError, match="beneficiary_clause"):
        validate_prompt_placeholders(app_config.prompt)


def test_unsupported_nullable_inference_controls_may_be_omitted(config_dict: dict[str, Any]) -> None:
    raw = deepcopy(config_dict)
    raw["models"]["test-model"]["inference"].pop("temperature")
    raw["models"]["test-model"]["inference"].pop("top_k")
    config = AppConfig.model_validate(raw)
    inference = config.models["test-model"].inference
    assert inference.temperature is None
    assert inference.top_k is None


def test_retry_attempt_count_is_frozen(config_dict: dict[str, Any]) -> None:
    raw = deepcopy(config_dict)
    raw["execution"]["retry"]["max_attempts"] = 3
    with pytest.raises(ValidationError, match="max_attempts"):
        AppConfig.model_validate(raw)


def test_load_config_and_resolve(tmp_path: Path, config_dict: dict[str, Any]) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.source_path == path.resolve()
    assert loaded.resolve(Path("games.csv")) == (tmp_path / "games.csv").resolve()


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_without_none_recurses() -> None:
    assert without_none({"a": None, "b": {"c": 1, "d": None}, "e": [None, {"f": None}]}) == {
        "b": {"c": 1},
        "e": [None, {}],
    }
