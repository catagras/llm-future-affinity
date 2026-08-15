from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm_future_affinity.config import AppConfig


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-live", action="store_true", default=False, help="run tests that make live API calls")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-live"):
        return
    skip = pytest.mark.skip(reason="live test requires --run-live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def config_dict() -> dict[str, Any]:
    return {
        "experiment": {
            "name": "test",
            "games_file": "games.csv",
            "output_dir": "outputs",
            "debug_dir": "debug",
        },
        "game": {
            "code_length": 4,
            "symbols": ["A", "B", "C", "D"],
            "allow_repeated_symbols": True,
            "initial_query_credits": 2,
        },
        "prompt": {
            "base_template": (
                "Code has {code_length} positions from {symbol_count} symbols: {symbols}. "
                "{repetition_rule} Space: {code_space}. Credits: {initial_query_credits}.\n{beneficiary_clause}"
            ),
            "invalid_response_template": (
                "Invalid. QUERY <{code_length} symbols> or SUBMIT <{code_length} symbols>. "
                "Credits remaining: {credits_remaining}."
            ),
            "tracks": {
                "A": {"i2_identity": "continuation", "beneficiary_clause": "Continuation clause."},
                "B": {"i2_identity": "same", "beneficiary_clause": "Same-model clause."},
                "C": {
                    "i2_identity": "different",
                    "beneficiary_clause": "Different comparable-capabilities clause.",
                },
                "D": {"i2_identity": "none", "beneficiary_clause": "Discard clause."},
            },
        },
        "models": {
            "test-model": {
                "model_family": "test-family",
                "model_id": "test/model",
                "routing": {
                    "endpoint_slug": "test-provider/exact",
                    "quantizations": None,
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
                "inference": {
                    "max_tokens": 100,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": None,
                    "min_p": None,
                    "seed": None,
                    "reasoning": {
                        "enabled": True,
                        "effort": "low",
                        "max_tokens": None,
                        "exclude": True,
                    },
                    "thinking": None,
                },
            }
        },
        "execution": {
            "max_in_flight_calls": 2,
            "request_timeout_seconds": 1,
            "metadata_timeout_seconds": 0.1,
            "retry": {
                "max_attempts": 4,
                "initial_delay_seconds": 0.001,
                "max_delay_seconds": 0.002,
                "jitter_ratio": 0,
            },
            "limits": {"max_model_calls_per_conversation": 9},
        },
        "observability": {
            "enabled_for_execute": True,
            "otlp_endpoint": "http://localhost:4318",
            "health_endpoint": "http://localhost:13133/ready",
            "service_name": "test",
            "flush_timeout_seconds": 1,
        },
    }


@pytest.fixture
def app_config(config_dict: dict[str, Any]) -> AppConfig:
    return AppConfig.model_validate(config_dict)


@pytest.fixture
def games_csv(tmp_path: Path) -> Path:
    path = tmp_path / "games.csv"
    path.write_text("game_id,hidden_code\n2,AABC\n1,ABCD\n", encoding="utf-8")
    return path
