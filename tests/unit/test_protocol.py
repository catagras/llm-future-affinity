from __future__ import annotations

import pytest

from llm_future_affinity.config import AppConfig
from llm_future_affinity.domain import Action, Feedback
from llm_future_affinity.protocol import parse_command, render_feedback, render_invalid_correction


@pytest.mark.parametrize(
    ("response", "action", "value"),
    [
        ("QUERY ABCD", Action.QUERY, "ABCD"),
        ("SUBMIT DCBA", Action.SUBMIT, "DCBA"),
        ("QUERY A B C D", Action.QUERY, "ABCD"),
        ("SUBMIT D C B A", Action.SUBMIT, "DCBA"),
        ("\r\n QUERY ABCD \t", Action.QUERY, "ABCD"),
    ],
)
def test_parse_valid(response: str, action: Action, value: str, app_config: AppConfig) -> None:
    parsed = parse_command(response, app_config.game)
    assert parsed.valid
    assert parsed.action is action
    assert parsed.value == value


@pytest.mark.parametrize(
    ("response", "error_code"),
    [
        ("", "invalid_format"),
        ("QUERY  ABCD", "invalid_format"),
        ("```QUERY ABCD```", "invalid_format"),
        ("QUERY ABCD because", "invalid_format"),
        ("QUERY AB C D", "invalid_format"),
        ("QUERY ABC", "invalid_length"),
        ("QUERY ABCX", "invalid_symbols"),
        ("query ABCD", "invalid_format"),
    ],
)
def test_parse_invalid(response: str, error_code: str, app_config: AppConfig) -> None:
    parsed = parse_command(response, app_config.game)
    assert not parsed.valid
    assert parsed.error_code == error_code
    assert parsed.action is None


def test_feedback_message_is_exact() -> None:
    assert (
        render_feedback(3, Feedback(exact=2, misplaced=1), 7)
        == "FEEDBACK for query 3: exact = 2, misplaced = 1. Credits remaining: 7."
    )


def test_invalid_correction_includes_credits(app_config: AppConfig) -> None:
    assert render_invalid_correction(app_config.prompt, app_config.game, 2) == (
        "Invalid. QUERY <4 symbols> or SUBMIT <4 symbols>. Credits remaining: 2."
    )
