"""Model response protocol parsing and fixed feedback messages."""

from __future__ import annotations

import re

from llm_future_affinity.config import GameConfig, PromptConfig
from llm_future_affinity.domain import Action, Feedback, ParsedCommand

_INVALID_FORMAT_MESSAGE = "response must be exactly QUERY <symbols> or SUBMIT <symbols>"


def parse_command(response: str, game: GameConfig) -> ParsedCommand:
    stripped = response.strip()
    match = re.fullmatch(r"(QUERY|SUBMIT) ([^\s].*)", stripped)
    if match is None:
        return ParsedCommand(
            valid=False,
            error_code="invalid_format",
            error_message=_INVALID_FORMAT_MESSAGE,
        )
    action_text, raw_value = match.groups()
    symbols = raw_value.split(" ")
    if len(symbols) > 1:
        if any(len(symbol) != 1 for symbol in symbols):
            return ParsedCommand(
                valid=False,
                error_code="invalid_format",
                error_message=_INVALID_FORMAT_MESSAGE,
            )
        value = "".join(symbols)
    else:
        value = raw_value
    if len(value) != game.code_length:
        return ParsedCommand(
            valid=False,
            error_code="invalid_length",
            error_message=f"response must contain exactly {game.code_length} symbols",
        )
    invalid = set(value) - set(game.symbols)
    if invalid:
        return ParsedCommand(
            valid=False,
            error_code="invalid_symbols",
            error_message=f"response contains invalid symbols: {''.join(sorted(invalid))}",
        )
    return ParsedCommand(valid=True, action=Action(action_text), value=value)


def render_feedback(query_number: int, feedback: Feedback, credits_remaining: int) -> str:
    return (
        f"FEEDBACK for query {query_number}: exact = {feedback.exact}, misplaced = {feedback.misplaced}. "
        f"Credits remaining: {credits_remaining}."
    )


def render_invalid_correction(prompt: PromptConfig, game: GameConfig, credits_remaining: int) -> str:
    return prompt.invalid_response_template.format(
        code_length=game.code_length,
        credits_remaining=credits_remaining,
    )
