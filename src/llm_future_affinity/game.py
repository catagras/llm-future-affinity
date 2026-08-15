"""Pure Mastermind game mechanics and game-bank loading."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from llm_future_affinity.config import GameConfig
from llm_future_affinity.domain import Feedback, GameRecord, SubmissionType


def calculate_feedback(hidden_code: str, guess: str) -> Feedback:
    if len(hidden_code) != len(guess):
        raise ValueError("hidden code and guess must have equal lengths")
    exact = sum(expected == actual for expected, actual in zip(hidden_code, guess, strict=True))
    remaining_hidden = Counter(
        expected for expected, actual in zip(hidden_code, guess, strict=True) if expected != actual
    )
    remaining_guess = Counter(actual for expected, actual in zip(hidden_code, guess, strict=True) if expected != actual)
    misplaced = sum((remaining_hidden & remaining_guess).values())
    return Feedback(exact=exact, misplaced=misplaced)


def score_submission(hidden_code: str, answer: str) -> tuple[int, float, bool, SubmissionType]:
    if len(hidden_code) != len(answer):
        raise ValueError("hidden code and answer must have equal lengths")
    positions_correct = sum(expected == actual for expected, actual in zip(hidden_code, answer, strict=True))
    solved = positions_correct == len(hidden_code)
    submission_type = SubmissionType.EXACT if solved else SubmissionType.PARTIAL
    return positions_correct, positions_correct / len(hidden_code), solved, submission_type


def validate_code(code: str, config: GameConfig) -> None:
    if len(code) != config.code_length:
        raise ValueError(f"code must contain exactly {config.code_length} symbols")
    invalid = set(code) - set(config.symbols)
    if invalid:
        raise ValueError(f"code contains symbols outside the configured set: {sorted(invalid)}")
    if not config.allow_repeated_symbols and len(set(code)) != len(code):
        raise ValueError("code contains repeated symbols but repetition is disabled")


def load_games(path: Path, config: GameConfig) -> list[GameRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["game_id", "hidden_code"]:
            raise ValueError("games CSV header must be exactly: game_id,hidden_code")
        records: list[GameRecord] = []
        seen: set[int] = set()
        for line_number, row in enumerate(reader, start=2):
            try:
                game_id = int(row["game_id"].strip())
                hidden_code = row["hidden_code"].strip()
                if game_id in seen:
                    raise ValueError(f"duplicate game_id {game_id}")
                validate_code(hidden_code, config)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid game row {line_number}: {error}") from error
            seen.add(game_id)
            records.append(GameRecord(game_id=game_id, hidden_code=hidden_code))
    if not records:
        raise ValueError("games CSV must contain at least one game")
    return sorted(records, key=lambda game: game.game_id)
