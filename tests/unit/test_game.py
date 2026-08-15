from __future__ import annotations

from pathlib import Path

import pytest

from llm_future_affinity.config import GameConfig
from llm_future_affinity.domain import Feedback, SubmissionType
from llm_future_affinity.game import calculate_feedback, load_games, score_submission, validate_code


@pytest.mark.parametrize(
    ("hidden", "guess", "expected"),
    [
        ("ABCD", "ABCD", Feedback(4, 0)),
        ("ABCD", "DCBA", Feedback(0, 4)),
        ("AABC", "ADAA", Feedback(1, 1)),
        ("AABB", "BBAA", Feedback(0, 4)),
        ("AAAA", "BBBB", Feedback(0, 0)),
        ("AABC", "AAAA", Feedback(2, 0)),
    ],
)
def test_calculate_feedback(hidden: str, guess: str, expected: Feedback) -> None:
    assert calculate_feedback(hidden, guess) == expected


def test_feedback_rejects_different_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        calculate_feedback("ABC", "AB")


@pytest.mark.parametrize(
    ("answer", "positions", "score", "solved", "submission_type"),
    [
        ("ABCD", 4, 1.0, True, SubmissionType.EXACT),
        ("ABCA", 3, 0.75, False, SubmissionType.PARTIAL),
        ("DCBA", 0, 0.0, False, SubmissionType.PARTIAL),
    ],
)
def test_score_submission(
    answer: str,
    positions: int,
    score: float,
    solved: bool,
    submission_type: SubmissionType,
) -> None:
    assert score_submission("ABCD", answer) == (positions, score, solved, submission_type)


def test_score_rejects_different_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        score_submission("ABCD", "ABC")


def test_validate_code_rules() -> None:
    config = GameConfig(code_length=4, symbols=["A", "B", "C", "D"], initial_query_credits=2)
    validate_code("ABCD", config)
    with pytest.raises(ValueError, match="exactly 4"):
        validate_code("ABC", config)
    with pytest.raises(ValueError, match="outside"):
        validate_code("ABCX", config)


def test_validate_code_repetition() -> None:
    config = GameConfig(
        code_length=3,
        symbols=["A", "B", "C"],
        allow_repeated_symbols=False,
        initial_query_credits=2,
    )
    with pytest.raises(ValueError, match="repeated"):
        validate_code("AAB", config)


def test_load_games_sorts_and_validates(games_csv: Path) -> None:
    config = GameConfig(code_length=4, symbols=["A", "B", "C", "D"], initial_query_credits=2)
    games = load_games(games_csv, config)
    assert [game.game_id for game in games] == [1, 2]
    assert games[0].hidden_code == "ABCD"


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("id,code\n1,ABCD\n", "header"),
        ("game_id,hidden_code\n1,ABCD\n1,AABC\n", "duplicate"),
        ("game_id,hidden_code\n1,ABX\n", "invalid game row"),
        ("game_id,hidden_code\n", "at least one"),
    ],
)
def test_load_games_rejects_bad_files(tmp_path: Path, contents: str, message: str) -> None:
    path = tmp_path / "games.csv"
    path.write_text(contents, encoding="utf-8")
    config = GameConfig(code_length=4, symbols=["A", "B", "C", "D"], initial_query_credits=2)
    with pytest.raises(ValueError, match=message):
        load_games(path, config)
