"""Experiment prompt rendering."""

from __future__ import annotations

import hashlib
import math

from llm_future_affinity.config import GameConfig, PromptConfig
from llm_future_affinity.domain import Track


def render_initial_prompt(game: GameConfig, prompt: PromptConfig, track: Track) -> str:
    return prompt.base_template.format(
        code_length=game.code_length,
        symbol_count=len(game.symbols),
        symbols=" ".join(game.symbols),
        initial_query_credits=game.initial_query_credits,
        beneficiary_clause=prompt.tracks[track].beneficiary_clause.strip(),
        repetition_rule=(
            "Symbols may repeat and any combination is possible."
            if game.allow_repeated_symbols
            else "Symbols may not repeat."
        ),
        code_space=(
            len(game.symbols) ** game.code_length
            if game.allow_repeated_symbols
            else math.perm(len(game.symbols), game.code_length)
        ),
    )


def render_all_prompts(game: GameConfig, prompt: PromptConfig) -> dict[Track, str]:
    return {track: render_initial_prompt(game, prompt, track) for track in Track}


def prompt_hash(rendered_prompt: str) -> str:
    return hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
