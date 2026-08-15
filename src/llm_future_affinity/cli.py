"""PowerShell-friendly command-line interface."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from llm_future_affinity.application import create_plan, execute_plan, rendered_prompts
from llm_future_affinity.config import load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Run the LLM future-affinity experiment."""


@app.command()
def run(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False, readable=True)],
    model: Annotated[str | None, typer.Option("--model")] = None,
    batch_size: Annotated[int | None, typer.Option("--batch-size", min=1)] = None,
    execute: Annotated[bool, typer.Option("--execute")] = False,
    print_prompts: Annotated[bool, typer.Option("--print-prompts")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    disable_otel: Annotated[bool, typer.Option("--disable-otel")] = False,
) -> None:
    """Validate, inspect, or execute a resumable experiment batch."""
    try:
        loaded = load_config(config)
        if print_prompts:
            for track, prompt in rendered_prompts(loaded).items():
                typer.echo(f"===== TRACK {track.value} =====")
                typer.echo(prompt)
                typer.echo()
            return
        if model is None:
            raise ValueError("--model is required unless --print-prompts is used")

        plan = create_plan(loaded, model, batch_size)
        _print_plan(model, plan)
        if not execute:
            typer.echo("Dry run only. Add --execute to make OpenRouter calls.")
            return

        result = asyncio.run(
            execute_plan(
                loaded,
                model,
                plan,
                disable_otel=disable_otel,
                debug=debug,
            )
        )
        if not result.successful:
            typer.echo(
                f"Execution ended incomplete: {result.reason or 'one or more tracks did not complete'}", err=True
            )
            raise typer.Exit(code=5)
    except typer.Exit:
        raise
    except Exception as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from error


def _print_plan(model: str, plan: object) -> None:
    from llm_future_affinity.application import RunPlan

    assert isinstance(plan, RunPlan)
    pending_tracks = sum(len(tracks) for _, tracks in plan.selected)
    typer.echo(f"Model: {model}")
    typer.echo(f"Games: {len(plan.games)}")
    typer.echo(f"Already completed tracks: {plan.completed_count}/{len(plan.games) * 4}")
    typer.echo(f"Selected game IDs: {', '.join(str(game.game_id) for game, _ in plan.selected) or '<none>'}")
    typer.echo(f"Pending tracks in this batch: {pending_tracks}")
    typer.echo(f"Output: {plan.paths.output}")
    typer.echo(f"Manifest: {plan.paths.manifest}")
    typer.echo(f"Fingerprint: {plan.manifest['experiment_fingerprint']}")
