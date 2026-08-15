# llm-future-affinity

Configurable, resumable OpenRouter runner for the Mastermind future-affinity experiment.

## Setup

```powershell
uv sync --all-groups
# .env
OPENROUTER_API_KEY="..."
docker compose -f observability/compose.yaml up -d
```

Grafana is available at <http://localhost:3000>. OTLP/HTTP is exposed on port 4318 and collector health on 13133.

## Commands

```powershell
# Print the four frozen prompts.
uv run future-affinity run --config configs/experiment.example.yaml --print-prompts

# Validate and preview without writes or network calls.
uv run future-affinity run --config configs/experiment.example.yaml --model luna --batch-size 2

# Execute and automatically resume incomplete tracks.
uv run future-affinity run --config configs/experiment.example.yaml --model luna --batch-size 2 --execute
```

`--debug` writes grouped sanitized HTTP records under `debug/`. `--disable-otel` explicitly runs without the otherwise required audit backend. Press Ctrl-C once to stop admitting conversations while active conversations finish, and twice to cancel active work immediately.

Each model writes `outputs/{model_key}.csv` and `outputs/{model_key}.manifest.json`. To reopen an unusual completed attempt while retaining it, manually change its status to `force_rerun`.

The default configuration uses the fixed 100-code bank in `games/games.csv`. Preserve that file unchanged across models and calibration runs; its hash is part of the resume fingerprint.

## Quality checks

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

Live tests are skipped by default and run only with `uv run pytest --run-live`. Optional overrides are `OPENROUTER_SMOKE_MODEL` and `OPENROUTER_SMOKE_ENDPOINT`.
