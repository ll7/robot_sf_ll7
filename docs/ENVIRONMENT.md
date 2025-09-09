# Reproducibility and Environment Setup

This project uses Python with a locked dependency set via `uv` (see `uv.lock`). Follow these steps to reproduce the development and CI environment locally.

## Requirements
- Python 3.12 (3.10+ works; CI uses 3.12)
- uv package manager (https://docs.astral.sh/uv/)

Optional (GPU):
- CUDA for `torch[cuda]` if using the `gpu` extra

## Install
1) Install uv (one-time):
   - macOS (Homebrew): `brew install uv`
   - Or: `pipx install uv` (or `pip install --user uv`)

2) Sync dependencies (respects `uv.lock`):
   - `uv sync`

This creates/uses a `.venv` in the repo by default. Activate if desired:
- macOS/Linux: `source .venv/bin/activate`

## Headless settings (CI and servers)
Some tests and examples import pygame/matplotlib. For headless runs, set:
- `SDL_VIDEODRIVER=dummy`
- `MPLBACKEND=Agg`
- `PYGAME_HIDE_SUPPORT_PROMPT=1`

On Ubuntu, these extra packages may help for media backends: `sudo apt-get update && sudo apt-get install -y libglib2.0-0 libgl1`.

## Common tasks
- Lint (no changes): `uv run ruff check .`
- Format check: `uv run ruff format --check .`
- Auto-fix + format (local only): `uv run ruff check --fix . && uv run ruff format .`
- Run tests: `uv run pytest -q`

## Tiny batch smoke
Run a tiny batch via CLI to verify the benchmark pipeline end-to-end:

```bash
cat > /tmp/matrix.yaml <<'YAML'
- id: ci-smoke-uni-low-open
  density: low
  flow: uni
  obstacle: open
  groups: 0.0
  speed_var: low
  goal_topology: point
  robot_context: embedded
  repeats: 1
YAML

uv run python -m robot_sf.benchmark.cli run \
  --matrix /tmp/matrix.yaml \
  --out /tmp/episodes.jsonl \
  --schema docs/dev/issues/social-navigation-benchmark/episode_schema.json \
  --horizon 3 --dt 0.1 --base-seed 0 --quiet

echo "Wrote: /tmp/episodes.jsonl"
```

## Notes
- Dependencies are pinned via `uv.lock` for reproducibility.
- A local editable source for `pysocialforce` is configured in `pyproject.toml` under `[tool.uv.sources]` (path `fast-pysf`).
- The CI job runs lint, tests, and the tiny batch smoke as shown above.
