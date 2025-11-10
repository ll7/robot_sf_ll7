# Reproducibility and Environment Setup

[← Back to Documentation Index](./README.md)

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

robot_sf_bench run \
  --matrix /tmp/matrix.yaml \
  --out /tmp/episodes.jsonl \
  --schema docs/dev/issues/social-navigation-benchmark/episode_schema.json \
  --horizon 3 --dt 0.1 --base-seed 0 --quiet

echo "Wrote: /tmp/episodes.jsonl"

# Optional: quick diversity summary plots
robot_sf_bench summary --in /tmp/episodes.jsonl --out-dir /tmp/figs
echo "Plots under /tmp/figs"
```

## SNQI Weight Tooling

Tools for recomputing and optimizing Social Navigation Quality Index (SNQI) weights live under `scripts/` and are documented in:

- [`docs/snqi-weight-tools/README.md`](./snqi-weight-tools/README.md)
- Design details: [`docs/dev/issues/snqi-recomputation/DESIGN.md`](./dev/issues/snqi-recomputation/DESIGN.md)

Use these after generating benchmark episode data (`robot_sf_bench run`) and baseline stats to derive or evaluate weight configurations. All outputs contain a reproducibility `_metadata` block (schema version, git commit, seed, invocation).

## Notes
- Dependencies are pinned via `uv.lock` for reproducibility.
- A local editable source for `pysocialforce` is configured in `pyproject.toml` under `[tool.uv.sources]` (path `fast-pysf`).
- The CI job runs lint, tests, and the tiny batch smoke as shown above.

### Pedestrian Density Reference
For guidance on choosing and validating pedestrian densities (units, canonical triad, advisory range, and test policy) see: [`ped_metrics/PEDESTRIAN_DENSITY.md`](./ped_metrics/PEDESTRIAN_DENSITY.md)

## Ergonomic Environment Factory Options (Feature 130)

Environment creation now supports structured option objects while retaining legacy convenience flags.

### Quick Examples
```python
from robot_sf.gym_env.environment_factory import (
  make_robot_env, make_image_robot_env,
  RecordingOptions, RenderOptions
)

# Minimal
env = make_robot_env()

# Recording via convenience boolean
env = make_robot_env(record_video=True, video_path="episode.mp4")

# Structured options (preferred for explicit control)
render_opts = RenderOptions(max_fps_override=30)
rec_opts = RecordingOptions(record=True, video_path="episode.mp4")
env = make_robot_env(render_options=render_opts, recording_options=rec_opts, debug=True)

# Image observations
img_env = make_image_robot_env(render_options=RenderOptions(max_fps_override=24))
```

### Precedence & Deprecation
- `record_video=True` overrides `RecordingOptions(record=False)` (warning emitted).
- `video_fps` maps to `RenderOptions.max_fps_override` unless that field already set.
- Legacy kwargs (`fps`, `video_output_path`) are mapped with warnings; unknown legacy kwargs raise unless `ROBOT_SF_FACTORY_LEGACY=1`.

### Logging
Each factory emits an INFO creation line with effective recording and fps settings plus WARNING lines for any deprecation or precedence events.

### Performance Guard
Test `tests/perf/test_factory_creation_perf.py` ensures mean creation time remains within the regression budget relative to `results/factory_perf_baseline.json`.

See detailed migration guidance in `docs/dev/issues/130-improve-environment-factory/migration.md`.
