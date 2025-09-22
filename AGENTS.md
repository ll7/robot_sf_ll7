# Repository Guidelines

## Project Structure & Module Organization
Core simulation code lives in `robot_sf/` with key subpackages: `gym_env` for Gymnasium bindings, `sim` for physics glue, `nav` for path planning, and `render` for playback tooling. Training and evaluation entry points sit in `scripts/`, while curated demos and notebooks live under `examples/`. Tests are split between `tests/` (unit and integration), `test_pygame/` (GUI regressions), and the `fast-pysf/` submodule, which must be initialized via `git submodule update --init --recursive`. Assets and checkpoints are versioned under `maps/svg_maps/`, `model/`, and `results/`.

## Build, Test & Development Commands
Set up dependencies with `uv sync --extra dev` and install hooks via `uv run pre-commit install`. Format and lint using `uv run ruff check .` followed by `uv run ruff format .`. Run the main suite with `uv run pytest tests`; add `-m "not slow"` to skip long benches. Headless GUI checks use `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest test_pygame`. Validate the SocialForce backend with `uv run python -m pytest fast-pysf/tests -v`. Typical training workflows call `uv run python scripts/training_ppo.py --config configs/scenarios/classic_interactions.yaml`.

## Coding Style & Naming Conventions
The project enforces Ruff with a 4-space indent, 100-character lines, and double-quoted strings (`pyproject.toml`). Prefer type-annotated interfaces and keep factory functions (`environment_factory.make_*`) as the public entry point. Modules and files use `snake_case`; classes and dataclasses follow `PascalCase`. Name tests `test_<feature>.py` and keep fixtures under `conftest.py`. Avoid ad-hoc prints in library code—use the existing structured logging.

## Testing Guidelines
Target the full `tests/` suite before pushing changes and rerun targeted slow markers when behavior or performance may shift. GUI and physics suites are mandatory for changes touching rendering, SocialForce integration, or pedestrian dynamics. Record notable validation runs with committed artifacts in `results/` when benchmarks change. Update or add smoke tests under `scripts/validation/` when introducing new critical workflows.

## Commit & Pull Request Workflow
Adopt the conventional commit style seen in history (e.g., `refactor: adjust observation scaling`). Each PR should summarize intent, reference related issues, and list the commands you ran. Include screenshots or short GIFs when UI or playback output changes, and note any new assets placed under `maps/` or `model/`. Ensure CI stays green by syncing with `main` and resolving lint or test failures locally before requesting review.
