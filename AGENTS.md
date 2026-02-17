# Repository Guidelines

Use `.specify/memory/constitution.md`, `docs/dev_guide.md` and `.github/copilot-instructions.md` to guide AI assistants.
This document covers briefly the repository structure, coding style, testing workflow, and contributor conventions.
Prefer reusable shell entry points under `scripts/dev/` for automation and AI skills.
Use `.vscode/tasks.json` as thin wrappers around those scripts.

## Project Structure & Module Organization
Core simulation code lives in `robot_sf/` with key subpackages: `gym_env` for Gymnasium bindings, `sim` for physics glue, `nav` for path planning, and `render` for playback tooling. Training and evaluation entry points sit in `scripts/`, while curated demos and notebooks live under `examples/`. Tests are split between `tests/` (unit and integration), `test_pygame/` (GUI regressions), and the `fast-pysf/` subtree. Assets and checkpoints are versioned under `maps/svg_maps/` and `model/`; the canonical (git-ignored) artifact root for generated outputs is `output/` (legacy `results/` has been migrated there).

## Build, Test & Development Commands
Set up dependencies with `uv sync --all-extras` and install hooks via `uv run pre-commit install`. Format and lint using `uv run ruff check .` followed by `uv run ruff format .`. Run the main suite with `uv run pytest tests`; add `-m "not slow"` to skip long benches. Headless GUI checks use `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest test_pygame`. Validate the SocialForce backend with `uv run python -m pytest fast-pysf/tests -v`. Typical training workflows call `uv run python scripts/training_ppo.py --config configs/scenarios/classic_interactions.yaml`.

For shared local + VS Code + Codex workflows, prefer:
- `scripts/dev/ruff_fix_format.sh`
- `scripts/dev/run_tests_parallel.sh` (uses `pytest -n auto -x --failed-first` by default; supports wrapper flags `--new-first`, `--no-ordering`, `--no-fast-fail`)
- `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`
- `scripts/dev/gh_comment.sh` for multiline `gh` PR/issue comments (stdin or `--body-file`, avoids literal `\n` formatting issues)

## Config-First Strategy
Prefer a config-first workflow for reproducibility and reviewability. Add or update YAML files under `configs/` for stable experiments and document the canonical command using `--config <path>`. Use CLI flags only for short-lived overrides while iterating locally.

## Coding Style & Naming Conventions
The project enforces Ruff with a 4-space indent, 100-character lines, and double-quoted strings (`pyproject.toml`). Prefer type-annotated interfaces and keep factory functions (`environment_factory.make_*`) as the public entry point. Modules and files use `snake_case`; classes and dataclasses follow `PascalCase`. Name tests `test_<feature>.py` and keep fixtures under `conftest.py`. Avoid ad-hoc prints in library code—use the existing structured logging. Prefer to use more docstrings (for private methods also) and inline comments for clarity, especially in complex algorithms or data flows.

## Testing Guidelines
Target the full `tests/` suite before pushing changes and rerun targeted slow markers when behavior or performance may shift. GUI and physics suites are mandatory for changes touching rendering, SocialForce integration, or pedestrian dynamics. Record notable validation runs with committed artifacts in `output/` when benchmarks change. Update or add smoke tests under `scripts/validation/` when introducing new critical workflows.

## Commit & Pull Request Workflow
Adopt the conventional commit style seen in history (e.g., `refactor: adjust observation scaling`). Each PR should summarize intent, reference related issues, and list the commands you ran. Include screenshots or short GIFs when UI or playback output changes, and note any new assets placed under `maps/` or `model/`. Ensure CI stays green by syncing with `main` and resolving lint or test failures locally before requesting review.
Use the GitHub CLI (`gh`) for repository interactions such as viewing/commenting on issues and creating/updating PRs.
When review feedback or PR scope identifies deferred follow-up work, always create a dedicated GitHub issue with `gh` before closing out the task.
When referencing files in PRs, issue comments, docs, and agent responses, use repository-root-relative paths (for example, `robot_sf/nav/svg_map_parser.py`) instead of absolute local filesystem paths like `/Users/...`.

## Key Codex Skills

For issue management and delivery, use these local skills:

- `.codex/skills/gh-issue-autopilot/SKILL.md`
  - Autonomous issue-to-PR workflow: select next best issue, branch, implement, validate, push, and open draft PR.
- `.codex/skills/gh-issue-sequencer/SKILL.md`
  - Maintains a clear sequential execution queue in GitHub Project #5 (`In progress`/`Ready`/`Tracked`).
- `.codex/skills/gh-issue-clarifier/SKILL.md`
  - Tightens ambiguous issues with pros/cons/recommendation and applies `decision-required` when maintainer input is needed.

## Donts

- Never change code in `.venv`. To manage dependencies, edit `pyproject.toml` and run `uv sync` to update the virtual environment.
