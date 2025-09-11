# Robot SF – Copilot Guide

Use this as the single source of truth for this repo. Prefer these rules over generic habits; fall back to ad‑hoc searches or shell when something is clearly out of sync.

## Executive summary
- Clarify first, then code: ask concise, optioned questions; propose sensible defaults; don’t block on non‑essentials.
- Design doc first for non‑trivial work; always add tests with code changes; keep the codebase ruff‑clean.
- Use uv + Ruff + pytest and the provided VS Code tasks; run quality gates before pushing.
- Always create environments via factory functions (no direct instantiation).
- Follow Git best practices: small PRs, meaningful commits with issue numbers, and rebased branches.

## Table of contents
- Core principles
- Tooling and tasks (uv, Ruff, pytest, VS Code)
- Git workflow (branches, commits, PRs)
- Testing policy and how to run tests
- Documentation standards and design docs
- Environment setup and usage (factory pattern)
- CI/CD expectations
- Validation scenarios and performance
- Training and examples (incl. Docker)
- Common issues and solutions
- Migration notes
- Helpful definitions and repository structure
- Quick reference and TL;DR checklist

---

## Core principles
### Code quality standards
- Clear, intent‑revealing names; small, cohesive functions; robust error handling.
- Follow existing style; document non‑obvious choices with comments/docstrings.
- Avoid duplication; prefer composition and reuse.
- Keep public behavior backward‑compatible unless explicitly stated.
- Write comprehensive unit tests for new features and bug fixes (GUI tests in `test_pygame/`).

### Always ask clarifying questions (with options)
- Before implementing, confirm requirements with targeted questions.
- Prefer multiple‑choice options to speed decisions; group by scope, interfaces, data, UX, performance.
- If answers are unknown, propose sensible defaults and proceed (don’t block on non‑essentials).

### Problem‑solving approach
- Break problems into smaller tasks; research prior art and patterns.
- Consider system‑wide impact, edge cases, error handling, and failure modes.
- Document architectural decisions and trade‑offs.

---

## Tooling and tasks (uv, Ruff, pytest, VS Code)
- Dependencies/runtime: uv
  - Install/resolve: VS Code task “Install Dependencies” (uv sync)
  - Run: `uv run <cmd>` for any Python command
  - Add deps: `uv add <package>` (or edit `pyproject.toml` and sync)
- Lint/format: Ruff
  - VS Code task “Ruff: Format and Fix” (keeps repo ruff‑clean; document exceptions with comments)
- Tests: pytest
  - VS Code task “Run Tests” (default suite)
  - “Run Tests (Show All Warnings)” for diagnostics
  - “Run Tests (GUI)” for display‑dependent tests (headless via environment vars)
- Code quality checks: VS Code task “Check Code Quality” (Ruff + pylint errors‑only)
- Diagrams: VS Code task “Generate UML”

Quality gates to run locally before pushing:
1) Install Dependencies → 2) Ruff: Format and Fix → 3) Check Code Quality → 4) Run Tests

Shortcuts (optional shell):
```bash
uv run ruff check . && uv run ruff format .
uv run pytest tests
```

---

## Git workflow (branches, commits, PRs)
- Branch naming:
  - `feature/42-fix-button-alignment`
  - `bugfix/89-memory-leak-in-simulator`
  - `enhancement/156-improve-lidar-performance`
- Commits (include issue number):
  - `fix: resolve 2x speed multiplier in VisualizableSimState (#42)`
  - `feat: add new lidar sensor configuration options (#156)`
  - `docs: update installation guide with GPU setup instructions`
  - `test: add comprehensive integration tests for pedestrian simulation`
- Keep branches rebased with `main`.
- Use PRs for review; run full test suite before merging.

---

## Testing policy and how to run tests
### Policy
- Write tests for all new features and bug fixes.
- Unit/integration tests in `tests/`; GUI‑dependent tests in `test_pygame/`.
- Deterministic tests: seed RNGs, use fixtures, avoid time/network flakiness.
- Add regression tests to prevent reintroduction of bugs.
- For perf‑sensitive code, add simple benchmarks or timing asserts (document variability).

### Running tests
```bash
# Main suite (2–3 min)
uv run pytest tests

# GUI tests (1–2 min; headless)
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest test_pygame

# fast-pysf submodule tests (2 tests may fail due to missing map files)
uv run python -m pytest fast-pysf/tests/ -v
```

---

## Documentation standards and design docs
### Technical documentation
- Place docs in `docs/` with a clear folder structure (kebab‑case per major feature/issue).
- Each folder has a README.md covering: problem, solution overview, implementation details, impact analysis, testing, future considerations, related links.
- Use diagrams/screenshots when helpful; keep docs up‑to‑date.

### Design doc first (required for non‑trivial work)
Create under `docs/dev/issues/<topic>/README.md` (or issue‑specific folder). Include:
- Context (link issue/PR), goals and non‑goals
- Constraints/assumptions; options and trade‑offs
- Chosen approach (diagram if helpful)
- Data shapes/APIs/contracts (inputs/outputs, error modes)
- Test plan (unit/integration, GUI if needed)
- Rollout/back‑compat notes; metrics/observability
- Open questions and follow‑ups

---

## Environment setup and usage (factory pattern)
### Installation and setup
```bash
# One‑time
git submodule update --init --recursive
uv sync && source .venv/bin/activate

# Dev extras and pre‑commit (optional)
uv sync --extra dev
uv run pre-commit install

# Quick import check
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; print('Import successful')"
```

### Environment creation (always use factory functions)
```python
from robot_sf.gym_env.environment_factory import (
    make_robot_env,
    make_image_robot_env,
    make_pedestrian_env,
)

# Basic robot environment
env = make_robot_env(debug=True)

# Robot with image observations
env = make_image_robot_env(debug=True)

# Pedestrian environment (requires trained robot model)
env = make_pedestrian_env(robot_model=model, debug=True)
```

### Configuration
```python
from robot_sf.gym_env.unified_config import (
    RobotSimulationConfig,
    ImageRobotConfig,
    PedestrianSimulationConfig,
)

config = RobotSimulationConfig()
config.peds_have_obstacle_forces = True
env = make_robot_env(config=config)
```

---

## CI/CD expectations
- Tests: `uv run pytest tests`
- Lint: `uv run ruff check .` and `uv run ruff format --check .`
- The pipeline mirrors the local quality gates. Ensure green locally first.

---

## Validation scenarios and performance
### Validation scenarios (run after changes)
```bash
./scripts/validation/test_basic_environment.sh
./scripts/validation/test_model_prediction.sh
./scripts/validation/test_complete_simulation.sh
```

### Performance benchmarking (optional)
```bash
# Run benchmark when performance impact is suspected
DISPLAY= MPLBACKEND=Agg uv run python scripts/benchmark02.py
# Expected: ~22 steps/second, ~45ms per step
```

### Performance expectations
- Environment creation: < 1 second
- Model loading: 1–5 seconds
- Simulation performance: ~22 steps/second (~45ms/step)
- Build time: 2–3 minutes (first time)
- Test suite: 2–3 minutes (≈170 tests)

---

## Training and examples
### Available demos
```bash
uv run python examples/demo_offensive.py
uv run python examples/demo_defensive.py
uv run python examples/demo_pedestrian.py
uv run python examples/demo_refactored_environments.py
```

### Training scripts
```bash
uv run python scripts/training_ppo.py
uv run python scripts/hparam_opt.py
uv run python scripts/evaluate.py
```

### Docker training (advanced)
```bash
# Build and run GPU training (requires NVIDIA Docker)
docker compose build && docker compose run robotsf-cuda python ./scripts/training_ppo.py
# NOTE: May fail in CI environments due to network restrictions
```

---

## Common issues and solutions
### Build issues
- uv not found → `pip install uv` (or use official installer)
- ffmpeg missing → `sudo apt-get install -y ffmpeg`
- Submodules empty → `git submodule update --init --recursive`

### Runtime issues
- Import errors → ensure venv is activated: `source .venv/bin/activate`
- Display errors → run headless: `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy`
- Model loading warnings → StableBaselines3 warnings about Gym are normal
- Model compatibility → use newest models for best compatibility (e.g., `ppo_model_retrained_10m_2025-02-01.zip`)

### Docker issues
- Docker build fails → often network related in CI; usually fine locally
- GPU support → see `docs/GPU_SETUP.md` for NVIDIA Docker setup

---

## Migration notes
- The project uses uv for env/runner and a factory pattern for environment creation.
- See `docs/UV_MIGRATION.md` and `docs/refactoring/` for details.

---

## Helpful definitions and repository structure
### Helpful definitions
- uv: Fast Python package/dependency manager and runner (`uv sync`, `uv run`).
- Ruff: Python linter and formatter (run via “Ruff: Format and Fix” task).
- pytest: Testing framework (run via “Run Tests” tasks).
- VS Code tasks: Standardized workflows (install, lint, test, diagram).
- Quality gates: Minimal checks before pushing (install → lint/format → quality check → tests).

### Repository structure (key dirs)
- `robot_sf/` (source), `examples/`, `tests/`, `test_pygame/`, `fast-pysf/` (submodule), `scripts/`, `model/`, `docs/`

---

## Quick reference and TL;DR checklist
### Quick reference commands
```bash
# Setup after installation
git submodule update --init --recursive && uv sync && source .venv/bin/activate

# Validate changes
uv run ruff check . && uv run ruff format . && uv run pytest tests

# Functional smoke (headless)
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; env = make_robot_env(); env.reset(); print('OK')"

# Optional perf benchmark
DISPLAY= MPLBACKEND=Agg uv run python scripts/benchmark02.py
```

### TL;DR workflow checklist
1) Clarify requirements (ask concise, optioned questions)
2) Draft design doc under `docs/` (link issue, add test plan)
3) Implement with small, reviewed commits
4) Add/extend tests in `tests/` or `test_pygame/`
5) Run quality gates via tasks: Install Dependencies → Ruff: Format and Fix → Check Code Quality → Run Tests
6) Update docs/diagrams; run “Generate UML” if classes changed
7) Open PR with summary, risks, and links to docs/tests