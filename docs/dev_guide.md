# Robot SF – Development Guide

- [Setup](#setup)
  - [Installation and setup](#installation-and-setup)
  - [Critical dependencies and setup: Fast-pysf submodule (REQUIRED)](#critical-dependencies-and-setup-fast-pysf-submodule-required)
  - [Quick Start Commands](#quick-start-commands)
  - [Environment factory pattern (CRITICAL)](#environment-factory-pattern-critical)
  - [Key architectural layers](#key-architectural-layers)
  - [Data flow and integration](#data-flow-and-integration)
  - [Configuration hierarchy](#configuration-hierarchy)
- [Design and development workflow recommendations](#design-and-development-workflow-recommendations)
  - [Testing strategy (THREE test suites)](#testing-strategy-three-test-suites)
  - [Must-have checklist](#must-have-checklist)
  - [Optional backlog (track but don’t block)](#optional-backlog-track-but-dont-block)
  - [Quick links](#quick-links)
  - [Executive summary](#executive-summary)
  - [Code quality standards](#code-quality-standards)
  - [Design decisions](#design-decisions)
    - [CLI vs programmatic use](#cli-vs-programmatic-use)
  - [Code reviews](#code-reviews)
    - [Docstrings](#docstrings)
  - [Ask clarifying questions (with options)](#ask-clarifying-questions-with-options)
  - [Problem‑solving approach](#problemsolving-approach)
  - [Tooling and tasks (uv, Ruff, pytest, ty, VS Code)](#tooling-and-tasks-uv-ruff-pytest-ty-vs-code)
- [Documentation Standards](#documentation-standards)
  - [Technical Documentation](#technical-documentation)
  - [Documentation Content Requirements](#documentation-content-requirements)
  - [Documentation Best Practices](#documentation-best-practices)
- [Tooling and tasks (uv, Ruff, pytest, VS Code)](#tooling-and-tasks-uv-ruff-pytest-vs-code)
- [CI/CD expectations](#cicd-expectations)
- [Validation scenarios and performance](#validation-scenarios-and-performance)
  - [Validation scenarios (run after changes)](#validation-scenarios-run-after-changes)
  - [Performance benchmarking (optional)](#performance-benchmarking-optional)
  - [Performance expectations](#performance-expectations)
- [Training and examples](#training-and-examples)
  - [Available demos](#available-demos)
  - [Training scripts](#training-scripts)
  - [Docker training (advanced)](#docker-training-advanced)
- [Common issues and solutions](#common-issues-and-solutions)
  - [Build issues](#build-issues)
  - [Runtime issues](#runtime-issues)
  - [Docker issues](#docker-issues)
- [Migration notes](#migration-notes)
- [Helpful definitions and repository structure](#helpful-definitions-and-repository-structure)
  - [Helpful definitions](#helpful-definitions)
  - [Repository structure (key dirs)](#repository-structure-key-dirs)
- [Definition of Done (DoD)](#definition-of-done-dod)
- [Templates](#templates)
- [Security \& network policy](#security--network-policy)
- [Large files \& artifacts policy](#large-files--artifacts-policy)
- [Quick reference and TL;DR checklist](#quick-reference-and-tldr-checklist)
  - [Quick reference commands](#quick-reference-commands)
  - [TL;DR workflow checklist](#tldr-workflow-checklist)

## Setup

### Installation and setup
```bash
# One‑time
git submodule update --init --recursive
uv sync && source .venv/bin/activate

# Dev extras and pre‑commit (optional)
uv sync --all-extras
uv run pre-commit install

# Quick import check
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; print('Import successful')"
```

### Critical dependencies and setup: Fast-pysf submodule (REQUIRED)
**Always initialize submodules** after git clone or checkout:
```bash
git submodule update --init --recursive
```
Without this, pedestrian simulation will fail. The `fast-pysf/` directory contains optimized SocialForce physics.

### Quick Start Commands
```bash
# Lint+format
uv run ruff check . && uv run ruff format .
# Tests
uv run pytest tests
```

One‑liner quality gates (CLI):
```bash
uv run ruff check . && uv run ruff format . && uv run pylint robot_sf --errors-only && uvx ty check . --exit-zero && uv run pytest tests
```

### Environment factory pattern (CRITICAL)
**Always use factory functions** — never instantiate gym environments directly:
```python
from robot_sf.gym_env.environment_factory import make_robot_env, make_image_robot_env, make_pedestrian_env

# Basic robot navigation
env = make_robot_env(debug=True)

# With image observations  
env = make_image_robot_env(debug=True)

# Pedestrian environment (requires trained robot model)
env = make_pedestrian_env(robot_model=model, debug=True)
```

### Key architectural layers
- **`robot_sf/gym_env/`**: Gymnasium environment implementations with factory pattern
- **`robot_sf/baselines/`**: Baseline navigation algorithms (e.g., SocialForce) for benchmarking
- **`robot_sf/benchmark/`**: Benchmark runner, CLI, metrics collection, and schema validation
- **`robot_sf/sim/`**: Core simulation components (FastPysfWrapper for pedestrian physics)
- **`fast-pysf/`**: Git submodule providing optimized SocialForce pedestrian simulation

### Data flow and integration
- **Training loop**: `scripts/training_ppo.py` → factory functions → vectorized environments → StableBaselines3
- **Benchmarking**: `robot_sf/benchmark/cli.py` → baseline algorithms → episode runs → JSON/JSONL output → analysis
- **Pedestrian simulation**: Robot environments → FastPysfWrapper → `fast-pysf` submodule → NumPy/Numba physics

### Configuration hierarchy
Use unified config classes from `robot_sf.gym_env.unified_config`:
```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig, ImageRobotConfig

config = RobotSimulationConfig()
config.peds_have_obstacle_forces = True  # Enable ped-robot physics interaction
env = make_robot_env(config=config)
```

## Design and development workflow recommendations

- Clarify exact requirements before starting implementation.
- Ask clarifying questions (with options) to confirm scope, interfaces, data handling, UX, and performance.
  - Discuss possible options and trade-offs.
  - Give arguments to the options for easy decision-making.
  - Provide options to quickly converge on a decision.
- For complex tasks:
  - Create a design doc (see template below) for non-trivial changes.
  - Create a file based TODO list (see example below).
  - Break task down into smaller subtasks and tackle them iteratively.
- Prioritize must-haves over nice-to-haves
- Document assumptions and trade-offs.
- Prefer programmatic use and factory functions over CLI; the CLI is not important.
- Working mode: prioritize a thin, end-to-end slice that runs. Optimize and polish after a green smoke test (env reset→step loop or demo run).

- Architecture in one line: Gym/Gymnasium envs → factory functions → FastPysfWrapper → fast-pysf physics; training/eval via StableBaselines3; baselines/benchmarks under `robot_sf/baselines` and `robot_sf/benchmark`.
- Environments: always create via factories (`make_robot_env`, `make_image_robot_env`, `make_pedestrian_env`). Configure via `robot_sf.gym_env.unified_config` only; toggle flags before passing to the factory.
- Simulation glue: interact with pedestrian physics through `robot_sf/sim/FastPysfWrapper`. Don’t import from `fast-pysf` directly inside envs.
- Baselines/benchmarks: get planners with `robot_sf.baselines.get_baseline(...)`. Prefer programmatic runners; CLI exists at `robot_sf/benchmark/cli.py` for convenience.
- Demos/trainings: keep runnable examples in `examples/` and scripts in `scripts/`. Place models in `model/`, maps in `maps/svg_maps/`, and write outputs under `results/`.
- Tests: core in `tests/`; GUI in `test_pygame/` (headless: `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy`). Physics-specific tests live in `fast-pysf/tests/`.
- Quality gates (local): Install Dependencies → Ruff: Format and Fix → Check Code Quality → Type Check → Run Tests (see VS Code Tasks).
- Progress cadence: during complex edits, report what ran and outcomes after ~3–5 tool calls or when you edit > ~3 files in one burst.

### Testing strategy (THREE test suites)

```bash
# 1. Main unit/integration tests (2-3 min)
uv run pytest tests

# 2. GUI/display-dependent tests (headless mode)  
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest test_pygame

# 3. fast-pysf submodule tests (some may fail without map files)
uv run python -m pytest fast-pysf/tests/ -v
```

### Must-have checklist
- [ ] Initialize submodules after clone: `git submodule update --init --recursive` (needed for `fast-pysf/`).
- [ ] Use factory env creators; do not instantiate env classes directly.
- [ ] Set config via `robot_sf.gym_env.unified_config` before env creation; avoid ad‑hoc kwargs.
- [ ] Keep lib code print-free; use logging for info and warnings.
- [ ] Run VS Code Tasks: Install Dependencies, Ruff: Format and Fix, Check Code Quality, Type Check, Run Tests.
- [ ] Add a test or smoke (e.g., env reset/step) when you change public behavior.
- [ ] For GUI-dependent tests, set headless env vars; avoid flaky display usage in CI.
- [ ] Treat `fast-pysf/` as a submodule; don’t modify unless scoped and justified.
- [ ] Put new demos under `examples/` and new runners under `scripts/`.
- [ ] Whenever a demo is possible, add one.

### Optional backlog (track but don’t block)
- [ ] Tighten type hints for new public APIs; migrate call sites gradually.
- [ ] Add programmatic benchmark examples and extend baseline coverage.
- [ ] Update or add docs under `docs/` for new components; include diagrams when useful.
- [ ] Add performance smoke (steps/sec) when touching hot paths.

### Quick links
- Environment overview: `docs/ENVIRONMENT.md`
- Simulation view: `docs/SIM_VIEW.md`
- Refactoring and architecture notes: `docs/refactoring/`
- SNQI tools and metrics: `docs/snqi-weight-tools/README.md`
- Data analysis helpers: `docs/DATA_ANALYSIS.md`

### Executive summary
- **Architecture**: Social navigation RL framework with gym/gymnasium environments, SocialForce pedestrian simulation via `fast-pysf` submodule, StableBaselines3 training pipeline
- **Core pattern**: Factory-based environment creation (`make_robot_env()` etc.) — never instantiate environments directly
- **Dependencies**: `fast-pysf` git submodule for pedestrian physics; always run `git submodule update --init --recursive` after clone
- **Toolchain**: uv + Ruff + ty + pytest with VS Code tasks; run quality gates before pushing
- **Testing**: Unit tests in `tests/`, GUI-dependent tests in `test_pygame/` (with headless env vars), integration tests for smoke/performance validation
- **Documentation**: Comprehensive docs under `docs/` with design principles, architecture, usage, and migration notes
  - Development notes: `docs/dev/*`

### Code quality standards
- Clear, intent‑revealing names; small, cohesive functions; robust error handling.
- Follow existing style; document non‑obvious choices with comments/docstrings.
- Avoid duplication; prefer composition and reuse.
- Keep public behavior backward‑compatible unless explicitly stated.
- Write comprehensive unit tests for new features and bug fixes (GUI tests in `test_pygame/`).

### Design decisions
- Favor readability and maintainability over micro‑optimizations.
- Use type hints for all public functions and methods; prefer `typing` over `Any`.
- Use exceptions for error handling; avoid silent failures.

#### CLI vs programmatic use

- This project prioritizes traceability and reproducibility of benchmarks. Prefer generating script- and config-driven workflows over ad-hoc command lines or inline parameter tweaks.
- Do not focus on the cli directly; prefer programmatic use and factory functions.
- The CLI is not important; prefer programmatic use and factory functions.
- Use logging for non‑error informational messages; avoid print statements except in CLI entry points.

- Configs: configs/<area>/<name>.yaml (single source of truth for all hyperparameters, seeds, envs).
-	Scripts: scripts/<task>_<runner>.py (read config path, set up run dirs, log metadata, call library code).
-	Runs/outputs/results: results/<timestamp>_<shortname>/ (store config.yaml, git_meta.json, logs, metrics, artifacts).
-	Deterministic seed in both config and code

### Code reviews
- All changes must be reviewed by at least one other team member.
- Reviewers should check for correctness, style, test coverage, and documentation.
- Use GitHub’s review tools to leave comments and approve changes.

#### Docstrings
- Every public module, function, class, and method should have a docstring.
- Docstrings should use triple double quotes (""").
- The first line should be a short summary of the object’s purpose, starting with a capital letter and ending with a period.
- If more detail is needed, leave a blank line after the summary, then continue with a longer description.
- For functions/methods: document parameters, return values, exceptions raised, and side effects.

### Ask clarifying questions (with options)
- Before implementing, confirm requirements with targeted questions.
- Prefer multiple‑choice options to speed decisions; group by scope, interfaces, data, UX, performance.
- Add arguments to the options for easy decision-making.
- If answers are unknown, propose sensible defaults and proceed (don't block on non‑essentials).

Examples (copy‑ready):
- Scope: Is the metric per episode or a per‑timestep aggregate?
- Interfaces: Return shape `dict[str, float]` or a dataclass?
- Data: How to handle NaN/missing — drop, impute, or error?
- UX: Any hotkey conflicts with existing controls; prefer `,` and `.`?
- Performance: Target budget for feature X (ms/frame)?

### Problem‑solving approach
- Break problems into smaller tasks; research prior art and patterns.
- Clearly prioritize must‑haves vs nice‑to‑haves.
- Consider system‑wide impact, edge cases, error handling, and failure modes.
- Document architectural decisions and trade‑offs.

### Tooling and tasks (uv, Ruff, pytest, ty, VS Code)
- Dependencies/runtime: uv
  - Install/resolve: VS Code task “Install Dependencies” (uv sync)
  - Run: `uv run <cmd>` for any Python command
  - Add deps: `uv add <package>` (or edit `pyproject.toml` and sync)
- Lint/format: Ruff
  - VS Code task “Ruff: Format and Fix” (keeps repo ruff‑clean; document exceptions with comments)
- Type checking: ty
  - VS Code task "Type Check" (uvx ty check . --exit-zero; runs type checking with exit-zero for current compatibility)
  - **All type errors must be addressed before merging PRs**
  - Warnings are allowed but should be triaged and gradually resolved
- Tests: pytest
  - VS Code task “Run Tests” (default suite)
  - “Run Tests (Show All Warnings)” for diagnostics
  - “Run Tests (GUI)” for display‑dependent tests (headless via environment vars)
- Code quality checks: VS Code task “Check Code Quality” (Ruff + pylint errors‑only)
- Diagrams: VS Code task “Generate UML”

Quality gates to run locally before pushing:
1) Install Dependencies → 2) Ruff: Format and Fix → 3) Check Code Quality → 4) Type Check → 5) Run Tests

Shortcuts (optional shell):
- Break down complex problems into smaller, manageable tasks
- Research existing solutions and patterns before implementing new approaches
- Consider the impact of changes on the entire system, not just the immediate problem
- Document architectural decisions and trade-offs made during implementation
- Think about edge cases, error handling, and potential failure modes

## Documentation Standards

### Technical Documentation

- Create comprehensive documentation for all significant changes and new features
- Save documentation files in the `docs/` directory using a clear folder structure
- Each major feature or issue should have its own subfolder named in kebab-case
  - Format: `docs/dev/issues/42-fix-button-alignment/` or `docs/dev/issuesfeature-name/`
- Use descriptive README.md files as the main documentation entry point for each folder

### Documentation Content Requirements

Documentation should include:
- **Problem Statement**: Clear description of the issue being addressed
- **Solution Overview**: High-level approach and architectural decisions
- **Implementation Details**: Code examples, API changes, and technical specifics
- **Impact Analysis**: What systems/users are affected and how
- **Testing Strategy**: How the changes were validated
- **Future Considerations**: Potential improvements or known limitations
- **Related Links**: References to GitHub issues, pull requests, or external resources

### Documentation Best Practices

- Use proper markdown formatting with clear headings and structure
- Include code examples with syntax highlighting
- Add diagrams or screenshots when they improve understanding
  - Mermaid diagrams are welcome and encouraged for visualizing workflows, architecture, and relationships
- Write for future developers who may be unfamiliar with the context
- Keep documentation up-to-date as code evolves
- Use consistent formatting and follow markdown linting standards
- Avoid duplications. Link to existing documentation when relevant.
- Always provide README.md files in new documentation folders for overview and reference.

## Tooling and tasks (uv, Ruff, pytest, VS Code)
- Dependencies/runtime: uv
  - Install/resolve: VS Code task "Install Dependencies" (uv sync)
  - Run: `uv run <cmd>` for any Python command
  - Add deps: `uv add <package>` (or edit `pyproject.toml` and sync)
- Lint/format: Ruff
  - VS Code task "Ruff: Format and Fix" (keeps repo ruff‑clean; document exceptions with comments)
- Tests: pytest
  - VS Code task "Run Tests" (default suite)
  - "Run Tests (Show All Warnings)" for diagnostics
  - "Run Tests (GUI)" for display‑dependent tests (headless via environment vars)
- Code quality checks: VS Code task "Check Code Quality" (Ruff + pylint errors‑only)
- Diagrams: VS Code task "Generate UML"

Quality gates to run locally before pushing:
1) Install Dependencies → 2) Ruff: Format and Fix → 3) Check Code Quality → 4) Run Tests

## CI/CD expectations
- Tests: `uv run pytest tests`
- Lint: `uv run ruff check .` and `uv run ruff format --check .`
- The pipeline mirrors the local quality gates. Ensure green locally first.

CI mapping to local tasks and CLI:
- Lint job → Task “Ruff: Format and Fix” → `uv run ruff check . && uv run ruff format --check .`
- Code quality job → Task “Check Code Quality” → `uv run ruff check . && uv run pylint robot_sf --errors-only`
- Type check job → Task "Type Check" → `uvx ty check . --exit-zero`
- Test job → Task “Run Tests” → `uv run pytest tests`

Workflow location: `.github/workflows/ci.yml`.

## Validation scenarios and performance
### Validation scenarios (run after changes)
```bash
./scripts/validation/test_basic_environment.sh
./scripts/validation/test_model_prediction.sh
./scripts/validation/test_complete_simulation.sh
```
Success criteria:
- Basic environment: exits 0; no exceptions.
- Model prediction: exits 0; logs model load and inference without errors.
- Complete simulation: exits 0; simulation runs to completion without errors.

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
- Ruff: Python linter and formatter (run via "Ruff: Format and Fix" task).
- pytest: Testing framework (run via "Run Tests" tasks).
- VS Code tasks: Standardized workflows (install, lint, test, diagram).
- Quality gates: Minimal checks before pushing (install → lint/format → quality check → tests).

### Repository structure (key dirs)
- `robot_sf/` (source), `examples/`, `tests/`, `test_pygame/`, `fast-pysf/` (submodule), `scripts/`, `model/`, `docs/`

---

## Definition of Done (DoD)
- Requirements clarified (with options/assumptions recorded).
- Design doc added/updated and linked (if non‑trivial).
- Code implemented with tests (unit/integration; GUI when needed).
- Ruff clean and “Check Code Quality” clean locally.
- Type check clean (no type errors; warnings documented if present).
- Docs updated (README in feature folder, diagrams if changed).
- Validation scripts run and pass; optional benchmark if perf‑sensitive.
- CI green (lint + tests) and PR opened with appropriate links.

## Templates

Use the following templates for specific tasks.

- [issue template](../.github/ISSUE_TEMPLATE/issue_default.md)
- [design doc template](./templates/design-doc-template.md)
- [PR template](../.github/PULL_REQUEST_TEMPLATE/pr_default.md)


## Security & network policy
- No secrets in code, configs, or commit messages.
- Avoid network access in tests; prefer local fixtures. If unavoidable, document and gate behind flags.
- Don't exfiltrate data; handle PII safely (none expected in this repo).

## Large files & artifacts policy
- Don't commit large binaries to the repo; prefer Git LFS for models/datasets when needed.
- Use the `model/` directory conventions; document artifact sources and versions.


## Quick reference and TL;DR checklist
### Quick reference commands
```bash
# Setup after installation
git submodule update --init --recursive && uv sync && source .venv/bin/activate

# Validate changes
uv run ruff check . && uv run ruff format . && uv run pylint robot_sf --errors-only && uvx ty check . --exit-zero && uv run pytest tests

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
5) Run quality gates via tasks: Install Dependencies → Ruff: Format and Fix → Check Code Quality → Type Check → Run Tests
6) Update docs/diagrams; run “Generate UML” if classes changed
7) Open PR with summary, risks, and links to docs/tests
