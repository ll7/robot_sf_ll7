# Robot SF – Copilot Guide

Use this as the single source of truth for this repo. Prefer these rules over generic habits; fall back to ad‑hoc searches or shell when something is clearly out of sync.

## Executive summary
- **Architecture**: Social navigation RL framework with gym/gymnasium environments, SocialForce pedestrian simulation via `fast-pysf` submodule, StableBaselines3 training pipeline
- **Core pattern**: Factory-based environment creation (`make_robot_env()` etc.) — never instantiate environments directly
- **Dependencies**: `fast-pysf` git submodule for pedestrian physics; always run `git submodule update --init --recursive` after clone
- **Toolchain**: uv + Ruff + pytest with VS Code tasks; run quality gates before pushing
- **Testing**: Unit tests in `tests/`, GUI-dependent tests in `test_pygame/` (with headless env vars), integration tests for smoke/performance validation

## Table of contents
- Executive summary
- Architecture and key components
- Critical dependencies and setup  
- Core principles
- Tooling and tasks (uv, Ruff, pytest, VS Code)
- Testing strategy and benchmark system
- Environment creation patterns
- CI/CD expectations
- Validation scenarios and performance
- Training and examples (incl. Docker)
- Common issues and solutions
- Migration notes
- Helpful definitions and repository structure
- Definition of Done (DoD)
- PR template (snippet)
- Design doc template (snippet)
- Security & network policy
- Large files & artifacts policy
- Quick reference and TL;DR checklist
 - Markdown linting (docs quality)

---

## Architecture and key components
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

## Critical dependencies and setup
### Fast-pysf submodule (REQUIRED)
**Always initialize submodules** after git clone or checkout:
```bash
git submodule update --init --recursive
```
Without this, pedestrian simulation will fail. The `fast-pysf/` directory contains optimized SocialForce physics.

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

### Working with baselines and benchmarks
**Baseline algorithms** in `robot_sf/baselines/` implement navigation strategies for benchmarking:
```python
from robot_sf.baselines import get_baseline

# Get Social Force planner
SocialForcePlanner = get_baseline("baseline_sf")
planner = SocialForcePlanner(config, seed=42)
```

**Benchmark system** provides standardized evaluation:
```bash
# Run baseline benchmarks with CLI
uv run python -m robot_sf.benchmark.cli baseline --algo baseline_sf --out results.json

# List available algorithms
uv run python -m robot_sf.benchmark.cli list-algorithms
```

### Testing strategy (THREE test suites)
```bash
# 1. Main unit/integration tests (2-3 min)
uv run pytest tests

# 2. GUI/display-dependent tests (headless mode)  
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest test_pygame

# 3. fast-pysf submodule tests (some may fail without map files)
uv run python -m pytest fast-pysf/tests/ -v
```

## Core principles
### Code quality standards
- Clear, intent‑revealing names; small, cohesive functions; robust error handling.
- Follow existing style; document non‑obvious choices with comments/docstrings.
- Avoid duplication; prefer composition and reuse.
- Keep public behavior backward‑compatible unless explicitly stated.
- Write comprehensive unit tests for new features and bug fixes (GUI tests in `test_pygame/`).

#### Docstrings
- Every public module, function, class, and method should have a docstring.
- Docstrings should use triple double quotes (""").
- The first line should be a short summary of the object’s purpose, starting with a capital letter and ending with a period.
- If more detail is needed, leave a blank line after the summary, then continue with a longer description.
- For functions/methods: document parameters, return values, exceptions raised, and side effects.

### Always ask clarifying questions (with options)
- Before implementing, confirm requirements with targeted questions.
- Prefer multiple‑choice options to speed decisions; group by scope, interfaces, data, UX, performance.
- If answers are unknown, propose sensible defaults and proceed (don't block on non‑essentials).

Examples (copy‑ready):
- Scope: Is the metric per episode or a per‑timestep aggregate?
- Interfaces: Return shape `dict[str, float]` or a dataclass?
- Data: How to handle NaN/missing — drop, impute, or error?
- UX: Any hotkey conflicts with existing controls; prefer `,` and `.`?
- Performance: Target budget for feature X (ms/frame)?

### Problem‑solving approach
- Break problems into smaller tasks; research prior art and patterns.
- Consider system‑wide impact, edge cases, error handling, and failure modes.
- Document architectural decisions and trade‑offs.

---

## Tooling and tasks (uv, Ruff, pytest, ty, VS Code)
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
  - Format: `docs/42-fix-button-alignment/` or `docs/feature-name/`
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

## Installation and Setup

For complete installation instructions, see the [main README.md](../README.md#installation). The project uses `uv` for modern Python dependency management.

## Working Effectively

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

---

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

---

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

---

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

## Markdown formatting (mdformat)
We use `mdformat` for deterministic Markdown formatting (installed via `pyproject.toml`). Prefer running the formatter before any substantial doc PR to minimize review noise and keep style consistent.

### Commands
```bash
uv run mdformat .          # Format all markdown files
uv run mdformat --check .  # CI-style check (no changes)
```

Config lives in `[tool.mdformat]` in `pyproject.toml` (wrap=0 disables hard wrapping; keep soft wrapping in editor).

### VS Code Task
Use the task: "Markdown: Format" (invokes `uv run mdformat .`).

### Adoption Strategy
1) Baseline format commit
2) Encourage contributors to run before commit
3) Optionally promote CI `markdown-format` job from advisory to blocking once noise stabilizes

### Optional Pre-commit Hook
```yaml
- repo: https://github.com/executablebooks/mdformat
  rev: 0.7.19
  hooks:
    - id: mdformat
      additional_dependencies:
        - mdformat-gfm
        - mdformat-frontmatter
```

If future style concerns arise (e.g., link validation), add a separate non-blocking job (e.g., link checker) rather than reintroducing a linter.

---

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

---

## PR template (snippet)
```markdown
## Summary
Brief description of the change.

## Linked issues
- #<id>

## Changes
- …

## Tests
- Unit/integration/GUI; results; timings

## Risks / rollout
- Migration/back‑compat notes; toggles; fallback

## Docs
- Links to updated docs/diagrams

## Validation
- Output of validation scripts; benchmarks (if any)
```

---

## Design doc template (snippet)
```markdown
# <Title>
(See "Documentation standards and design docs" for full guidance)

## Context, Goals, Non‑goals
## Constraints & assumptions
## Options & trade‑offs
## Chosen approach (diagram optional)
## Contracts (APIs/data, error modes)
## Test plan
## Rollout & back‑compat
## Metrics/observability
## Open questions / follow‑ups
```

---

## Security & network policy
- No secrets in code, configs, or commit messages.
- Avoid network access in tests; prefer local fixtures. If unavoidable, document and gate behind flags.
- Don't exfiltrate data; handle PII safely (none expected in this repo).

---

## Large files & artifacts policy
- Don't commit large binaries to the repo; prefer Git LFS for models/datasets when needed.
- Use the `model/` directory conventions; document artifact sources and versions.

---

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