# Robot SF – Development Guide

[← Back to Documentation Index](./README.md)

Welcome to the Robot SF Development Guide! This document serves as the central reference for contributors working on the Robot SF codebase. It covers setup instructions, architectural overviews, coding standards, and best practices to ensure a smooth development experience.
<!--
This document should be kept as short as possible to maintain clarity and ease of navigation.
Whenever possible, link out to more detailed documents or external resources.
Refacotr this document regularly to be as concise as possible.
LLM Constitution and guides can be found here:
- `.specify/memory/constitution.md`
- `.github/copilot-instructions.md`
- `AGENTS.md`
-->

## Setup

### Installation and setup

```bash
# One‑time setup with dev extras and pre-commit
uv sync --all-extras
source .venv/bin/activate
uv run pre-commit install

# Quick import check
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; print('Import successful')"
```

### Critical dependencies and setup: Fast-pysf integration

The `fast-pysf/` directory contains the optimized SocialForce physics engine and is now integrated as a **git subtree** (previously a submodule). After cloning the repository, the fast-pysf code is automatically available—no additional initialization steps required.

**Note**: If you're working with an older branch that still uses submodules, see the [Subtree Migration Guide](./SUBTREE_MIGRATION.md) for migration instructions and workflow differences.

### Quick Start Commands

```bash
# source .venv
source .venv/bin/activate
# Lint+format
uv run ruff check --fix . && uv run ruff format .
# Tests
uv run pytest tests
```

### Examples Quickstart Walkthrough

The `examples/README.md` file now captures a curated onboarding path. New contributors
can get a full tour in roughly five minutes by running the quickstart trio in order:

```bash
uv run python examples/quickstart/01_basic_robot.py
uv run python examples/quickstart/02_trained_model.py
uv run python examples/quickstart/03_custom_map.py
```

- `01_basic_robot.py` introduces the environment factory pattern and headless rollouts.
- `02_trained_model.py` replays the bundled PPO baseline and writes JSONL metrics to
  `output/benchmarks/episodes_demo_ppo.jsonl`.
- `03_custom_map.py` shows how to load `maps/svg_maps/debug_06.svg` via
  `RobotSimulationConfig.map_pool` for custom layouts.

See `examples/README.md` for the decision tree, prerequisites, and links to additional
tiers (advanced features, benchmarks, plotting, and archived scripts).

### Advanced Feature Demos

Developers exploring specific capabilities should jump to the curated scripts in
`examples/advanced/`. Each file follows the numbered naming scheme surfaced in
`examples/README.md` and comes with a manifest-backed docstring describing how to
run it. Highlights include:

- **Backends & factory ergonomics**: `01_backend_selection.py` and
  `02_factory_options.py` demonstrate switching simulators and recording options via
  unified configs.
- **Observation & training workflows**: `03_image_observations.py` and
  `04_feature_extractors.py` showcase image sensors and feature extractor presets
  (run with `uv sync --all-extras`).
- **Pedestrian & policy scenarios**: Scripts `06`–`11` cover factory-based
  pedestrian environments, single/multi pedestrian setups, and PPO rollouts using
  the maintained checkpoints under `model/`.
- **Tooling, validation, and visualization**: `12_social_force_planner_demo.py`
  through `15_view_recording.py` provide the Social Force planner showcase, SVG
  map validation helper, trajectory visualization, and recording playback flows.

Check the Advanced table in `examples/README.md` for prerequisites, tags, and whether
a script is enabled for CI smoke execution.

### One‑liner quality gates (CLI):

```bash
uv run ruff check --fix . && uv run ruff format . && uvx ty check . --exit-zero && uv run pytest tests
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
- **`fast-pysf/`**: Git subtree providing optimized SocialForce pedestrian simulation
- **`docs/`**: Documentation, design notes, and development guides

### Schema Management

**Canonical schema location**: `robot_sf/benchmark/schemas/`
- Episode schemas: `episode.schema.v1.json` (single source of truth)
- Runtime resolution: Use `robot_sf.benchmark.schema_loader.load_schema()` for schema loading
- Schema validation: Automatic validation against JSON Schema draft 2020-12
- Version management: Semantic versioning with breaking change detection
- Git hooks: Prevent duplicate schema files from being committed

### Data flow and integration

- **Training loop**: `scripts/training_ppo.py` → factory functions → vectorized environments → StableBaselines3
- **Benchmarking**: `robot_sf/benchmark/cli.py` → baseline algorithms → episode runs → JSON/JSONL output → analysis
- **Pedestrian simulation**: Robot environments → FastPysfWrapper → `fast-pysf` subtree → NumPy/Numba physics

### Configuration hierarchy

**For complete documentation, see [Configuration Architecture](./architecture/configuration.md)** (precedence rules, migration guide, module structure).

Use unified config classes from `robot_sf.gym_env.unified_config`:

```python
from robot_sf.gym_env.unified_config import RobotSimulationConfig, ImageRobotConfig

config = RobotSimulationConfig()
config.peds_have_obstacle_forces = True  # Enable ped-robot physics interaction
env = make_robot_env(config=config)
```

### Backend selection (simulator swap)

The simulation backend can be selected via configuration without modifying environment code. Available backends are registered in `robot_sf.sim.registry`:

```python
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig

# Use fast-pysf backend (default)
config = RobotSimulationConfig()
config.backend = "fast-pysf"  # Default; can be omitted
env = make_robot_env(config=config)

# Use dummy backend (for testing)
config = RobotSimulationConfig()
config.backend = "dummy"
env = make_robot_env(config=config)
```

**Available backends:**
- `"fast-pysf"` (default): SocialForce pedestrian simulation via fast-pysf subtree
- `"dummy"`: Minimal test simulator with constant positions (for smoke tests)

**Backend registration:**
Custom backends can be registered via `robot_sf.sim.registry.register_backend()`. See `robot_sf/sim/backends/` for implementation examples.

**Error handling:**
Unknown backend names fall back to legacy `init_simulators()` with a warning. For strict validation, use `robot_sf.gym_env.config_validation.validate_config()` before environment creation.

### Utility Modules

All shared utility functions and type definitions live in `robot_sf/common/`:
- `robot_sf/common/types` - Type aliases (Vec2D, Line2D, RobotPose, Circle2D, etc.)
- `robot_sf/common/errors` - Error handling utilities (raise_fatal_with_remedy, warn_soft_degrade)
- `robot_sf/common/seed` - Random seed management for reproducibility (set_global_seed, SeedReport)
- `robot_sf/common/compat` - Compatibility helpers (validate_compatibility)

**Example imports:**
```python
from robot_sf.common.types import Vec2D, RobotPose, Line2D
from robot_sf.common.errors import raise_fatal_with_remedy
from robot_sf.common.seed import set_global_seed

# Convenience imports also available:
from robot_sf.common import Vec2D, RobotPose, set_global_seed
```

**Troubleshooting:**
- If IDE autocomplete doesn't work after importing from `robot_sf.common`, restart your IDE's language server:
  - **VS Code**: Command Palette → "Python: Restart Language Server"
  - **PyCharm**: File → Invalidate Caches / Restart

## Design and development workflow recommendations

- Consider using <https://github.com/github/spec-kit> for complex specifications and design docs.
  - Examples can be found in the `specs` directory.
  - Prompts are unique to the llm provider used. Adjust accordingly.
  - Copilot prompts can be found in `.github/prompts`
  - LLM Constitution and guides can be found here:
    - `.specify/memory/constitution.md`
    - `.github/copilot-instructions.md`
    - `AGENTS.md`
- Clarify exact requirements before starting implementation.
- If necessary, ask clarifying questions (with options) to confirm scope, interfaces, data handling, UX, and performance.
  - Discuss possible options and trade-offs.
  - Give arguments to the options for easy decision-making.
  - Provide options to quickly converge on a decision.
- For complex tasks:
  - Create a design doc (see template below) for non-trivial changes.
  - Create a file based TODO list (see example below).
  - Break task down into smaller subtasks and tackle them iteratively.
- Prioritize must-haves over nice-to-haves
- Document assumptions and trade-offs.
- Ensure that the documentation, docstrings, and comments are updated to reflect code changes.
- Docstring style is specified in `pyproject.toml` -> `[tool.ruff.lint.pydocstyle]` -> `convention`
- Progress cadence: always keep tests and documentation up-to-date. As long as you document your chain of thought and what ran, you can report outcomes after finishing the work.
- Prefer programmatic use and factory functions over CLI; the CLI is not important.
- Working mode: prioritize a thin, end-to-end slice that runs. Optimize and polish after a green smoke test (env reset→step loop or demo run).
- Whenever possible, add a demo or example to illustrate new functionality.
- Avoid disabling linters, type checks, or tests unless absolutely necessary.
  - Whenever you have the chance, refactor to fix issues rather than suppressing them. Especially `# noqa: C901` (complexity) and `# type: ignore` (type hints).
- Always document the purpose of documents at the top of the file. (e.g., Python files, README.md, design docs, issue folders)
- Use American English.

### One-liner architecture summary

- Architecture in one line: Gym/Gymnasium envs → factory functions → FastPysfWrapper → fast-pysf physics; training/eval via StableBaselines3; baselines/benchmarks under `robot_sf/baselines` and `robot_sf/benchmark`.
- Environments: always create via factories (`make_robot_env`, `make_image_robot_env`, `make_pedestrian_env`). Configure via `robot_sf.gym_env.unified_config` only; toggle flags before passing to the factory.
- Simulation glue: interact with pedestrian physics through `robot_sf/sim/FastPysfWrapper`. Don’t import from `fast-pysf` directly inside envs.
- Baselines/benchmarks: get planners with `robot_sf.baselines.get_baseline(...)`. Prefer programmatic runners; CLI exists at `robot_sf/benchmark/cli.py` for convenience.
- Demos/trainings: keep runnable examples in `examples/` and scripts in `scripts/`. Place models in `model/`, maps in `maps/svg_maps/`, and write outputs under `output/`.
- Tests: core in `tests/`; GUI in `test_pygame/` (headless: `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy`). Physics-specific tests live in `fast-pysf/tests/`.
- Quality gates (local): Install Dependencies → Ruff: Format and Fix → Check Code Quality → Type Check → Run Tests (see VS Code Tasks).

### Artifact policy & tooling

- Canonical outputs live under `output/` with stable subdirectories: `output/coverage/`, `output/benchmarks/`, `output/recordings/`, `output/wandb/`, and `output/tmp/`.
- Run `uv run python scripts/tools/migrate_artifacts.py` (or the console entry point `uv run robot-sf-migrate-artifacts`) after pulling to consolidate any legacy `results/`, `recordings/`, `htmlcov/`, or `coverage.json` paths.
- Enforce the policy locally and in CI with `uv run python scripts/tools/check_artifact_root.py`; the guard fails fast when new top-level artifacts appear.
- Override the artifact destination by exporting `ROBOT_SF_ARTIFACT_ROOT=/path/to/custom/output` before invoking scripts; the helpers and guard honor the override consistently.
- Canonical helpers in `robot_sf.common.artifact_paths` (e.g., `ensure_canonical_tree`) create the required layout for tests and tooling—prefer them over hard-coded paths.
- Need a guided walkthrough? Follow the [artifact policy quickstart](../specs/243-clean-output-dirs/quickstart.md) for migration, guard usage, and override examples end to end.

### Testing strategy (UNIFIED test suite)

**The project now uses a unified test suite** running both robot_sf and fast-pysf tests via a single command.

#### Unified Test Suite

```bash
# Run ALL tests (robot_sf + fast-pysf) - RECOMMENDED
uv run pytest  # Number of test is steadily increasing, ca. 1200

# Run only robot_sf tests
uv run pytest tests

# Run only fast-pysf tests  
uv run pytest fast-pysf/tests  # → 12 tests

# Run with parallel execution (faster)
uv run pytest -n auto
```

#### Legacy / Specialized Test Suites

```bash
# 1. Main unit/integration tests (2-3 min) - NOW PART OF UNIFIED SUITE
uv run pytest tests  # → 881 tests

# 2. GUI/display-dependent tests (headless mode)  
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest test_pygame

# 3. fast-pysf subtree tests - NOW PART OF UNIFIED SUITE
uv run pytest fast-pysf/tests  # → 12 tests (all passing with map fixtures)
```

**Note**: The unified test command (`uv run pytest`) automatically discovers and runs tests from both `tests/` and `fast-pysf/tests/` directories. Test count increased from ~43 (legacy documentation) to 893 tests after fast-pysf integration.

#### Test Significance Verification

**Before fixing or investigating test failures, verify the test's value and necessity.** Not all tests provide equal value; some may be outdated, overly brittle, or testing non-critical behavior.

**Evaluation Questions** (ask before investing fix effort):

1. **Core Feature Coverage**: Does this test verify a public contract?
   - Factory behavior, schema compliance, metric correctness, deterministic reproducibility
   - **If YES** → High priority, fix immediately

2. **User Impact**: Would failure in production affect users?
   - Incorrect metrics, broken benchmarks, environment crashes
   - **If YES** → High priority, fix immediately

3. **Regression Prevention**: Does it catch known past bugs?
   - Validates recent fixes, prevents known failure modes
   - **If YES** → Medium priority, fix within sprint

4. **Edge Case vs. Common Path**: Does it test rare scenarios?
   - Low real-world occurrence, no documented incidents
   - **If YES and no incidents** → Low priority, consider archiving

5. **Brittleness**: Does it fail frequently without indicating real bugs?
   - Timing issues, display dependencies, environmental flakiness
   - **If YES** → Candidate for refactoring or removal

6. **Redundancy**: Is the same behavior tested elsewhere?
   - Identical logic covered by unit and integration tests
   - **If YES** → Consider consolidation

**Decision Actions**:
- **High Priority** (Core/User Impact): Fix immediately, these protect critical invariants
- **Medium Priority** (Regression/Important Edges): Fix or update within sprint; document if deferred
- **Low Priority** (Rare edges, redundant): Consider archiving; reassess value before fix effort
- **Flaky/Brittle**: Stabilize with retries/mocks if valuable; otherwise remove with documented rationale

**Maintenance Discipline**:
- Removing tests requires documented reason in commit message
- Deferred low-priority failures need tracking issue with "test-debt" label
- New tests MUST include docstring stating: (1) what contract/behavior is verified, (2) why it matters
- Quarterly audit: review tests by runtime and failure frequency; challenge bottom 10% on value

**Example**: If a test for an obscure edge case in trajectory smoothing fails but:
- No user has ever reported this scenario
- The edge case requires artificial setup unlikely in real usage
- Core smoothing is covered by other tests

→ Consider documenting the edge case in code comments and archiving the test, rather than spending hours debugging environmental setup issues.

### Coverage workflow (automatic collection)

**Coverage collection is enabled by default** — no extra commands needed! When you run tests, coverage data is automatically collected and reported.

The test harness sets the ``ROBOT_SF_ARTIFACT_ROOT`` environment variable so that
example scripts and helpers write into a temporary directory instead of the
repository tree. This keeps the canonical ``output/`` hierarchy clean while
preserving the examples' default behavior for normal runs. To opt-in manually,
export the same variable before invoking scripts.

Try to increase the test coverage over time by adding tests when touching code. See the must-have checklist below for guidance.

#### Quick start
```bash
# Run tests (coverage collected automatically)
uv run pytest tests

# View HTML report (preferred helper)
uv run python scripts/coverage/open_coverage_report.py

# Manual fallback
open output/coverage/htmlcov/index.html

# Or use VS Code task: "Run Tests with Coverage" → "Open Coverage Report"
```

#### What gets measured

- **Included**: All code in `robot_sf/` package
- **Excluded**: Tests, examples, scripts, `fast-pysf/` subtree
- **Output formats**: 
  - Terminal summary (printed after test run)
  - HTML report (`output/coverage/htmlcov/index.html` - interactive, detailed)
  - JSON data (`output/coverage/coverage.json` - for tooling)

#### Understanding coverage output

```
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
robot_sf/gym_env/environment.py           150     15  90.00%  42-45, 89-92
robot_sf/sim/simulator.py                 200     50  75.00%  10-20, 150-180
---------------------------------------------------------------------
TOTAL                                   10605    876  91.73%
```

- **Stmts**: Total executable lines
- **Miss**: Uncovered lines
- **Cover**: Percentage covered
- **Missing**: Line numbers not executed by tests

#### Coverage configuration

Configured in `pyproject.toml`:
- `[tool.coverage.run]` — collection settings (source, omit patterns, parallel support)
- `[tool.coverage.report]` — report formatting (precision, exclusions)
- `[tool.pytest.ini_options]` — automatic pytest integration

No changes needed for normal development — defaults are production-ready.

#### Advanced usage
```bash
# Run with parallel workers (coverage merges automatically)
uv run pytest tests -n auto

# Run specific test file with coverage
uv run pytest tests/test_gym_env.py -v

# View coverage data programmatically
python -c "import json; print(json.load(open('output/coverage/coverage.json'))['totals'])"
```

For coverage gap analysis, trend tracking, and CI integration, see `docs/coverage_guide.md` (created as part of US2/US3).

### Must-have checklist

- [ ] Use factory env creators; do not instantiate env classes directly.
- [ ] Set config via `robot_sf.gym_env.unified_config` before env creation; avoid ad‑hoc kwargs.
- [ ] Keep lib code print-free; use logging from loguru for info and warnings.
- [ ] Run VS Code Tasks: Install Dependencies, Ruff: Format and Fix, Check Code Quality, Type Check, Run Tests.
- [ ] Add a test or smoke (e.g., env reset/step) when you change public behavior.
- [ ] For GUI-dependent tests, set headless env vars; avoid flaky display usage in CI.
- [ ] Treat `fast-pysf/` as part of the repository, changes can be made.
- [ ] Put new demos under `examples/` and new runners under `scripts/`.
- [ ] Whenever a demo is possible, add one.

### Optional backlog (track but don’t block)

- [ ] Tighten type hints for new public APIs; migrate call sites gradually.
- [ ] Add programmatic benchmark examples and extend baseline coverage.
- [ ] Update or add docs under `docs/` for new components; include diagrams when useful.
- [ ] Add performance smoke (steps/sec) when touching hot paths.
- [ ] Add proper docstrings to comply with pydoclint and and pydocstyle

### Quick links

- Environment overview: `docs/ENVIRONMENT.md`
- Simulation view: `docs/SIM_VIEW.md`
- Refactoring and architecture notes: `docs/refactoring/`
- SNQI tools and metrics: `docs/snqi-weight-tools/README.md`
- Data analysis helpers: `docs/DATA_ANALYSIS.md`
- Contributor onboarding / repo structure: `AGENTS.md`

### Executive summary

- **Architecture**: Social navigation RL framework with gym/gymnasium environments, SocialForce pedestrian simulation via `fast-pysf` subtree, StableBaselines3 training pipeline
- **Core pattern**: Factory-based environment creation (`make_robot_env()` etc.) — never instantiate environments directly
- **Dependencies**: `fast-pysf` git subtree for pedestrian physics (automatically included after clone, see [Subtree Migration Guide](./SUBTREE_MIGRATION.md))
- **Toolchain**: uv + Ruff + ty + pytest with VS Code tasks; run quality gates before pushing
- **Testing**: Unit tests in `tests/`, GUI-dependent tests in `test_pygame/` (with headless env vars), integration tests for smoke/performance validation
- **Documentation**: Comprehensive docs under `docs/` with design principles, architecture, usage, and migration notes
  - Development notes: `docs/dev/*`

### Logging & Observability (Principle XII)

The canonical logging facade is **Loguru**. Library code (anything under `robot_sf/` or wrappers over `fast-pysf`) must not use bare `print()` for informational or warning messages. Acceptable `print()` exceptions: (1) short CLI entry scripts in `scripts/` or `examples/` where stdout is the UX, (2) early bootstrap failures before logging configuration, (3) tests explicitly asserting stdout content. Migration of stray prints to `from loguru import logger` with `logger.info|warning|error` is treated as maintenance (PATCH) unless it changes user‑visible contract output.

Guidelines:
 - Prefer structured context (e.g., `logger.info("Reset complete seed={seed} scenario={sid}")`).
 - Avoid inside per‑timestep loops; aggregate and log at episode boundaries to protect performance budgets.
 - Use WARNING for degraded but continuing states (e.g., zero frames when recording requested), ERROR for aborting conditions, CRITICAL for irreversible state corruption.
 - Tests may temporarily raise log level to DEBUG for diagnosing flakes but should reset after.
 - Provide a toggle (env var or parameter) when adding verbose debug logging to hot paths.

Rationale: Centralized logging enables deterministic capture/suppression in benchmarks, simplifies CI noise control, and aligns with Constitution Principle XII (Preferred Logging & Observability).

### Code quality standards

- Clear, intent‑revealing names; small, cohesive functions; robust error handling.
- Follow existing style; document non‑obvious choices with comments/docstrings.
- Add helpful comments to quickly understand the cod
- Avoid duplication; prefer composition and reuse.
- Keep public behavior backward‑compatible unless explicitly stated.
- Write comprehensive unit tests for new features and bug fixes (GUI tests in `test_pygame/`).
- **Verify test value before investing fix effort** (see Test Significance Verification in Testing Strategy section).

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
-	Runs/outputs/benchmarks: output/benchmarks/<timestamp>_<shortname>/ (store config.yaml, git_meta.json, logs, metrics, artifacts).
-	Deterministic seed in both config and code

### Code reviews

- All changes must be reviewed by at least one other team member.
- Reviewers should check for correctness, style, test coverage, and documentation.
- Use GitHub’s review tools to leave comments and approve changes.

#### Docstrings

- Every module, function, class, and method should have a docstring.
- Docstrings should use triple double quotes (""").
- The first line should be a short summary of the object’s purpose, starting with a capital letter and ending with a period.
- If more detail is needed, leave a blank line after the summary, then continue with a longer description.
- For functions/methods: document parameters, return values, exceptions raised, and side effects.
- Private/internal code should also have docstrings explaining their purpose for easier maintainability.
- Follow the pydocstyle convention specified in `pyproject.toml`.

### Clarify questions (with options)
-In case of ambiguity or uncertainty about requirements, always ask clarifying questions before starting implementation. Provide multiple-choice options to facilitate quick decision-making. Group questions by scope, interfaces, data handling, UX, and performance.
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
- Use existing libraries and frameworks when possible to avoid reinventing the wheel
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

### Docs Folder Structure

Here’s a concise map of the docs folder to help you find the right guidance quickly. Each folder should include a README.md for context, links, and references.

#### Top-level guides (entry points)
- README.md — Main docs landing page.
- dev_guide.md — Primary development reference (setup, workflow, testing, CI).
- `ENVIRONMENT.md` — Environment overview and usage.
- `SIM_VIEW.md` — Simulation view/UI notes.
- `GPU_SETUP.md` — GPU/NVIDIA Docker setup.
- `UV_MIGRATION.md` — Migration notes to uv.
- Topic-specific guides:
  - `DATA_ANALYSIS.md`, `trajectory_visualization.md`, `SVG_MAP_EDITOR.md`, `fast_pysf_wrapper.md`, `pyreverse.md`, `curvature_metric.md`, `snqi_weight_cli_updates.md`.

#### Focused subfolders
- `2x-speed-vissimstate-fix/`
  - README.md — Notes and outcome for the VissimState 2x speed fix.
- `baselines/`
  - `social_force.md` — Baseline Social Force documentation.
- `docs/dev/` — In-progress/engineering docs and design notes
- `extract-pedestrian-action-helper/`
  - README.md — Helper tool documentation.
- `img/` — Images used across docs
- `ped_metrics/` Pedestrian metrics documentation and analysis notes.
- `refactoring/` Migration/architecture reports and plans
- `snqi-weight-tools/` — SNQI weight tooling user docs and schema
- `templates/` Template for new design docs.
- `video/` Demo animations for docs.

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
- When the document is longer htan 50 lines, create a table of contents at the top for easy navigation. Ideally, use `markdown.extension.toc.create` to *Markdown All in One: Create Table of Contents*.

#### Visualizations and Reports

- Use visualizations to illustrate complex concepts or data flows
- Include performance reports or benchmarks when relevant
- Ensure all visual assets are stored in the `docs/img/`, `docs/figures` or `docs/video/` directories for easy access and consistency
- Generate figures using code when possible to ensure reproducibility
- Figures should be exported in high-quality vector formats (e.g., SVG, PDF) for clarity

#### Figure and Visualization Guidelines
All figures must be **reproducible from code** and directly **integratable into LaTeX documents**:
- **Output format**
  - Always export **vector PDFs** (`.pdf`) for inclusion in LaTeX.
  - Optionally export `.png` (300 dpi) for slides/presentations.
- **Reproducibility**
  - Each figure = **one standalone script** in `robot_sf/benchmark/figures/` or `scripts/figures/` (tracked under version control).
  - Script must read data, generate plot, and save into `docs/figures/`.
  - No manual edits in Illustrator, Inkscape, etc.
  - Clear and unique filenames: `fig-<short-description>.py` and `fig-<short-description>.pdf`.
- **Version control**
  - Scripts and generated figures go into version control.
  - Data files (if any) go into `output/figures/` (respecting the canonical artifact root).
- **Consistent style**
  - Use Matplotlib with predefined `rcParams`:
    - `savefig.bbox = "tight"`
    - `pdf.fonttype = 42`
    - font sizes: 9 pt labels, 8 pt ticks/legend
    - line width ~1.2–1.6 pt
  - Axis labels and math should use LaTeX syntax: `r"$\sin(x)$"`.
- **Figure sizing**
  - Provide helper function for resizing
  - Default: single-column width (`fraction=1.0`).
- **File locations**
  - Figures go into `docs/figures/` (tracked).
  - Data exports (if used) into `output/figures/`.


## Tooling and tasks (uv, Ruff, pytest, VS Code)
- Dependencies/runtime: uv
  - Install/resolve: VS Code task "Install Dependencies" (uv sync)
  - Run: `uv run <cmd>` for any Python command
  - Add deps: `uv add <package>` (or edit `pyproject.toml` and sync)
- Lint/format: Ruff
  - VS Code task "Ruff: Format and Fix" (keeps repo ruff‑clean with expanded rule set including bug catchers, modernization, and performance checks; document exceptions with comments)
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

## CI Performance Monitoring
The CI pipeline includes integrated performance monitoring for system package installation optimization:

**Performance Targets**:
- Package installation: < 73 seconds (50% reduction from 2min 26sec baseline)
- Overall CI job completion within acceptable time limits

**Monitoring Infrastructure**:
- **CI Monitoring Script**: `scripts/ci_monitoring.py` - Tracks job and step timing
- **Performance Metrics**: `scripts/ci-tests/performance_metrics.py` - Analyzes and reports metrics
- **Package Validation**: `scripts/ci-tests/package_validation.py` - Validates package availability

**CI Workflow Integration**:
- Performance monitoring starts before package installation
- Metrics recorded during package installation steps
- Performance data saved as artifacts for analysis
- Automatic validation against performance targets

**Local Testing**:
- Use `act` tool for local CI workflow testing: `act push --container-architecture linux/amd64 --job ci --verbose`
- See `scripts/ci-tests/README.md` for comprehensive `act` usage guide
- Performance metrics can be analyzed locally using the metrics scripts

**Performance Breach Handling**:
- Soft breaches (< 20s): Warning logged, CI continues
- Hard breaches (≥ 60s): Test failure, CI stops
- Override with `ROBOT_SF_PERF_RELAX=1` for known variance (temporary only)

## Validation scenarios and performance
### Validation scenarios (run after changes)
```bash
./scripts/validation/test_basic_environment.sh
./scripts/validation/test_model_prediction.sh
./scripts/validation/test_complete_simulation.sh
uv run python scripts/validation/run_examples_smoke.py --dry-run
uv run python scripts/validation/run_examples_smoke.py --perf-tests-only
uv run python scripts/validation/run_examples_smoke.py
uv run python scripts/tools/check_artifact_root.py

# Performance baseline validation
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/performance_smoke_test.py
```
Success criteria:
- Basic environment: exits 0; no exceptions.
- Model prediction: exits 0; logs model load and inference without errors.
- Complete simulation: exits 0; simulation runs to completion without errors.
- Example smoke harness: exits 0; all `ci_enabled` examples pass and archived entries are reported as skipped via manifest metadata.
- Artifact guard: exits 0; repository root remains clean with all artifacts under `output/` (mirror of the CI enforcement step).
- Performance smoke test: exits 0; meets baseline performance targets (see `docs/performance_notes.md`).
  - Threshold logic now includes soft vs hard tiers with environment overrides. Soft breaches on CI default to WARN (exit 0) unless `ROBOT_SF_PERF_ENFORCE=1`.
    - Environment variables:
      - `ROBOT_SF_PERF_CREATION_SOFT` (default 3.0)
      - `ROBOT_SF_PERF_CREATION_HARD` (default 8.0)
      - `ROBOT_SF_PERF_RESET_SOFT` (default 0.50 resets/sec)
      - `ROBOT_SF_PERF_RESET_HARD` (default 0.20 resets/sec)
  - `ROBOT_SF_PERF_ENFORCE=1` to fail on soft (and hard) breaches (use locally for strict tuning).
  - (Advanced) `ROBOT_SF_PERF_SOFT` / `ROBOT_SF_PERF_HARD` may be set to numeric seconds to temporarily override thresholds (intended only for internal testing of enforcement logic; not part of the stable public interface).
    - Hard threshold breaches always FAIL.

  ### Example maintenance workflow

  1. **Validate catalog** – `uv run python scripts/validation/validate_examples_manifest.py` ensures the manifest enumerates every script and that docstrings stay aligned with summaries.
  2. **Review planned changes** – `uv run python scripts/validation/run_examples_smoke.py --dry-run` prints the `ci_enabled` set before executing pytest, making it easy to confirm archive decisions.
  3. **Run tracker/perf gate** – `uv run python scripts/validation/run_examples_smoke.py --perf-tests-only --perf-num-resets 2` exercises the imitation pipeline tracker smoke plus telemetry perf wrapper without re-running the entire pytest suite.
  4. **Execute smoke harness** – `uv run python scripts/validation/run_examples_smoke.py` runs all active examples headlessly; pytest fixtures already configure pygame for a dummy display.
  5. **Archive responsibly** – whenever a script moves into `examples/_archived/`, update `examples/_archived/README.md`, set `ci_enabled: false` with a `ci_reason`, and point the module docstring at the maintained replacement.

### Run tracker & history CLI

- Enable the tracker with `--enable-tracker` on `examples/advanced/16_imitation_learning_pipeline.py`. A background guard now snapshots manifests roughly every five seconds and traps `SIGINT`/`SIGTERM`, so failed or cancelled runs emit a `failed` manifest entry automatically.
- Inspect live progress with `status` or `watch`:
  ```bash
  uv run python scripts/tools/run_tracker_cli.py status <run_id>
  uv run python scripts/tools/run_tracker_cli.py watch <run_id> --interval 1.0
  ```
  Both commands read the latest manifest snapshot and show current step, elapsed time, ETA, and the last completed step.
- Use `list` to review prior runs (defaults to the most recent 20). Helpful filters:
  - `--status pending|running|completed|failed|cancelled`
  - `--since 2025-01-15T00:00:00+00:00` (UTC ISO timestamps)
  - `--format table|json` for human vs machine-readable output
- `summary` (aliased as `show`) prints per-run breakdowns, with `--format text|json|markdown`. Markdown output intentionally mirrors the exported summaries so docs/changelogs can embed them verbatim.
- `export` writes Markdown or JSON summaries directly to disk:
  ```bash
  uv run python scripts/tools/run_tracker_cli.py export <run_id> \
    --format markdown \
    --output output/run-tracker/summaries/<run_id>.md
  ```
  Exports include per-step durations, artifact paths, and any failure context produced by the guard.
- Mirror telemetry to TensorBoard when you need dashboards:
  ```bash
  uv run python scripts/tools/run_tracker_cli.py enable-tensorboard <run_id> --logdir output/run-tracker/tb/<run_id>
  uv run tensorboard --logdir output/run-tracker/tb
  ```
  The CLI replays `telemetry.jsonl` into SummaryWriter so you can inspect CPU/GPU trends without touching the canonical JSON artifacts.
- Run the performance smoke wrapper straight from the CLI instead of calling scripts manually:
  ```bash
  uv run python scripts/tools/run_tracker_cli.py perf-tests \
    --scenario configs/validation/minimal.yaml \
    --output output/run-tracker/perf-tests/latest \
    --num-resets 5
  ```
  Results are persisted in `perf_test_results.json` with pass/soft-breach/fail classification plus any recommendations triggered by the telemetry rules.
- Because the guard writes manifests on a timer and on signals, partial runs survive restarts—`list`/`show` will always have at most a five-second gap between what ran and what was recorded.

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

## Benchmark runner: parallel workers and resume

The benchmark runner supports process-based parallel execution and safe resume.

- Parallelism: Use multiple workers to run independent episodes concurrently.
- Resume: Skips episodes that are already present in the output JSONL.

Key points
- Parent-only writes: only the parent process writes JSONL lines to avoid corruption.
- Episode identity: jobs are identified deterministically from scenario params and seed; existing episodes are skipped when resume is enabled.
- macOS: workers > 1 uses the spawn start method; ensure worker code is importable/picklable and defined at module top level (no lambdas/closures).

CLI usage
- Run a batch with parallel workers and default resume behavior:
  - robot_sf_bench run --scenarios configs/baselines/example.yaml --output output/benchmarks/episodes.jsonl --workers 4
- Force recomputation (disable resume):
  - robot_sf_bench run --scenarios configs/baselines/example.yaml --output output/benchmarks/episodes.jsonl --workers 4 --no-resume
- Baseline computation also accepts the same flags:
  - robot_sf_bench baseline --episodes output/benchmarks/episodes.jsonl --output output/benchmarks/baseline.jsonl --workers 4

Programmatic usage
- Prefer factory functions and programmatic APIs in library code:
  - from robot_sf.benchmark.runner import run_batch
  - from robot_sf.benchmark import baseline_stats
  - run_batch(scenarios, out_path=..., schema_path=..., workers=4, resume=True)
  - baseline_stats.run_and_compute_baseline(episodes_path=..., out_path=..., workers=4, resume=True)

Notes
- Default behavior is resume=True for programmatic APIs and CLI (omit --no-resume to keep it enabled).
- When resuming, open files in append mode if you want to keep existing lines; the runner will not duplicate episodes.
- On macOS spawn, module-level top-level functions are required for worker processes to import successfully.
- Resume accelerator: The runner writes a small sidecar manifest (episodes.jsonl.manifest.json) caching episode ids and file stat. On subsequent runs, resume uses this manifest when valid and transparently falls back to scanning the JSONL if the sidecar is stale or missing. No user action required.

## Aggregation and Confidence Intervals

Once you have a JSONL of episodes, you can aggregate metrics by group and optionally attach bootstrap confidence intervals.

CLI usage

- Aggregate without CIs (default):
  - robot_sf_bench aggregate --in output/benchmarks/episodes.jsonl --out output/benchmarks/summary.json
- Aggregate with CIs (enable with >0 samples):
  - robot_sf_bench aggregate --in output/benchmarks/episodes.jsonl --out output/benchmarks/summary_ci.json --bootstrap-samples 1000 --bootstrap-confidence 0.95 --bootstrap-seed 123

Options

- --group-by: Dotted path for grouping (default: scenario_params.algo)
- --fallback-group-by: Used when group-by is missing (default: scenario_id)
- --bootstrap-samples: Number of bootstrap resamples; 0 disables CI keys
- --bootstrap-confidence: Confidence level, e.g., 0.90, 0.95
- --bootstrap-seed: Optional deterministic seed for CIs
- --snqi-weights/--snqi-baseline: Recompute metrics.snqi during aggregation

Output format

- For each group and metric, the aggregator returns mean, median, p95.
- When CIs are enabled, additional keys are included: mean_ci, median_ci, p95_ci as [low, high].

Programmatic usage

```python
from robot_sf.benchmark.aggregate import read_jsonl, compute_aggregates_with_ci

records = read_jsonl("output/benchmarks/episodes.jsonl")
summary = compute_aggregates_with_ci(
    records,
    group_by="scenario_params.algo",
    fallback_group_by="scenario_id",
    bootstrap_samples=1000,
    bootstrap_confidence=0.95,
    bootstrap_seed=123,
)
```

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

### Imitation Learning Pipeline (PPO Pre-training)

The project supports accelerating PPO training via behavioral cloning pre-training from expert trajectories. This enables sample-efficient training by warm-starting agents with expert demonstrations.

**Quick Overview:** Expert PPO Training → Trajectory Collection → BC Pre-training → PPO Fine-tuning → Comparison Analysis

**For complete documentation, see [Imitation Learning Pipeline Guide](./imitation_learning_pipeline.md)** which includes:
- Detailed step-by-step workflow
- Configuration file examples
- Validation and debugging tools
- Artifact locations and manifest tracking
- Sample-efficiency metrics (target: ≤70% of baseline timesteps)
- Troubleshooting and best practices

**Quick Start:**
```bash
# End-to-end wrapper (recommended for new users)
uv run python examples/advanced/16_imitation_learning_pipeline.py

# Or run individual steps manually:
uv run python scripts/training/train_expert_ppo.py --config configs/training/ppo_imitation/expert_ppo.yaml
uv run python scripts/training/collect_expert_trajectories.py --dataset-id expert_v1 --policy-id ppo_expert_v1 --episodes 200
uv run python scripts/training/pretrain_from_expert.py --config configs/training/ppo_imitation/bc_pretrain.yaml
uv run python scripts/training/train_ppo_with_pretrained_policy.py --config configs/training/ppo_imitation/ppo_finetune.yaml
```

Set `--log-level DEBUG` if you need the full resolved-config dumps from the factory helpers (default is INFO to keep console noise down). Use `--backend <name>` to override the auto-selected simulator backend (defaults to the fastest available choice via `select_best_backend`). The end-to-end example auto-generates BC/ppo fine-tuning configs under `output/tmp/imitation_pipeline/`, so you only need to edit the YAML files when running the scripts manually.

**Also see:**
- End-to-end example: `examples/advanced/16_imitation_learning_pipeline.py`
- Detailed workflows: `specs/001-ppo-imitation-pretrain/quickstart.md`

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
- `robot_sf/` (source), `examples/`, `tests/`, `test_pygame/`, `fast-pysf/` (subtree), `scripts/`, `model/`, `docs/`

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
uv sync && source .venv/bin/activate

# Validate changes
uv run ruff check . && uv run ruff format . && uv run pylint robot_sf --errors-only && uvx ty check . --exit-zero && uv run pytest tests

# Functional smoke (headless)
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; env = make_robot_env(); env.reset(); print('OK')"

# Optional perf benchmark
DISPLAY= MPLBACKEND=Agg uv run python scripts/benchmark02.py

# If running commands outside of `uv run`, activate the virtual environment:
source .venv/bin/activate
```

### TL;DR workflow checklist
1) Clarify requirements (ask concise, optioned questions)
2) Draft design doc under `docs/` (link issue, add test plan)
3) Implement with small, reviewed commits
4) Add/extend tests in `tests/` or `test_pygame/`
5) Run quality gates via tasks: Install Dependencies → Ruff: Format and Fix → Check Code Quality → Type Check → Run Tests
6) Update docs/diagrams; run “Generate UML” if classes changed
7) Open PR with summary, risks, and links to docs/tests

### Per-Test Performance Budget

To prevent regression of integration test runtime, a performance budget policy is enforced for all tests:

Policy defaults (feature 124):
- Soft threshold: < 20s (advisory – prints guidance when exceeded)
- Hard timeout: 60s (enforced via `@pytest.mark.timeout(60)` or signal alarms inside long-running integration tests)
- Report count: Top 10 slowest tests printed at session end
- Relax mode: Set `ROBOT_SF_PERF_RELAX=1` to suppress soft breach warnings (use sparingly; still prints report)
- Enforce mode: Set `ROBOT_SF_PERF_ENFORCE=1` to escalate any soft or hard breach to a test session failure

Implementation components (all under `tests/perf_utils/`):
- `policy.py` – `PerformanceBudgetPolicy` dataclass providing `classify(duration)-> none|soft|hard`
- `reporting.py` – aggregation and formatted slow test report
- `guidance.py` – deterministic heuristic suggestions (reduce episodes, horizon, matrix size, etc.)
- `minimal_matrix.py` – single-source helper for minimal benchmark scenario matrix (used by resume & reproducibility tests)

Collector flow:
1. Each test call duration captured via a timing hook in `tests/conftest.py`.
2. At terminal summary the top-N slow tests are ranked and printed with breach classification & guidance lines.
3. If `ROBOT_SF_PERF_ENFORCE=1` (and relax not set) any soft or hard breach converts the run to a failure (exit code changed). Optional internal overrides: set `ROBOT_SF_PERF_SOFT` / `ROBOT_SF_PERF_HARD` for targeted enforcement tests.

Guidance examples:
- Soft breach near 25s: "Reduce episode count / seeds", "Use minimal scenario matrix helper"
- Very long (>40s) test: horizon + matrix recommendations prioritized

Authoring guidance for new tests:
- Keep semantic assertions; minimize episodes (`max_episodes=2`), horizon, seed list
- Reuse `write_minimal_matrix` instead of duplicating inline YAML
- Assert absence of heavy artifacts (videos) where not required

Performance troubleshooting checklist:
1. Confirm `smoke=True` or minimal workload flags applied
2. Reduce `max_episodes`, `initial_episodes`, `batch_size`
3. Disable bootstrap sampling (`bootstrap_samples=0`)
4. Lower `horizon_override`
5. Ensure `workers=1` for deterministic ordering in timing-sensitive tests

When relaxing:
Use `ROBOT_SF_PERF_RELAX=1` temporarily only for known CI variance; file a follow-up issue if sustained.

Hard timeout breaches should be rare; investigate infinite loops or large scenario expansions if encountered.
