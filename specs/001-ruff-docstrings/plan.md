# Implementation Plan: Ruff Docstring Enforcement

**Branch**: `001-ruff-docstrings` | **Date**: 2025-12-02 | **Spec**: [`spec.md`](./spec.md)  
**Input**: Feature specification from `/specs/001-ruff-docstrings/spec.md`

## Summary

Enable Ruff docstring rules (D100–D107, D417, D419, D102, D201) across the entire repository, remediate all resulting violations, and document enforcement so CI and local workflows fail fast when docstrings are missing or incomplete. Technical approach: extend the existing `pyproject.toml` Ruff config, add repo-wide lint tasks, and iterate over Python packages (library, scripts, tests, fast-pysf subtree where applicable) to add concise docstrings aligned with governance requirements.

## Technical Context

**Language/Version**: Python 3.11 (uv-managed virtual environment)  
**Primary Dependencies**: Ruff (docstring rules), Loguru (logging referenced in docstrings), pytest for regression validation  
**Storage**: Not applicable (source-only change)  
**Testing**: `uv run ruff check`, `uv run pytest tests`, optional targeted module imports to ensure docstrings do not break runtime  
**Target Platform**: Cross-platform dev (macOS/Linux CI) with headless compatibility  
**Project Type**: Monorepo with Python packages (`robot_sf`, `fast-pysf`, scripts, tests)  
**Performance Goals**: Ruff docstring lint completes in <90s on CI runners; no noticeable slowdown in developer workflow  
**Constraints**: Must respect Principle XI (docstrings as helper contracts) and avoid modifying vendored third-party code unless necessary; preserve existing public API behavior  
**Scale/Scope**: ~900 Python modules across core library, scripts, examples, tests, and subtree integrations; goal is 100% docstring compliance for public APIs while allowing intentional exclusions via Ruff configuration

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I (Reproducibility)**: Docstring enforcement improves reproducibility documentation; compliant.
- **Principle VIII (Documentation as API Surface)**: Feature directly advances this; compliant provided docs index updated when new guides/quickstart created.
- **Principle XI (Helper Documentation)**: Aligns with requirement; no violation.
- **Principle XII (Logging)**: Not impacted; ensure docstring edits do not introduce prints.
- **Principle XIII (Test Value)**: Lint/test updates must still evaluate necessity; plan includes lint gate updates without weakening tests.

Gate Status: **PASS** (no constitutional conflict identified).

**Post-Phase-1 Recheck**: Research, data model, contracts, and quickstart artifacts reinforce Principles I, VIII, and XI by documenting reproducible lint workflows; status remains **PASS**.

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
robot_sf_ll7/
├── robot_sf/            # Primary library package (factories, configs, metrics)
├── fast-pysf/           # Subtree physics engine (Python + Cython bindings)
├── scripts/             # CLI entry points and training utilities
├── examples/            # Demos and smoke workflows
├── tests/               # Unified pytest suite for robot_sf
├── test_pygame/         # GUI-dependent tests (headless via env vars)
├── docs/                # Development and user guides
├── .github/             # CI workflows and prompts
└── specs/001-ruff-docstrings/  # Feature documentation artifacts
```

**Structure Decision**: Treat the repository as a single Python-focused monorepo; docstring enforcement must cover `robot_sf`, `fast-pysf`, scripts, examples, and tests. No new top-level projects are introduced.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
