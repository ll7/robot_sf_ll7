# Implementation Plan: Clean Root Output Directories

**Branch**: `243-clean-output-dirs` | **Date**: November 13, 2025 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/243-clean-output-dirs/spec.md`

## Summary

Consolidate generated artifacts into a single `output/` directory with mandated subdirectories, migrate all legacy root-level paths, and enforce guard checks that fail fast on regressions. Documentation and CI updates ensure contributors adopt the policy while still supporting the `ROBOT_SF_ARTIFACT_ROOT` override for custom paths.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11 (uv-managed virtual environment)  
**Primary Dependencies**: Python standard library (`pathlib`, `json`, `shutil`), Loguru, pytest, uv CLI  
**Storage**: Local filesystem (repository-relative `output/` tree or overridden path)  
**Testing**: pytest suite, dedicated guard-check unit tests, migration dry-run regression tests  
**Target Platform**: macOS and Linux developer machines; GitHub Actions CI  
**Project Type**: Monorepo research platform (single project)  
**Performance Goals**: Guard execution < 1s; migration < 5s for typical artifact volumes; no observable slowdown in CI jobs  
**Constraints**: Preserve reproducibility (Principles I & IV), honor `ROBOT_SF_ARTIFACT_ROOT`, fail fast on legacy paths post-remediation  
**Scale/Scope**: Touches all artifact-producing scripts (tests, validation, examples) plus docs and CI workflows

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I – Reproducible Social Navigation Core**: Centralized artifacts and migration tooling keep outputs deterministic and discoverable. ✅
- **Principle IV – Unified Configuration & Deterministic Seeds**: Continues to rely on documented environment variable override; no ad-hoc knobs introduced. ✅
- **Principle VII – Backward Compatibility & Evolution Gates**: Migration helper + documentation provide the required deprecation path before fail-fast enforcement. ✅
- **Principle VIII – Documentation as API Surface**: Plan includes updates to README/dev guide and dedicated quickstart. ✅
- **Principle IX – Test Coverage for Public Behavior**: Guard and migration scripts will gain new tests ensuring enforcement is verified. ✅

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
```text
robot_sf/
├── benchmark/
├── gym_env/
├── common/
└── sim/

scripts/
├── validation/
├── tools/                 # new migration + guard scripts live here
└── benchmark02.py

docs/
├── README.md
├── dev_guide.md           # updated with artifact policy
└── architecture/

tests/
├── test_guard/            # new guard + migration unit tests
└── test_examples/

.github/
└── workflows/             # guard check wired into CI

output/ (gitignored)
├── coverage/
├── benchmarks/
├── recordings/
├── wandb/
└── tmp/
```

**Structure Decision**: Maintain single-project layout; add tooling under `scripts/tools/`, tests under `tests/test_guard/`, and documentation updates under existing `docs/` hierarchy. No new packages or subprojects required.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
