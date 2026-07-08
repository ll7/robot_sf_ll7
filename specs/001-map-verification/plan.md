# Implementation Plan: Map Verification Workflow

**Branch**: `001-map-verification` | **Date**: 2025-11-20 | **Spec**: `/specs/001-map-verification/spec.md`
**Input**: Feature specification from `/specs/001-map-verification/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build a reproducible map-verification workflow that (1) runs as a single CLI/CI command covering every SVG in `maps/svg_maps/`, (2) instantiates maps via the correct robot or pedestrian factory to guarantee runtime compatibility, (3) emits structured JSON manifests plus actionable console diagnostics, and (4) enforces performance/time-boxing so large assets cannot stall CI.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11 (uv-managed virtual environment)  
**Primary Dependencies**: `robot_sf.gym_env` factories + unified configs, Loguru logging, SVG parsing utilities already present in `robot_sf.maps`, optional geometry helpers (Shapely)  
**Storage**: File-based inputs/outputs (SVG maps under `maps/svg_maps/`, JSON/JSONL manifests under `output/validation/`)  
**Testing**: Pytest suite (`tests/maps/test_map_verifier.py`, CI tasks)  
**Target Platform**: Linux/macOS headless runners (CI + local dev)
**Project Type**: Python research/benchmark toolkit module  
**Performance Goals**: Median verification ≤ 3s per map, soft budget 20s/map, hard timeout 60s run-wide, CI job < 5 min  
**Constraints**: Must run fully offline/headless, obey artifact policy, and emit Loguru-based structured logs alongside the JSON manifest for observability  
**Scale/Scope**: ≈50 SVG maps today, each ≤5 MB, with growth expectation <100 maps in near term

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I (Reproducible Core)**: PASS – workflow is scriptable via CLI + CI with deterministic config inputs.
- **Principle II (Factory Abstraction)**: PASS – verifier instantiates environments exclusively through `make_robot_env` / `make_pedestrian_env`.
- **Principle III (Benchmark & Metrics)**: PASS – emits structured JSON manifest enabling downstream aggregation/metrics.
- **Principle VIII (Documentation as API)**: CONDITIONAL – must add verifier docs to `docs/SVG_MAP_EDITOR.md` + central index during implementation.
- **Principle XII (Logging)**: PASS – research settled on Loguru structured INFO logs + JSON manifest outputs.

*Post-Phase-1 Recheck*: Principle XII remains satisfied via planned logging approach. Principle VIII still conditional until doc updates land during implementation.

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
robot_sf/
├── maps/
│   ├── svg_loader.py        # SVG parsing + metadata helpers
│   └── verification/        # NEW verifier module (rule registry, runners)
├── gym_env/
│   └── environment_factory.py (existing factories invoked by verifier)
└── common/
  └── artifact_paths.py    # Existing helpers for output routing

scripts/
└── validation/
  └── verify_maps.py       # CLI/CI entry point wrapping verifier APIs

tests/
└── maps/
  └── test_map_verifier.py # Unit/integration tests covering rules + CLI

specs/001-map-verification/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── contracts/

docs/
├── SVG_MAP_EDITOR.md        # Update with verifier instructions
└── README.md                # Link from central index

output/
└── validation/
  └── map_verification.json (structured manifest)
```

**Structure Decision**: Extend existing monorepo layout by adding a `robot_sf.maps.verification` module plus `scripts/validation/verify_maps.py` CLI, backed by focused tests under `tests/maps`. Documentation updates land under `docs/` per constitution Principle VIII.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
