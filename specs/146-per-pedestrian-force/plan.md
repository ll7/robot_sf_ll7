# Implementation Plan: Per-Pedestrian Force Quantiles

**Branch**: `146-per-pedestrian-force` | **Date**: 2025-10-24 | **Spec**: `specs/146-per-pedestrian-force/spec.md`
**Input**: Feature specification from `/specs/146-per-pedestrian-force/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement per-pedestrian force magnitude quantiles (q50, q90, q95) as new metrics that compute quantiles for each pedestrian individually across time, then average across pedestrians to form episode-level values. This differs from existing aggregated quantiles that flatten all (t,k) samples. The implementation will be vectorized using NumPy, expose keys `ped_force_q50`, `ped_force_q90`, `ped_force_q95`, integrate into `compute_all_metrics()`, update `METRIC_NAMES`, add unit tests, and document formulas and edge cases.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.13 (repo tested with 3.13.1)  
**Primary Dependencies**: NumPy, pytest (existing stack)  
**Storage**: N/A (in-memory arrays)  
**Testing**: pytest (extend `tests/test_metrics.py`)  
**Target Platform**: Linux/macOS CI (headless)  
**Project Type**: Single Python package/library (`robot_sf`) with tests  
**Performance Goals**: Vectorized O(T×K); T=1000, K=50 completes < 50 ms on dev hardware  
**Constraints**: Maintain stable metric naming contract; no schema breaking; handle NaN robustly (use nan-aware ops)  
**Scale/Scope**: Episodes up to thousands of timesteps and dozens of pedestrians

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Principle I (Reproducibility): Deterministic vectorized computation; seed-independent pure function. PASS.
- Principle III (Benchmark & Metrics First): Adds transparent metrics with formal docs; included in JSON metrics map. PASS.
- Principle VI (Metrics Transparency): Definitions and edge cases documented; tests assert semantics. PASS.
- Principle VII (Backward Compatibility): New metric keys only; existing names unchanged. Episode schema allows additionalProperties. PASS.
- Principle VIII (Docs as API): Update `metrics_spec.md` and central docs index reference. PASS.
- Principle IX (Tests for Public Behavior): Unit tests added; no CLI changes. PASS.

## Project Structure

### Documentation (this feature)

```
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

```
robot_sf/
└── benchmark/
  └── metrics.py        # Add per-ped quantiles function and METRIC_NAMES

tests/
└── test_metrics.py       # Add unit tests for per-ped quantiles

docs/
└── dev/issues/social-navigation-benchmark/metrics_spec.md  # Update definitions
```

**Structure Decision**: Single Python library repository; extend `robot_sf/benchmark/metrics.py`, existing tests file, and docs. No new packages.

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
