# Implementation Plan: Performance Tracking & Telemetry for Imitation Pipeline

**Branch**: `001-performance-tracking` | **Date**: 2025-11-19 | **Spec**: [`spec.md`](./spec.md)
**Input**: Feature specification from `/specs/001-performance-tracking/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Execution Flow (/speckit.plan scope)

```
1. Run `.specify/scripts/bash/setup-plan.sh --json` → capture FEATURE_SPEC, IMPL_PLAN, SPECS_DIR, BRANCH
2. Load feature spec + constitution + plan template; fill Technical Context (mark unknowns if any)
3. Record initial Constitution Check (halt if violations lack justification)
4. Phase 0: Resolve clarifications and document decisions in `research.md`
5. Phase 1: Produce `data-model.md`, `contracts/`, `quickstart.md`, and update agent context
6. Re-run Constitution Check after design updates
7. Phase 2: Describe task-planning approach (do NOT create `tasks.md` yet)
8. Stop and report artifacts/branch to user (ready for `/speckit.tasks`)
```

## Summary

Add canonical run tracking to the imitation-learning pipeline so operators always see "step X of N", elapsed time, ETA, live resource telemetry, historical manifests, and actionable performance recommendations. The tracker will emit structured JSON/Markdown artifacts under `output/`, optionally mirror metrics to TensorBoard, and provide scripted performance smoke tests plus CLI access to past runs.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11 (uv-managed virtual environment)  
**Primary Dependencies**: `robot_sf` core modules, Loguru logging, `psutil` for CPU/memory metrics, optional NVIDIA/NVML bindings, optional TensorBoard event writer, YAML/JSON helpers already present in repo  
**Storage**: Structured JSON/Markdown manifests in `output/` (respecting `ROBOT_SF_ARTIFACT_ROOT`), optional TensorBoard log directory  
**Testing**: Pytest (unit + integration), plus existing validation scripts under `scripts/validation/`  
**Target Platform**: Headless Linux/macOS developers + CI runners (no GUI requirement)  
**Project Type**: Python library + CLI/scripts layered on existing `robot_sf` package  
**Performance Goals**: <5% telemetry overhead on pipeline runtime, ETA accuracy within ±20% after second step, telemetry sampling default interval ≥1s to avoid hot-path slowdown  
**Constraints**: Must honor artifact policy (everything under `output/`), operate without network access, degrade gracefully when GPU metrics unavailable, concurrency-safe for multiple runs  
**Scale/Scope**: Designed for tens of long-running training runs per day (hours each), manifests must retain at least 20 historical entries and support parallel execution on shared workstations/CI

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I (Reproducibility)**: JSON/Markdown manifests and CLI quickstart preserve deterministic run records with seeds/config references. ✅
- **Principle II (Factory Pattern)**: Instrumentation wraps existing pipeline factories (`make_robot_env`, etc.) without instantiating envs directly. ✅
- **Principle III (Benchmark & Metrics First)**: Telemetry outputs append-only JSONL aligned with benchmark schema expectations and SNQI metrics. ✅
- **Principle VIII (Documentation as API)**: Spec/plan/research/quickstart update docs, and agent context will mention new tooling. ✅
- **Artifact Policy**: All trackers write under `output/` or override path; TensorBoard adapter only mirrors data when enabled. ✅
- **Logging Principle XII**: Tracker uses Loguru and avoids noisy prints in library code; CLI wrappers may print summaries. ✅

Result: Gates satisfied; proceed to Phase 0.

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
├── benchmark/        # JSONL schema helpers, aggregation utilities
├── common/           # artifact paths, logging, seed helpers to extend
├── gym_env/          # factory + unified config (telemetry hooks attach here)
├── sim/              # untouched (FastPysf integration)
└── training/         # training orchestration scripts consuming telemetry

examples/
└── advanced/16_imitation_learning_pipeline.py   # primary entry point to augment

scripts/
├── validation/       # performance + artifact guard scripts to extend
└── tools/            # potential location for CLI wrappers (e.g., run summary)

tests/
├── test_tracking/    # new unit/emulation tests for tracker (to add)
├── integration/      # scenario smoke tests verifying manifests
└── perf_utils/       # existing performance policy helpers reused
```

**Structure Decision**: Reuse the existing `robot_sf` Python package and `examples/advanced/16_imitation_learning_pipeline.py`, adding a new `robot_sf/telemetry/` module (or subpackage under `robot_sf/common/`) plus test coverage under `tests/test_tracking/`. CLI helpers live in `scripts/` and respect artifact policy.

## Phase 0: Outline & Research

1. Confirmed canonical telemetry sink remains JSON/Markdown, with TensorBoard as optional adapter (aligns with artifact policy + Principle I). Documented in `research.md`.
2. Selected `psutil` + optional NVML for telemetry stack with graceful fallbacks to standard library, ensuring low sampling overhead.
3. Defined rule-based recommendation engine (threshold driven) to keep behavior deterministic/testable; captured rationale vs. ML-driven alternatives.
4. Chose to extend existing `scripts/validation/performance_smoke_test.py` via a wrapper instead of inventing a new harness so we reuse current baseline enforcement.

**Outcome**: `research.md` resolves all clarifications; no remaining NEEDS CLARIFICATION items.

## Phase 1: Design & Contracts

1. Authored `data-model.md` enumerating `PipelineRunRecord`, `StepExecutionEntry`, `TelemetrySnapshot`, `PerformanceRecommendation`, and `PerformanceTestResult` with validation rules, relationships, and retention strategy.
2. Produced `contracts/run-tracker.openapi.yaml` covering run lifecycle, telemetry ingestion, recommendations, and perf-test recording; mirrors functional requirements FR-001–FR-011.
3. Wrote `quickstart.md` describing how to enable the tracker, inspect live status, optionally stream to TensorBoard, run perf tests, and list historical runs—all via reproducible `uv run` commands.
4. Ran `.specify/scripts/bash/update-agent-context.sh copilot`, propagating the new telemetry stack details into `.github/copilot-instructions.md` per repository policy.

## Post-Design Constitution Check

- **Principle I (Reproducibility)**: Data model + quickstart uphold deterministic manifests under `output/`; optional adapters cannot mutate canonical JSON. ✅
- **Principle II (Factory Pattern)**: Design keeps instrumentation around factory-created pipelines; no direct env instantiation introduced. ✅
- **Principle VIII (Documentation as API)**: Spec folder now includes quickstart + contracts; agent context updated. ✅
- **Principle IX (Tests for Public Behavior)**: Contracts and quickstart scope future tests (CLI + perf smoke) prior to implementation. ✅
- **Principle XII (Preferred Logging & Observability)**: Plan explicitly routes telemetry through Loguru-backed modules; quickstart highlights CLI contexts where prints are acceptable. ✅

Result: No new violations; proceed to Phase 2 planning.

## Phase 2: Task Planning Approach (for /speckit.tasks)

**Task Generation Strategy**
- Derive tracker runtime tasks from `data-model.md` (manifest writer, step lifecycle hooks, telemetry sampler, recommendation engine) and tie each to corresponding tests in `tests/test_tracking/`.
- Translate OpenAPI operations into scriptable interfaces/CLI commands, ensuring each endpoint maps to an integration or contract test stub.
- Map quickstart steps to validation tasks (pipeline smoke run, live status CLI, TensorBoard opt-in, perf test wrapper) so docs stay executable.
- Capture documentation + CHANGELOG updates as standalone tasks referencing `docs/dev_guide.md` and `docs/README.md` per artifact policy guidance.
- Include migration + guard tasks: manifest rotation, lockfile handling, perf baseline storage, CI wiring for perf smoke tests.

**Ordering / Dependencies**
1. Implement manifest writer + locking, because all other modules depend on reliable persistence.
2. Add step lifecycle instrumentation + CLI progress output before telemetry sampling to guarantee core FR-001/FR-002 coverage.
3. Layer telemetry sampler + recommendation engine with unit tests mocking psutil/NVML; follow with optional TensorBoard adapter.
4. Extend performance smoke tests + wrappers; wire CLI commands and update docs/changelog last.
5. Mark documentation/perf-test tasks as parallelizable once core tracker + telemetry modules are stable.

Expected `/speckit.tasks` output: ~20 ordered tasks spanning tests, implementation, docs, validation, and rollout safeguards.

## Phase 3+: Implementation & Validation (out of scope for /speckit.plan)

- **Phase 3**: `/speckit.tasks` generates `tasks.md` from the strategy above.
- **Phase 4**: Execute tasks (code, tests, docs) while maintaining artifact policy.
- **Phase 5**: Validation—run pytest + validation scripts + quickstart flows, then prep PR/CI evidence.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |

## Progress Tracking

**Phase Status**
- [x] Phase 0: Research complete (`research.md`)
- [x] Phase 1: Design + contracts complete (`data-model.md`, `contracts/`, `quickstart.md`, agent context)
- [x] Phase 2: Task planning approach documented (ready for `/speckit.tasks`)
- [ ] Phase 3: Tasks generated (`tasks.md` pending)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation complete

**Gate Status**
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved in `research.md`
- [x] Agent context updated via `.specify/scripts/bash/update-agent-context.sh copilot`
