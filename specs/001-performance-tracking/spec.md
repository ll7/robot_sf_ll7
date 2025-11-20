# Feature Specification: Performance Tracking & Telemetry for Imitation Pipeline

**Feature Branch**: `001-performance-tracking`  
**Created**: 2025-11-19  
**Status**: Draft  
**Input**: User description: "Add duration estimates, progress tracking, run telemetry, resource monitoring, and automated performance recommendations/tests for imitation-learning pipeline workflows."

## Clarifications

### Session 2025-11-19

- Q: Should we rely on JSON run tracking or switch to TensorBoard/W&B for telemetry? → A: Keep the JSON tracker as the canonical artifact and add an opt-in TensorBoard adapter only when the extra effort is justified.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Monitor live pipeline progress (Priority: P1)

Pipeline operators running `examples/advanced/16_imitation_learning_pipeline.py` need clear "step X of N" status, elapsed time, and ETA so they can tell whether the run is healthy without diving into logs.

**Why this priority**: Most runs already take several minutes and span multiple scripts. Lack of ETA/progress creates uncertainty and makes it hard to decide whether to cancel or continue when a step appears stuck.

**Independent Test**: Execute the pipeline with telemetry enabled and confirm that each step announces its ordinal, descriptive name, elapsed duration, and updated ETA without depending on other new capabilities.

**Acceptance Scenarios**:

1. **Given** a user starts the pipeline with five enabled steps, **When** step 2 begins, **Then** the console/log displays "Step 2/5 – Collect expert trajectories (ETA 11m, elapsed 2m)" and updates ETA at least once while the step runs.
2. **Given** a step completes, **When** the tracker records the event, **Then** it logs the actual duration and adjusts the ETA for remaining steps based on observed runtimes.

---

### User Story 2 - Review historical runs (Priority: P2)

Researchers need a persisted manifest of past pipeline executions (timestamps, parameters, success/failure, artifacts) so they can answer "what happened last night" without re-running jobs.

**Why this priority**: Post-run triage and team communication depend on trustworthy records; without them people re-run expensive jobs just to infer durations or failure causes.

**Independent Test**: Run the pipeline twice, then inspect the generated manifests via a CLI helper to confirm each run entry is independent, complete, and readable even if the other stories are disabled.

**Acceptance Scenarios**:

1. **Given** at least one prior run exists, **When** the user invokes the run-tracking summary command, **Then** they can filter runs by date/status and view per-step durations plus links to stored artifacts.
2. **Given** a run aborts mid-way, **When** the user inspects the manifest, **Then** it marks the run as failed, lists the last completed step, and records partial telemetry collected up to that point.

---

### User Story 3 - Receive performance telemetry & recommendations (Priority: P3)

Power users want live resource metrics (steps/sec, FPS, CPU/GPU/memory) and automated recommendations (e.g., "increase env workers", "switch backend") plus the ability to trigger performance smoke tests before long runs.

**Why this priority**: After progress visibility, the next blocker is diagnosing why a run is slow. Telemetry paired with actionable advice reduces trial-and-error and keeps training throughput predictable.

**Independent Test**: Enable telemetry with a stress configuration, observe that snapshots are captured and summarized, then trigger the performance test command to receive recommendations without needing the historical run manifest.

**Acceptance Scenarios**:

1. **Given** telemetry thresholds (e.g., steps/sec below baseline) are breached, **When** the run completes, **Then** the summary includes at least one recommendation explaining the issue and pointing to concrete actions.
2. **Given** the user runs the dedicated performance test entry point, **When** it finishes, **Then** the tracker records measured throughput and compares it to stored baselines, flagging regressions beyond a configurable tolerance.

---

### Edge Cases

- Pipeline step skipped via configuration: progress totals must adjust dynamically so ETA stays accurate even when optional steps are disabled.
- Telemetry source unavailable (e.g., psutil missing, no GPU sensors): feature must degrade gracefully, mark metrics as "unavailable", and still persist progress data.
- Run interrupted by crash or user cancel: manifest must flush partial data within 5 seconds so no run is left undocumented.
- Concurrent runs on same machine: tracker must namespace run IDs and artifact paths to avoid overwriting telemetry files.
- Performance test triggered without benchmark history: recommendation engine must fall back to documented defaults and signal lack of baseline rather than guessing.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The pipeline orchestrator MUST present deterministic progress output (current step label, position within total enabled steps, elapsed time, and ETA) whenever a step starts or ends.
- **FR-002**: The system MUST measure and persist per-step durations, status (pending/running/completed/failed/skipped), and timestamps in a structured artifact under the canonical `output/` tree for every run.
- **FR-003**: Users MUST be able to retrieve a chronological list of prior runs (at least the last 20) with filters by date, status, and scenario identifier via a scriptable interface or CLI flag.
- **FR-004**: The telemetry layer MUST sample core resource metrics (steps per second, CPU %, memory, GPU utilization when present) at a configurable interval and degrade gracefully by labeling metrics as "unavailable" instead of failing when sensors cannot be read.
- **FR-005**: The tracker MUST emit a summarized run report (human-readable plus machine-readable JSON) containing key metrics, anomalies detected, and links to artifacts immediately after each run finishes or aborts.
- **FR-006**: The system MUST generate actionable recommendations whenever observed throughput drops below a documented baseline or resource saturation is detected, including rationale and suggested configuration changes.
- **FR-007**: Users MUST be able to trigger scripted performance smoke tests that run the minimal benchmark scenarios, capture throughput metrics, and store results alongside regular runs for comparison.
- **FR-008**: Telemetry and run-tracking capabilities MUST be opt-in/opt-out via documented flags or configuration settings so that headless CI can enable them without code changes.
- **FR-009**: The solution MUST respect the existing artifact policy by storing all new logs, telemetry snapshots, and manifests within `output/` (or the `ROBOT_SF_ARTIFACT_ROOT` override) and never scatter files elsewhere.
- **FR-010**: The feature MUST support concurrent runs by generating unique run identifiers and locking per-run artifact directories to prevent race conditions.
- **FR-011**: The telemetry module MUST preserve the JSON/Markdown tracker as the primary source of truth while exposing an optional adapter (e.g., TensorBoard event writer) that can be enabled when teams need richer dashboards without breaking headless runs.

### Key Entities *(include if feature involves data)*

- **Pipeline Run Record**: Canonical manifest capturing run ID, initiated timestamp, initiating command/config, enabled steps, final status, overall duration, and pointers to telemetry/performance outputs.
- **Step Execution Entry**: Child structure nested under the run record storing step name, order, start/end timestamps, elapsed duration, status, and per-step artifacts (logs, metrics, checkpoints).
- **Telemetry Snapshot**: Time-series sample containing steps/sec, FPS, CPU/GPU utilization, memory footprint, and sensor availability flags; referenced by both live console output and persisted reports.
- **Performance Recommendation**: Structured item produced by analysis rules detailing the detected issue (e.g., "steps/sec 30% below baseline"), recommended action (e.g., "increase workers to 4"), severity, and the evidence that triggered it.
- **Performance Test Result**: Output record for standalone perf-smoke runs noting scenario matrix used, throughput metrics, pass/fail classification, and linkage to the baseline it was compared against.

## Assumptions & Constraints

- Instrumentation targets the imitation-learning pipeline scripts first; other runners can opt in later but are out of scope for this feature.
- No graphical dashboards are introduced; outputs remain textual/JSON so they work in headless CI.
- Telemetry collection must add less than 5% wall-clock overhead to the pipeline; heavy sampling is optional and defaults stay lightweight.
- GPU metrics are best-effort—absence of NVIDIA tooling must not be treated as an error.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of pipeline runs display per-step progress plus ETA updates, and ETA accuracy improves to within ±20% after the second step completes.
- **SC-002**: Run manifests persist for at least the most recent 20 executions without manual intervention, with zero data loss even if a run is aborted mid-step (verified via kill-test).
- **SC-003**: Telemetry snapshots capture CPU and memory usage on 95% of supported hosts and add no more than 5% overhead to total runtime as measured by the new performance tests.
- **SC-004**: Whenever throughput drops ≥25% below the documented baseline, the system surfaces at least one actionable recommendation and records it in the run summary; performance smoke tests fail fast (under 10 minutes) when regressions exceed that threshold.

## Testing Notes (2025-11-20)

- Tracker quickstart smoke executed via `uv run python examples/advanced/16_imitation_learning_pipeline.py --demo-mode --enable-tracker --tracker-output output/run-tracker/quickstart_demo --tracker-smoke`, producing manifests + telemetry under `output/run-tracker/quickstart_demo/`.
- Verified CLI workflows:
	- `status` and `watch` confirm per-step ETA/duration updates.
	- `summary --format json` and `export --format markdown --output output/run-tracker/quickstart_demo/summary.md` capture telemetry aggregates, recommendations, and artifact links.
	- `list --limit 5` shows historical runs including the smoke entry.
- Optional telemetry mirroring documented; TensorBoard adapter not exercised in this run (requires torch/tensorboardX).
- Performance wrapper validated with `uv run python scripts/tools/run_tracker_cli.py perf-tests --scenario configs/validation/minimal.yaml --output output/run-tracker/perf-tests/latest --num-resets 3`, writing `perf_test_results.json` plus manifest lines under `output/run-tracker/perf-tests/latest/`.
- CI harness validated with `uv run python scripts/validation/run_examples_smoke.py --perf-tests-only --perf-num-resets 1`, confirming tracker smoke + telemetry perf wrapper succeed via the new entry point that CI now calls.
