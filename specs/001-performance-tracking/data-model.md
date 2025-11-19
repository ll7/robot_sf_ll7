# Data Model: Performance Tracking & Telemetry

## Overview

The telemetry subsystem persists structured manifests describing each pipeline run, its constituent steps, sampled resource metrics, recommendations, and optional performance-test executions. All records live under the canonical artifact root (`output/` or `ROBOT_SF_ARTIFACT_ROOT`).

## Entities

### PipelineRunRecord
- **Identifier**: `run_id` (UUID4) — unique per execution, reused across all nested entities.
- **Fields**:
  - `created_at` (ISO-8601) — UTC timestamp when the run started.
  - `completed_at` (ISO-8601 | null) — timestamp when run finished/aborted.
  - `status` (`pending` | `running` | `completed` | `failed` | `cancelled`).
  - `initiator` (string) — CLI command or script invoking the pipeline.
  - `scenario_config_path` (path) — absolute/relative path to scenario YAML used.
  - `enabled_steps` (array of strings) — ordered list of step identifiers (e.g., `collect_expert`, `pretrain_bc`).
  - `artifact_dir` (path) — resolved output directory for run-specific files.
  - `summary` (object) — aggregated metrics (total_duration, avg_steps_per_sec, recommendation count, etc.).
- **Relationships**:
  - `steps` → array of `StepExecutionEntry` objects keyed by `run_id`.
  - `telemetry_stream` → array of `TelemetrySnapshot` objects keyed by `run_id`.
  - `recommendations` → array of `PerformanceRecommendation` objects keyed by `run_id`.
  - `perf_tests` → array of `PerformanceTestResult` entries referencing `run_id` (optional; stored when user executes smoke tests).
- **Validation**:
  - `created_at <= completed_at` when `completed_at` is present.
  - `enabled_steps` must match actual recorded steps (no extraneous names).
  - `artifact_dir` must live under canonical artifact root.

### StepExecutionEntry
- **Identifier**: composite (`run_id`, `step_id`).
- **Fields**:
  - `step_id` (string) — stable identifier (e.g., `collect_expert`).
  - `display_name` (string) — human-readable step title.
  - `order` (int) — 1-indexed sequence.
  - `status` (`pending` | `running` | `completed` | `failed` | `skipped`).
  - `started_at` / `ended_at` (ISO-8601 | null).
  - `duration_seconds` (float) — computed `ended_at - started_at`.
  - `eta_snapshot_seconds` (float) — ETA displayed when step started.
  - `artifacts` (array of paths) — per-step outputs (checkpoints, JSONL chunks).
- **Relationships**: belongs to exactly one `PipelineRunRecord`.
- **Validation**:
  - `order` contiguous w.r.t. `enabled_steps`.
  - `duration_seconds >= 0`.
  - If `status == completed`, `ended_at` must be present.

### TelemetrySnapshot
- **Identifier**: composite (`run_id`, `timestamp_ms`).
- **Fields**:
  - `timestamp_ms` (int) — epoch milliseconds when sample recorded.
  - `step_id` (string) — current step at sampling time.
  - `steps_per_sec` (float | null) — measured throughput.
  - `fps` (float | null) — if visual env active.
  - `cpu_percent_process` (float | null) — psutil sample for process.
  - `cpu_percent_system` (float | null).
  - `memory_rss_mb` (float | null).
  - `gpu_util_percent` (float | null) — NVML sample.
  - `gpu_mem_used_mb` (float | null).
  - `notes` (string | null) — reason when metric unavailable ("sensor-unavailable").
- **Relationships**: belongs to `PipelineRunRecord`; optionally linked to `StepExecutionEntry.step_id`.
- **Validation**:
  - All percentages ∈ [0, 100].
  - Null allowed when sensors unavailable but `notes` must explain why.

### PerformanceRecommendation
- **Identifier**: composite (`run_id`, incremental index).
- **Fields**:
  - `trigger` (string) — rule name (e.g., `throughput_drop`, `cpu_sat`).
  - `severity` (`info` | `warning` | `critical`).
  - `message` (string) — human guidance.
  - `evidence` (object) — summary stats that triggered rule (baseline vs actual throughput, CPU %, etc.).
  - `suggested_actions` (array of strings) — concrete remediations ("increase workers to 4").
  - `timestamp_ms` (int) — when recommendation generated.
- **Relationships**: belongs to `PipelineRunRecord`.
- **Validation**:
  - `suggested_actions` non-empty.
  - `severity` consistent with thresholds (critical only on hard breaches).

### PerformanceTestResult
- **Identifier**: composite (`run_id`, `test_id`).
- **Fields**:
  - `test_id` (string) — slug of performance smoke scenario.
  - `matrix` (string) — reference to scenario matrix manifest.
  - `throughput_baseline` (float) — expected steps/sec.
  - `throughput_measured` (float).
  - `duration_seconds` (float).
  - `status` (`passed` | `soft-breach` | `failed`).
  - `recommendations_ref` (array) — pointers to `PerformanceRecommendation` entries generated from this test.
- **Relationships**: optional child of `PipelineRunRecord`; when perf tests run independently, `run_id` may be `perf-test::<timestamp>` to keep IDs consistent.
- **Validation**:
  - `status` derived from comparing measured to baseline thresholds (≥25% drop => `failed`).
  - When `status != passed`, at least one recommendation reference must exist.

## Data Flow
1. `PipelineRunRecord` created at run start and appended to JSON manifest.
2. Each step transitions `StepExecutionEntry` from `pending` → `running` → `completed/failed/skipped` with updated ETA/duration.
3. Telemetry sampling loop appends `TelemetrySnapshot` entries and optionally streams to TensorBoard.
4. Rule engine inspects snapshots + steps to emit `PerformanceRecommendation` entries.
5. If user triggers perf smoke test, `PerformanceTestResult` appended to same manifest (or dedicated `perf_tests` JSONL if run standalone) and cross-links to recommendations.

## Retention & Concurrency
- Every run writes under `output/run-tracker/<run_id>/manifest.jsonl` with a stable naming scheme to avoid collisions.
- A lightweight lock file (`run.lock`) prevents concurrent writers from clobbering manifests when multiple processes run simultaneously.
- A rotation policy keeps the last N manifests (default 20) while leaving older ones on disk unless user opts to prune.
