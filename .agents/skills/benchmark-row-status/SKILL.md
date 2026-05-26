---
name: benchmark-row-status
description: Classify benchmark campaign rows under the fail-closed policy so fallback or degraded execution
  never counts as successful evidence.
category: benchmark-evidence
kind: policy
phase: analysis
requires_write: false
requires_slurm: false
requires_benchmark_artifacts: true
delegates_to: []
output_schema: benchmark_row_status.v1
---

# Benchmark Row Status

## When to use

Use this skill when classifying benchmark or campaign rows before summarizing success, exclusions, failures, fallback, degraded runs, or blocked planners.

## Statuses

- `successful_evidence`: native or accepted adapter execution satisfied the benchmark contract.
- `accepted_unavailable`: known unavailable or excluded planner/scenario with documented blocker.
- `unexpected_failure`: should have run but failed.
- `fallback`: fallback path ran; never counts as success evidence.
- `degraded`: degraded contract ran; never counts as success evidence without explicit degraded-mode scope.
- `blocked`: preflight or dependency prevented a valid run.

## Workflow

1. Read `docs/context/issue_691_benchmark_fallback_policy.md` and the campaign contract.
2. For each row, inspect planner mode, adapter mode, error field, status field, and metric completeness.
3. Set `counts_as_success_evidence` explicitly.
4. Link blockers or exclusion decisions where available.
5. Summarize counts separately from metric aggregation.

## Guardrails

- Fallback/degraded execution is a caveat, not a success.
- Do not hide unavailable rows inside aggregate success rates.
- Do not infer row validity from metric presence alone.

## Output

Use `benchmark_row_status.v1`.
