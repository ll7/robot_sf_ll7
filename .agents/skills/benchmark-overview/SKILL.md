---
name: benchmark-overview
description: Fast benchmark-faithful orientation for scenario splits, baselines, metrics, artifacts, and
  reproducibility constraints in robot_sf_ll7.
category: benchmark-evidence
kind: atomic
phase: context
requires_write: false
requires_slurm: false
requires_benchmark_artifacts: true
delegates_to: []
output_schema: skill_run_summary.v1
---

# Benchmark Overview

## When to use

Use this skill when you need a fast, benchmark-faithful read of scenario definitions, seeds, baselines,
metrics, and artifact layout before changing code or producing claims.

## Read First

- `docs/benchmark_spec.md`
- `docs/benchmark.md`
- `docs/benchmark_planner_family_coverage.md`
- `docs/benchmark_camera_ready.md` when the task is publication-facing

## Workflow

1. Identify the active benchmark contract for the request.
2. Check scenario/config entrypoints and baseline family boundaries.
3. Map required reproducibility artifacts and seeds.
4. Return allowed conclusions and known caveats.

## Proof and Guardrails

- Verify artifact provenance and contract mapping before interpretation.
- Preserve fail-closed policy where fallback/degraded behavior exists.
- Avoid implying success from runs with incomplete seeds, mismatched episodes, or missing metrics.
- Use issue/context notes to capture non-obvious assumptions.

## Output

- benchmark contract and why it applies,
- what evidence supports the interpretation,
- known caveats and non-covered cases.
## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.
