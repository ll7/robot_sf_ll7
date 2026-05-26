---
name: carla-replay-parity
description: Review CARLA replay parity evidence with scenario, replay, metric, and limitation tracking.
category: campaign-analysis
kind: analysis
phase: analysis
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: true
delegates_to:
- benchmark-row-status
- artifact-provenance
- evidence-synthesis
output_schema: benchmark_analysis_summary.v1
---

# CARLA Replay Parity

## When to use

Use this skill for the campaign lane named by the title when you need canonical command surfaces, expected artifacts, row-status handling, and conservative claim boundaries.

## Command surfaces

Start with `scripts/carla_bridge/compare_oracle_replay_metrics.py` and CARLA replay fixtures. Prefer config-first commands and existing validation launch-packet scripts when available.

## Workflow

1. Record replay id, CARLA fixture source, metric parity threshold, and unsupported scenario caveats.
2. Capture commit SHA, config path, command, output root, seeds, and expected artifacts.
3. Classify each planner/scenario row with `benchmark-row-status` when benchmark rows exist.
4. Classify generated outputs with `artifact-provenance`.
5. Feed multi-source conclusions through `evidence-synthesis`.

## Guardrails

- Do not count fallback or degraded runs as success evidence.
- Do not make paper-facing or population-level claims beyond the inspected evidence.
- Keep raw generated outputs out of git unless they are deliberate compact fixtures.

## Output

Use the `output_schema` declared in frontmatter.
