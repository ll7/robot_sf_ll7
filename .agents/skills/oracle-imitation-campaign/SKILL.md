---
name: oracle-imitation-campaign
description: Analyze oracle-imitation campaign outputs with lineage, dataset, checkpoint, metric, and
  caveat discipline.
category: campaign-analysis
kind: analysis
phase: analysis
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: true
delegates_to:
- artifact-provenance
- benchmark-row-status
- evidence-synthesis
output_schema: benchmark_analysis_summary.v1
---

# Oracle Imitation Campaign

## When to use

Use this skill for the campaign lane named by the title when you need canonical command surfaces, expected artifacts, row-status handling, and conservative claim boundaries.

## Command surfaces

Start with `scripts/validation/validate_oracle_imitation_launch_packet.py` and imitation training/evaluation commands. Prefer config-first commands and existing validation launch-packet scripts when available.

## Workflow

1. Record dataset lineage, oracle source, checkpoint identity, seed schedule, and policy eligibility.
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
