---
name: hybrid-learning-component-eval
description: Evaluate hybrid-learning components with mechanism-level evidence tables, config provenance,
  and claim boundaries.
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
output_schema: evidence_synthesis_summary.v1
---

# Hybrid Learning Component Eval

## When to use

Use this skill for the campaign lane named by the title when you need canonical command surfaces, expected artifacts, row-status handling, and conservative claim boundaries.

## Command surfaces

Start with hybrid-learning component configs, ablations, and report generators under `scripts/research/` and `scripts/tools/`. Prefer config-first commands and existing validation launch-packet scripts when available.

## Workflow

1. Record component boundary, mechanism evidence, ablation config, seed schedule, and artifact provenance.
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
