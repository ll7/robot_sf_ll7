---
name: evidence-synthesis
description: Synthesize multiple issues, configs, seeds, metrics, and artifacts into conservative mechanism-level
  conclusions with caveats.
category: benchmark-evidence
kind: analysis
phase: analysis
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: true
delegates_to:
- artifact-provenance
- benchmark-row-status
- paper-facing-docs
output_schema: evidence_synthesis_summary.v1
---

# Evidence Synthesis

## When to use

Use this skill when multiple issues, campaigns, configs, seeds, metrics, and artifacts need one conservative conclusion.

## Workflow

1. Inventory source issues, PRs, context notes, configs, seeds, commands, artifacts, and metric tables.
2. Classify each evidence row with `benchmark-row-status` when benchmark data is involved.
3. Classify artifacts with `artifact-provenance`.
4. Build the required synthesis table.
5. Separate observed evidence from hypothesis and state caveats before conclusions.
6. For mixed or limited benchmark evidence, open the synthesis with the claim boundary: evidence
   tier, fallback/degraded exclusions, major caveats, and uncertainty before result interpretation.

## Required table

| Mechanism | Source issue | Evidence tier | Config | Seeds | Artifacts | Metrics | Verdict | Caveats |
|---|---:|---|---|---|---|---|---|---|

## Guardrails

- Do not upgrade exploratory evidence into paper-facing claims.
- Do not average away blocked, fallback, or degraded rows.
- Do not put rankings, success language, or mechanism conclusions before the claim boundary when
  evidence is diagnostic-only, smoke-level, mixed, fallback-tainted, or below high confidence.
- Use `paper-facing-docs` before manuscript-support language is published.

## Output

Use `evidence_synthesis_summary.v1`.
