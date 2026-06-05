---
name: trace-mechanism-review
description: Review exact planner/scenario/seed/episode traces and videos without overgeneralizing from
  qualitative samples.
category: campaign-analysis
kind: analysis
phase: analysis
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: true
delegates_to:
- artifact-provenance
- evidence-synthesis
output_schema: trace_mechanism_summary.v1
---

# Trace Mechanism Review

## When to use

Use this skill for the campaign lane named by the title when you need canonical command surfaces, expected artifacts, row-status handling, and conservative claim boundaries.

## Command surfaces

Start with rendered trace/video artifacts, exact planner/scenario/seed/episode identifiers. Prefer config-first commands and existing validation launch-packet scripts when available.

## Workflow

1. Record claim status `supported | contradicted | unresolved` and no population-level inference from tiny samples.
2. Capture commit SHA, config path, command, output root, seeds, and expected artifacts.
3. Fill `mechanism_activation` from observed telemetry: activation status, activation count,
   command-source change, outcome change, and likely failure reason. Use `unknown` when the trace
   or report does not preserve the needed signal.
4. Classify each planner/scenario row with `benchmark-row-status` when benchmark rows exist.
5. Classify generated outputs with `artifact-provenance`.
6. Feed multi-source conclusions through `evidence-synthesis`.

## Guardrails

- Do not count fallback or degraded runs as success evidence.
- Do not make paper-facing or population-level claims beyond the inspected evidence.
- Treat mechanism activation as diagnostic unless the same report ties it to benchmark-valid row
  status, controlled baseline/intervention identity, and a measured outcome change.
- Keep raw generated outputs out of git unless they are deliberate compact fixtures.

## Output

Use the `output_schema` declared in frontmatter. The `mechanism_activation` block is:

```yaml
mechanism_activation:
  activated: true | false | unknown
  activation_count: integer | unknown
  changed_command_source: true | false | unknown
  changed_outcome: true | false | unknown
  likely_failure_reason: string
```
