# Issue #2269 Research-v1 Trace Case Selection

This directory preserves the compact analysis-only case-selection manifest for Issue #2269 /
parent Issue #2159.

## Result

Five candidate trace-review cases are selected. Three have durable compact trace-slice artifacts;
two are research-v1 AMV-specific but still require renderable trace exports before a trace-viewer
pack can use them.

## Tracked Files

- `case_selection_manifest.yaml`: selected cases, source evidence, rationale, row status, required
  render inputs, blockers, and recommended next action.

## Claim Boundary

This is case-selection evidence only. It is not rendering, pack assembly, benchmark-strength
planner evidence, or paper-facing failure analysis.
