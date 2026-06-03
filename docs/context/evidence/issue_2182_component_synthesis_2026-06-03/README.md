# Issue #2182 Component Synthesis Evidence

Status: synthesis evidence over the Issue #2180 h500 diagnostic run.

This bundle preserves the compact component-effect classification table derived from
`docs/context/evidence/issue_2180_one_factor_h500_2026-06-03/summary.json`. It is intentionally
small and reviewable; raw h500 episode rows remain in ignored `output/` paths from Issue #2180.

## Artifact

- `component_effects.csv`: component-level classifications and effect deltas.

## Figure-Like Summary

| Component | Classification | Main h500 effect |
| --- | --- | --- |
| Static escape only | neutral | No independent outcome gain; runtime cost. |
| Static recenter only | supported | `+0.056` success, no collision/near-miss penalty. |
| Recenter after static escape | supported | `+0.111` success; strongest positive signal. |
| Corridor-transit terms | neutral | No measurable effect after escape plus recentering. |
| Continuous static checks | trade-off | `-0.222` near-miss rate, but `-0.111` success. |
| Scenario-adaptive ORCA selector | weaker | `-0.056` success versus grouped static. |
| Speed/progress 2.4 | weaker | `-0.056` success and `+0.111` near-miss rate. |

The table is the durable figure-like artifact for this synthesis because the evidence has only
seven component comparisons; a chart would not add information beyond the classified deltas.
