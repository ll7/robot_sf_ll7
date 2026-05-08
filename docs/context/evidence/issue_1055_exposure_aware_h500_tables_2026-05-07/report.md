# Issue 1055 Exposure-Aware H500 Tables

Generated from the retained issue #1049 representative trace summary.

## Tables

- `fixed_vs_h500_outcome_table.csv`: success, collision, duration, and #1056 classification by representative cell.
- `exposure_aware_trace_table.csv`: near-miss, force-exposure, comfort-exposure, and duration-normalized rates by variant.

## Interpretation

These tables are representative trace tables, not a full h500 campaign ranking. They show why h500
must not be reported as a single success table: one case is clean time-budget relief, one succeeds
with higher exposure/comfort pressure, and one longer run reaches collision.

The retained traces support per-step and per-second rates for `near_misses`, `force_exceed_events`,
and `comfort_exposure`. The selected traces record zero discrete near-miss events, so near-miss rates
are zero for this pilot. The exposure-enabled case is supported by force/comfort exposure and reduced
minimum pedestrian distance, not by near-miss timing.

Fallback/degraded rows are not present in this pilot table. Future full-campaign h500 reports must
carry planner execution mode and exclude or caveat fallback/degraded rows before aggregation.
