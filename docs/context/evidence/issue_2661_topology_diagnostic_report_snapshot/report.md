# Topology Diagnostic Report Snapshot: issue-2661-smoke

Diagnostic-only evidence; this is not planner-promotion or benchmark success evidence.

- Claim boundary: `diagnostic_only_not_benchmark_success`
- Trace count: 2
- Total steps: 6
- Diagnostic status counts: `{'diagnostic_complete': 2}`
- Terminal outcome counts: `{'success': 1, 'truncated': 1}`

## Selected Hypotheses

- `divert_left`: 1
- `primary_route`: 3

## Near-Parity Gate Reasons

- `eligible_near_parity_alternative`: 2
- `route_distance_exceeds_slack`: 1
- `static_clearance_below_floor`: 1

## Reuse-Penalty Activations

- Applied steps: 2
- Eligible near-parity alternative steps: 3
  - `cooldown_eligible`: 2

## Route-Progress Deltas

| Scenario | Seed | Rank | Samples | Progress delta (m) |
|---|---:|---:|---:|---:|
| double_bottleneck_high | 42 | 0 | 3 | 5.3 |
| double_bottleneck_high | 42 | 1 | 3 | 1.3 |
| narrow_corridor | 99 | 0 | 3 | 0.0 |
| narrow_corridor | 99 | 1 | 3 | -1.7 |

## Top Regressions

| Scenario | Seed | Rank | Progress delta (m) | Last corridor |
|---|---:|---:|---:|---|
| narrow_corridor | 99 | 1 | -1.7 | side_passage |

## Top Unchanged Cases

| Scenario | Seed | Rank | Progress delta (m) | First corridor |
|---|---:|---:|---:|---|
| narrow_corridor | 99 | 0 | 0.0 | main_corridor |
