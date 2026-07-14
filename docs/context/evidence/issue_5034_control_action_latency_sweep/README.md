<!-- AI-GENERATED (robot_sf#5034, 2026-07-14) - NEEDS-REVIEW -->
# Issue #5034 Control-action-latency sweep evidence 2026-07-14

Plain-language summary: this bundle promotes raw fidelity-campaign episode rows into a compact control-action-latency evidence summary. It reports the 0/100/300 ms-equivalent delay cells' success, collision, and minimum-clearance metrics and excludes any fallback/degraded/non-native rows. It is not paper-facing evidence.

- Schema: `control-action-latency-sweep-evidence-promotion.v2`
- Git head: `9e69b82e411a9efba3dc125593fd2c550cf15589`
- Raw rows: `ignored_output/fidelity_latency_raw/episode_rows.jsonl` (non-durable local input)
- Preflight decision: `ready`
- Evidence tier: `targeted smoke`
- Result classification: `diagnostic-only`
- Distance convention: `surface_clearance`
- Claim boundary: control-action-latency metric-evidence promotion only: reads raw fidelity-campaign episode rows, isolates the control_action_latency axis, and reports action-latency metadata plus success / collision / minimum-clearance metrics per native latency cell. It runs no episode and promotes no claim beyond the declared campaign evidence tier; it is not simulator-realism evidence, not sim-to-real evidence, and not paper-facing evidence.

## Scope

- Latency rows: `36` (results `36`, excluded `0`)
- Planners: `baseline_social_force, goal_seek`
- Seeds: `101, 102, 103`
- Latency-step coverage: required `[0, 1, 3]`, observed `[0, 1, 3]`, missing `none`

## Aggregate metrics per latency cell

| Planner | Latency steps | Latency ms | Cells | Success | Collision | Min clearance |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_social_force` | 0 | 0.0 | 6 | 0 | 0 | 3.707000020970527 |
| `baseline_social_force` | 1 | 100.0 | 6 | 0 | 0 | 3.785761483268774 |
| `baseline_social_force` | 3 | 300.0 | 6 | 0 | 0 | 4.084346606732809 |
| `goal_seek` | 0 | 0.0 | 6 | 0 | 1 | -0.044683821077205144 |
| `goal_seek` | 1 | 100.0 | 6 | 0 | 1 | -0.03975043027283831 |
| `goal_seek` | 3 | 300.0 | 6 | 0 | 1 | -0.0485444065216765 |

## Exclusions (fallback / degraded / non-native)

- Excluded rows: `0`
- Reasons: `none`

Per the issue #691 benchmark fallback policy, excluded rows never contribute to the result metrics above.

## Files

- `summary.json`: full promotion packet (aggregate + per-cell + exclusions).
- `per_cell_metrics.csv`: compact per-cell latency metrics table.
- `manifest.sha256`: checksums for promoted compact artifacts.
- `README.md`: this human-readable summary.
