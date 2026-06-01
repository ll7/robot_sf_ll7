# Issue #1951 Intersection-Wait Phase Grid

Issue: [#1951](https://github.com/ll7/robot_sf_ll7/issues/1951)
Parent: [#1610](https://github.com/ll7/robot_sf_ll7/issues/1610)
Depends on: [PR #1950](https://github.com/ll7/robot_sf_ll7/pull/1950) (merged)
Predecessors: [#1941](https://github.com/ll7/robot_sf_ll7/issues/1941),
[#1943](https://github.com/ll7/robot_sf_ll7/issues/1943),
[#1947](https://github.com/ll7/robot_sf_ll7/issues/1947),
[#1949](https://github.com/ll7/robot_sf_ll7/issues/1949)

## Goal

Run a compact local phase-grid sweep for `francis2023_intersection_wait` pedestrian `h1` using the
existing perturbation families: start-delay offset, speed offset, and wait-duration offset. This
note is diagnostic local evidence only; it is not benchmark-strength or paper-facing evidence.

## Implementation

The tracked manifest is
`configs/scenarios/perturbations/issue_1610_intersection_wait_phase_grid_v1.yaml`. It contains one
no-op baseline and the requested magnitude rows:

- `single_pedestrian_start_delay_offset`: `-0.5 s`, `+0.5 s`, `+1.0 s`;
- `single_pedestrian_speed_offset`: `-0.25 m/s`, `+0.25 m/s`, `+0.5 m/s`;
- `single_pedestrian_wait_duration_offset`: `-0.5 s`, `+0.5 s`, `+1.0 s`.

The `-0.5 s` start-delay row is intentionally represented in the manifest but excluded from paired
execution by preflight, because `h1` has no baseline `start_delay_s` and the offset would make the
value negative. That row is a fail-closed limitation, not evidence.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
uv run python scripts/validation/run_scenario_perturbation_criticality_pilot.py \
  configs/scenarios/perturbations/issue_1610_intersection_wait_phase_grid_v1.yaml \
  --materialized-output-dir output/scenario_perturbations/issue1951_intersection_wait_phase_grid/materialized \
  --pilot-output-dir output/scenario_perturbations/issue1951_intersection_wait_phase_grid/pilot \
  --seed-limit 4 \
  --horizon 80 \
  --dt 0.1 \
  --workers 1 \
  --planner goal \
  --planner orca \
  --planner scenario_adaptive_hybrid_orca_v2_collision_guard \
  --evidence-summary docs/context/evidence/issue_1951_intersection_wait_phase_grid_2026-06-01/summary.json
```

## Result

Preflight validated the manifest/schema and found 9 eligible materialized variants: one no-op
baseline plus eight perturbation rows. It excluded only
`francis2023_intersection_wait_start_delay_h1_m050` with
`single_pedestrian_start_delay_offset would make pedestrian 'h1' start_delay_s negative`.

The pilot ran 81 episodes: 9 materialized variants x 3 seeds `[240, 241, 242]` x 3 planners. It
completed 72/72 no-op-versus-perturbation pairs with no fallback, degraded, missing, invalid, or
failed pair statuses. All completed-pair mean deltas for success, collision, and timeout were
`0.0`.

| Family | Magnitude | Completed Pairs | Mean Min-Distance Delta | Success | Collision | Timeout |
|---|---:|---:|---:|---:|---:|---:|
| start delay | `-0.5 s` | 0 | excluded: negative start delay | n/a | n/a | n/a |
| start delay | `+0.5 s` | 9 | `+4.159358 m` | `0.0` | `0.0` | `0.0` |
| start delay | `+1.0 s` | 9 | `+4.159358 m` | `0.0` | `0.0` | `0.0` |
| speed | `-0.25 m/s` | 9 | `+2.172367 m` | `0.0` | `0.0` | `0.0` |
| speed | `+0.25 m/s` | 9 | `-2.002917 m` | `0.0` | `0.0` | `0.0` |
| speed | `+0.5 m/s` | 9 | `-3.862581 m` | `0.0` | `0.0` | `0.0` |
| wait duration | `-0.5 s` | 9 | `0.0 m` | `0.0` | `0.0` | `0.0` |
| wait duration | `+0.5 s` | 9 | `0.0 m` | `0.0` | `0.0` | `0.0` |
| wait duration | `+1.0 s` | 9 | `0.0 m` | `0.0` | `0.0` | `0.0` |

Family means over completed pairs:

| Family | Completed Pairs | Mean Min-Distance Delta |
|---|---:|---:|
| start delay | 18 | `+4.159358 m` |
| speed | 27 | `-1.231043 m` |
| wait duration | 27 | `0.0 m` |

Interpretation: in this short-horizon local diagnostic, speed offsets show the clearest signed
closest-clearance response: slowing `h1` increases clearance, while speeding `h1` decreases it and
the `+0.5 m/s` point is more negative than `+0.25 m/s`. Positive start-delay offsets reproduce the
large positive closest-clearance response from #1941/#1947 but do not distinguish `+0.5 s` from
`+1.0 s` on this boundary. Wait-duration offsets remain flat across all tested magnitudes.

Observed evidence stops there. The grid does not show outcome changes, does not prove planner
robustness, and should not be generalized beyond `francis2023_intersection_wait`, `h1`, seeds
`[240, 241, 242]`, horizon `80`, dt `0.1`, and the three planners above.

## Evidence Boundary

Tracked compact evidence:

- [summary.json](evidence/issue_1951_intersection_wait_phase_grid_2026-06-01/summary.json)
- [family_magnitude_summary.json](evidence/issue_1951_intersection_wait_phase_grid_2026-06-01/family_magnitude_summary.json)
- [README.md](evidence/issue_1951_intersection_wait_phase_grid_2026-06-01/README.md)
- [SHA256SUMS](evidence/issue_1951_intersection_wait_phase_grid_2026-06-01/SHA256SUMS)

Ignored local outputs:

- preflight report under
  `output/scenario_perturbations/issue1951_intersection_wait_phase_grid/preflight/`
- materialized matrix under
  `output/scenario_perturbations/issue1951_intersection_wait_phase_grid/materialized/`
- raw planner JSONL and local Markdown summary under
  `output/scenario_perturbations/issue1951_intersection_wait_phase_grid/pilot/`

The preflight and pilot emitted the known `uni_campus_big.svg` invalid obstacle warning during
combined scenario loading. The warning did not affect the eligible `francis2023_intersection_wait`
rows.

## Routing

Treat any downstream search-policy or benchmark work as a separate issue with a stronger proof bar.
