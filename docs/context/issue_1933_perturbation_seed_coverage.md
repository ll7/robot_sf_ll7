# Issue #1933 Perturbation Seed Coverage

Issue: [#1933](https://github.com/ll7/robot_sf_ll7/issues/1933)
Parent: [#1610](https://github.com/ll7/robot_sf_ll7/issues/1610)
Predecessor: [#1904](https://github.com/ll7/robot_sf_ll7/issues/1904)
Successor: [#1935](https://github.com/ll7/robot_sf_ll7/issues/1935)

## Goal

Expand the first #1610 paired perturbation pilot from one seed per variant to the full four-seed
manifest slice while keeping the manifest, perturbation family, and cheap planner set fixed.

This is diagnostic local evidence only. It is not benchmark-strength or paper-facing evidence.

## Command

```bash
uv run python scripts/validation/run_scenario_perturbation_criticality_pilot.py \
  configs/scenarios/perturbations/issue_1858_seed_sensitive_pilot_v1.yaml \
  --materialized-output-dir output/scenario_perturbations/issue1933_seed4/materialized \
  --pilot-output-dir output/scenario_perturbations/issue1933_seed4/pilot \
  --seed-limit 4 \
  --horizon 80 \
  --workers 1 \
  --evidence-summary docs/context/evidence/issue_1933_perturbation_seed_coverage_2026-05-31/summary.json
```

## Result

- Materialized variants: 6 included, 0 excluded.
- Planners: `goal`, `orca`.
- Pair rows: 24 completed pairs, 0 excluded pairs.
- Mean deltas over completed pairs:
  - success delta: `0.0000`
  - collision delta: `0.0000`
  - timeout delta: `0.0000`
  - min-distance delta: `+0.0007 m`

Grouped min-distance deltas:

| Group | Pairs | Mean Min-Distance Delta |
|---|---:|---:|
| `goal` | 12 | `+0.0103 m` |
| `orca` | 12 | `-0.0089 m` |
| `classic_group_crossing_high` | 8 | `-0.0138 m` |
| `classic_head_on_corridor_low` | 8 | `+0.0119 m` |
| `francis2023_join_group` | 8 | `+0.0039 m` |
| `robot_route_offset` | 24 | `+0.0007 m` |

Interpretation: the route-offset perturbation family remains neutral for success, collision, and
timeout across the four-seed local pilot for `goal` and `orca`. Clearance shifts are tiny and
mixed by planner/scenario, so this evidence supports treating the current route-offset family as a
low-criticality diagnostic slice, not as a discovered robustness failure.

## Evidence Boundary

Tracked compact evidence:

- [summary.json](evidence/issue_1933_perturbation_seed_coverage_2026-05-31/summary.json)
- [SHA256SUMS](evidence/issue_1933_perturbation_seed_coverage_2026-05-31/SHA256SUMS)

Ignored local outputs:

- materialized matrix and route overrides under `output/scenario_perturbations/issue1933_seed4/`
- raw planner episode JSONL
- local `summary.json` / `summary.md` generated beside the raw outputs

Fallback, degraded, invalid, missing, and failed rows are classified separately by the pilot script
and excluded from completed-pair means. This run had no such rows.

## Routing

Successor [issue_1935_stronger_perturbation_planner.md](issue_1935_stronger_perturbation_planner.md)
added one stronger local planner candidate to this same route-offset slice. The next #1610 child
should again change exactly one dimension: add a second perturbation family while keeping the
planner/seed budget fixed, or keep route-offset fixed and inspect the stronger candidate's
trace-level clearance mechanism. Do not broaden both at once.
