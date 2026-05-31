# Issue #1935 Stronger Perturbation Planner

Issue: [#1935](https://github.com/ll7/robot_sf_ll7/issues/1935)
Parent: [#1610](https://github.com/ll7/robot_sf_ll7/issues/1610)
Predecessor: [#1933](https://github.com/ll7/robot_sf_ll7/issues/1933)
Successor: [#1937](https://github.com/ll7/robot_sf_ll7/issues/1937)

## Goal

Add exactly one stronger local planner to the #1610 paired route-offset perturbation pilot while
keeping the seed count, manifest slice, horizon, and existing `goal`/`orca` baselines fixed.

This is diagnostic local evidence only. It is not benchmark-strength or paper-facing evidence.

## Runner Change

`scripts/validation/run_scenario_perturbation_criticality_pilot.py` now accepts a `--planner`
token that is either a raw benchmark algorithm or a policy-search candidate key from
`docs/context/policy_search/candidate_registry.yaml`.

For this run, `scenario_adaptive_hybrid_orca_v2_collision_guard` was resolved to:

- algorithm: `hybrid_rule_local_planner`
- config: `configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard.yaml`
- source: `policy_search_candidate`

The evidence label remains the candidate key so grouped summaries distinguish the candidate from
the raw benchmark algorithm.

## Command

```bash
uv run python scripts/validation/run_scenario_perturbation_criticality_pilot.py \
  configs/scenarios/perturbations/issue_1858_seed_sensitive_pilot_v1.yaml \
  --materialized-output-dir output/scenario_perturbations/issue1935_stronger_planner/materialized \
  --pilot-output-dir output/scenario_perturbations/issue1935_stronger_planner/pilot \
  --seed-limit 4 \
  --horizon 80 \
  --workers 1 \
  --planner goal \
  --planner orca \
  --planner scenario_adaptive_hybrid_orca_v2_collision_guard \
  --evidence-summary docs/context/evidence/issue_1935_stronger_perturbation_planner_2026-05-31/summary.json
```

## Result

- Materialized variants: 6 included, 0 excluded.
- Planners: `goal`, `orca`, `scenario_adaptive_hybrid_orca_v2_collision_guard`.
- Pair rows: 36 completed pairs, 0 excluded pairs.
- Mean deltas over completed pairs:
  - success delta: `0.0000`
  - collision delta: `0.0000`
  - timeout delta: `0.0000`
  - min-distance delta: `+0.0116 m`

Grouped min-distance deltas:

| Group | Pairs | Mean Min-Distance Delta |
|---|---:|---:|
| `goal` | 12 | `+0.0103 m` |
| `orca` | 12 | `-0.0089 m` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 12 | `+0.0335 m` |
| `classic_group_crossing_high` | 12 | `+0.0072 m` |
| `classic_head_on_corridor_low` | 12 | `+0.0237 m` |
| `francis2023_join_group` | 12 | `+0.0039 m` |
| `robot_route_offset` | 36 | `+0.0116 m` |

Interpretation: adding the stronger local policy-search candidate increases the observed clearance
response relative to the `goal`/`orca` seed-coverage pilot, especially for the candidate row itself.
The route-offset family remains neutral for success, collision, and timeout in this bounded local
slice, so the result is a useful planner-sensitivity diagnostic but not evidence of a robustness
failure or benchmark-strength perturbation-criticality claim.

## Evidence Boundary

Tracked compact evidence:

- [summary.json](evidence/issue_1935_stronger_perturbation_planner_2026-05-31/summary.json)
- [SHA256SUMS](evidence/issue_1935_stronger_perturbation_planner_2026-05-31/SHA256SUMS)

Ignored local outputs:

- materialized matrix and route overrides under `output/scenario_perturbations/issue1935_stronger_planner/`
- raw planner episode JSONL
- local `summary.json` / `summary.md` generated beside the raw outputs

Fallback, degraded, invalid, missing, and failed rows are classified separately by the pilot script
and excluded from completed-pair means. This run had no such rows.

## Routing

The next #1610 child should change exactly one dimension again. Useful next options are:

- add a second perturbation family while keeping this three-planner, four-seed pilot budget fixed,
- or keep route-offset fixed and use a targeted trace-level review for the candidate's larger
  clearance response before treating that mechanism as robust.

Successor [issue_1937_ped_route_offset.md](issue_1937_ped_route_offset.md) added the pedestrian
route-offset perturbation family while keeping the #1935 planner and seed budget fixed.
