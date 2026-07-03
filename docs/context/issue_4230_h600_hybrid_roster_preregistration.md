# Issue #4230 H600 Hybrid-Roster Pre-Registration

This note pre-registers the missing horizon 600 (`h600`) hybrid-roster benchmark slice before any
Slurm submission or benchmark interpretation. It is protocol evidence only: it does not report
campaign results, planner rankings, or paper/dissertation claims.

## Scope

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/4230>
- Parent issue: #3810
- Feeds: #4195 F-C4(ii) promotion gate
- Config: `configs/benchmarks/paper_experiment_matrix_v1_h600_hybrid_roster.yaml`
- Source h600 comparison jobs: `13268` and `13273`
- Required scenario matrix hash: `c10df617a87c`
- Seeds: `eval` seed set from `configs/benchmarks/seed_sets_v1.yaml`, expanding to `[111, 112, 113]`
- Claim boundary: no h600 ranking, F-C4(ii), paper, dissertation, or release claim before the
  #4195 aggregation/interpretation gate accepts the comparable rows.

## Pre-Registered Roster

The four arms are the verified h500 hybrid/scenario-adaptive leaders from the h500 candidate
comparison and policy-search registry:

| Planner key | Candidate config |
| --- | --- |
| `scenario_adaptive_hybrid_orca_v1` | `configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v1.yaml` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | `configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard.yaml` |
| `hybrid_rule_v3_fast_progress_static_escape` | `configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape.yaml` |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | `configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape_continuous.yaml` |

Roster provenance:

- `docs/context/issue_1454_s10_h500_candidate_comparison.md` records these four rows above ORCA and
  PPO on the h500 candidate comparison surface.
- `docs/context/policy_search/candidate_registry.yaml` records all four as implemented runnable
  hybrid-rule candidates.
- `configs/benchmarks/issue_1454_s10_scenario_horizons_h500_candidates.yaml` carries the same h500
  benchmark-admitted candidate family.

## Hypothesis

Constraint-first hybrid planners should land at or above the ORCA/PPO h600 baselines. If they do
not, the result is a real long-horizon finding that should revise the F-C4(ii) interpretation rather
than invalidate the campaign.

## No-Submit CPU Preflight

Run this before private queue submission:

```bash
scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_h600_hybrid_roster.yaml \
  --output-root output/issue_4230_h600_hybrid_roster_cpu_smoke \
  --label issue4230-h600-hybrid-smoke \
  --mode preflight \
  --log-level WARNING
```

The preflight is disposable local proof only. It must verify config loading, scenario-matrix hash
`c10df617a87c`, AMV/comparability artifact generation, seed surface `[111, 112, 113]`, and candidate
config resolution without Slurm/GPU submission.
