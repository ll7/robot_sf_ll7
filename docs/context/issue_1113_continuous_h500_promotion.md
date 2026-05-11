# Issue #1113 Continuous H500 Promotion Matrix

Date: 2026-05-10

Related issues and PRs:

- #1113: <https://github.com/ll7/robot_sf_ll7/issues/1113>
- Parent #1059: <https://github.com/ll7/robot_sf_ll7/issues/1059>
- Predecessor #1034: <https://github.com/ll7/robot_sf_ll7/issues/1034>
- Related #884: <https://github.com/ll7/robot_sf_ll7/issues/884>

## Goal

Run the full `full_matrix_h500` promotion-scale validation for
`hybrid_rule_v3_fast_progress_static_escape_continuous` before treating the #1034 targeted
continuous-corridor recovery as broader benchmark-strengthening evidence.

## Command

```bash
rtk timeout 7200s env LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg \
  SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
  uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape_continuous \
  --stage full_matrix_h500 \
  --allow-expensive-stage \
  --workers 2 \
  --output-dir output/policy_search/hybrid_rule_v3_fast_progress_static_escape_continuous/full_matrix_h500/issue1113_continuous_h500
```

The stage is marked `requires_slurm: true` in `configs/policy_search/funnel.yaml`; this run used
the local machine because `local.machine.md` disallows SLURM submission. The run kept the stage's
configured two workers and produced 144 full h500 episodes.

## Results

| Candidate | Success | Collision | Near Miss | Classic Collision | Francis Collision |
|---|---:|---:|---:|---:|---:|
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | 0.9167 | 0.0139 | 0.3958 | 0.0145 | 0.0133 |
| `scenario_adaptive_hybrid_orca_v1` | 0.9097 | 0.0208 | 0.4236 | 0.0290 | 0.0133 |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | 0.9028 | 0.0139 | 0.4236 | 0.0145 | 0.0133 |

Failure taxonomy for the continuous candidate:

- `near_miss_intrusive`: 5
- `timeout_low_progress`: 4
- `static_collision`: 2
- `overconservative_stop`: 1

Scenario-family split:

- `classic`: 69 episodes, success 0.9275, collision 0.0145, near miss 0.4783
- `francis2023`: 75 episodes, success 0.9067, collision 0.0133, near miss 0.3200

## Gate Decision

`scripts/tools/promote_policy_search_candidate.py` evaluated the summary against
`configs/policy_search/promotion_gates.yaml` using the candidate's registered `tier_b` gate.
All checks passed:

- candidate registered,
- gate configured,
- promotion-scale stage accepted,
- success-rate threshold,
- aggregate collision threshold,
- classic-family collision threshold,
- Francis-family collision threshold.

The candidate therefore earns broader h500 promotion consideration. The #1034 targeted-only caveat
is superseded for full-matrix h500 evidence, but the remaining failure taxonomy above must still be
reported when using this planner as paper-facing evidence.

## Evidence

Compact tracked bundle:

- `docs/context/evidence/issue_1113_continuous_h500_2026-05-10/continuous_candidate_summary.json`
- `docs/context/evidence/issue_1113_continuous_h500_2026-05-10/promotion_decision.md`
- `docs/context/evidence/issue_1113_continuous_h500_2026-05-10/comparison.md`

Ignored local output:

- `output/policy_search/hybrid_rule_v3_fast_progress_static_escape_continuous/full_matrix_h500/issue1113_continuous_h500/`

The ignored output is about 4.4 MiB and contains raw JSONL records plus generated algo YAML files.
It is reproducible from tracked configs and remains out of git.

