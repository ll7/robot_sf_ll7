# Issue #4365 H600 Hybrid-vs-ORCA S30 Pre-Registration

This note pre-registers the arm-restricted 30-seed (`S30`) horizon-600 (`h600`) benchmark
configuration for the four Issue #4230 hybrid arms plus ORCA and PPO anchors. It is protocol
evidence only: it records the intended comparison surface before any campaign submission and does
not promote a benchmark, paper, or dissertation claim.

## Scope

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/4365>
- Config: `configs/benchmarks/paper_experiment_matrix_v1_h600_hybrid_vs_orca_s30.yaml`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Required scenario-matrix hash: `c10df617a87c`
- Expected expanded S30 scenario-plus-seed hash: `152eba3969a9`
- Horizon: `600`
- Seed set: `paper_eval_s30` from `configs/benchmarks/seed_sets_v1.yaml`, expanding seeds
  `111` through `140` inclusive.
- Roster: four Issue #4230 hybrid arms plus `orca` and `ppo`.

## Pre-Registered Roster

| Planner key | Role | Candidate config |
| --- | --- | --- |
| `scenario_adaptive_hybrid_orca_v1` | Issue #4230 hybrid arm | `configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v1.yaml` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | Issue #4230 hybrid arm | `configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard.yaml` |
| `hybrid_rule_v3_fast_progress_static_escape` | Issue #4230 hybrid arm | `configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape.yaml` |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | Issue #4230 hybrid arm | `configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape_continuous.yaml` |
| `orca` | ORCA baseline anchor | n/a |
| `ppo` | PPO learned-policy anchor | `configs/baselines/ppo_issue_791_eval_aligned_large_capacity.yaml` |

## Claim Boundary

The intended future analysis is an F-C4(ii) separation-strengthening test: whether hybrid
control-law arms retain advantage over non-hybrid baselines, here narrowed to whether the
targeted hybrid-vs-ORCA success lead survives the predeclared 30-seed schedule on the h600
surface. This PR does not run the campaign, submit Slurm or GPU work, archive rows, change
seed schedules, or edit paper/dissertation claims. No claim is promoted without author
sign-off and retained campaign evidence that satisfies the fail-closed benchmark-row policy.

The full-roster S30 campaign remains deferred by the 2026-07-03 S30 ruling. This config is a
reversible escalation path for the arm-restricted comparison only.

## Runtime Estimate

The closest prior h600 hybrid-roster run reference is job `13282`. Scaling from 3 seeds to 30
seeds increases the per-arm episode budget by roughly `10x`; adding ORCA and PPO changes the roster
from 4 to 6 arms. Expected campaign runtime should therefore be treated as approximately
`15x` the Issue #4230 h600 hybrid-roster budget before queue, dependency, and fallback effects.

## Preflight Contract

Static preflight must verify:

- config loads through `load_campaign_config`;
- source h600 scenario-matrix identity remains preregistered as `c10df617a87c`;
- expanded S30 scenario-plus-seed payload hashes to `152eba3969a9`;
- `paper_eval_s30` resolves exactly 30 seeds, `111..140`;
- planner roster is exactly the six keys listed above, in order;
- hybrid candidate config paths and PPO anchor config path resolve locally.

The canonical focused validation command is:

```bash
uv run pytest tests/benchmark/test_h600_hybrid_vs_orca_s30_config.py
```

## Terminality Status

PR #4368 landed the complete issue #4365 config and preflight-only slice on 2026-07-04:
the arm-restricted h600 S30 config, this pre-registration note, and the static identity
test. No distinct CPU implementation follow-up remains inside this issue scope.

Remaining empirical work is intentionally outside this terminality slice: authorized
campaign submission, Slurm/GPU execution, full-roster S30 expansion, retained-row archive
generation, benchmark interpretation, and paper/dissertation claim edits.
