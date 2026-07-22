# Issue #6095 S10 ORCA/PPO Nominal-vs-Stress Discriminability Calibration

**Status**: configs ready, preflight passed. Execution requires SLURM.

## Summary

Two issue-specific benchmark configs compare ORCA and PPO across the nominal
`nominal_v1.yaml` matrix (4 scenarios) and the stress
`classic_interactions_francis2023.yaml` matrix (48 scenarios). Both use
seeds 111-120 (`paper_eval_s10`), horizon 100, dt=0.1, differential-drive
kinematics.

## Configs

- `configs/benchmarks/issue_6095_nominal_discriminability_v1.yaml`
- `configs/benchmarks/issue_6095_stress_discriminability_v1.yaml`

Both configs include only ORCA (`algo: orca`) and PPO
(`algo: ppo`, `algo_config: configs/baselines/ppo_15m_grid_socnav.yaml`).

## Preflight Results

Both configs passed preflight validation:

| Property | Nominal | Stress |
|---|---|---|
| Scenarios | 4 | 48 |
| Planners | orca, ppo | orca, ppo |
| Seeds | 111-120 (10) | 111-120 (10) |
| Expected rows | 80 | 960 |
| PPO checkpoint resolved | yes | yes |
| PPO checkpoint status | stageable_remote | stageable_remote |
| PPO checkpoint SHA256 | `2b30df81...` | `2b30df81...` |

## Tests

`tests/benchmark/test_issue_6095_s10_discriminability_configs.py` validates:

- Planner entries are ORCA and PPO only
- PPO algo_config points at existing `ppo_15m_grid_socnav.yaml`
- Seed policy uses `paper_eval_s10` (111-120)
- Resolved seed inventory matches S10 exactly
- Horizon=100, dt=0.1, kinematics=differential_drive
- Both configs share same planner rows, seed policy, horizon/dt/kinematics
- Expected row counts: 80 (nominal), 960 (stress)
- Configs reference different scenario matrices

## Evidence

Compact preflight outputs archived at:
`docs/context/evidence/issue_6095_s10_discriminability_2026-07-22/`

## Execution Requirements

Full benchmark execution requires SLURM. The PPO checkpoint
(`ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`)
is stageable-remote and must be staged before execution.

## PPO Model Provenance

- **Checkpoint**: `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`
- **Source job**: 11724 (auxme-imech093, L40s, 8h04m)
- **WandB**: ll7/robot_sf/ibo3aqus
- **Training config**: `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity.yaml`
- **Checkpoint SHA256**: `2b30df812bfcc737924b126b0763d69c567fe20716dc1c1eba8f56f926b49c1d`

### Overlap Caveat

Per the PPO baseline doc (`configs/baselines/ppo_15m_grid_socnav.yaml`):

> This policy was trained on the eval superset
> `configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`.

That eval superset includes `classic_interactions_francis2023.yaml` (the stress
matrix) and `atomic_navigation_minimal_full_v1.yaml`. This means:
- **Stress matrix evaluation is in-distribution** for PPO.
- **Nominal matrix evaluation** uses scenarios (`empty_map_8_directions_east`,
  `single_ped_crossing_orthogonal`, `classic_doorway_low`,
  `classic_bottleneck_low`) that do not appear in the documented training set
  components, but may overlap through atomic archetype inclusion.
- This does not block the ORCA-vs-PPO comparison (both planners see the same
  scenarios), but limits any generalization or planner-family superiority claims.

## Claims and Limitations

This is proposal-phase evidence. No benchmark results exist yet because execution
requires SLURM (`compute_submit` not authorized in current lane).
Configs are validated and ready for submission. See issue #6095 for the full
analysis contract.

## References

- Issue #6095: benchmark campaign specification
- Issue #1344: paired nominal/stress AMV protocol (parent config pattern)
- `docs/context/issue_691_benchmark_fallback_policy.md`
- `docs/context/artifact_evidence_vocabulary.md`
- `configs/baselines/ppo_15m_grid_socnav.yaml`
