# Issue #6095 S10 ORCA/PPO Nominal-vs-Stress Discriminability Calibration

**Status**: proposal-phase preflight complete; full execution requires SLURM and is not benchmark
evidence.

## Summary

Two issue-specific benchmark configs compare ORCA and PPO across the nominal
`nominal_v1.yaml` matrix (4 scenarios) and the stress
`classic_interactions_francis2023.yaml` matrix (48 scenarios). Both use
the frozen S10 (ten-seed) schedule 111-120 (`paper_eval_s10`), horizon 100, dt=0.1, differential-drive
kinematics.

## Configs

- `configs/benchmarks/issue_6095_nominal_discriminability_v1.yaml`
- `configs/benchmarks/issue_6095_stress_discriminability_v1.yaml`

Both configs include only ORCA (`algo: orca`) and PPO
(`algo: ppo`, `algo_config: configs/baselines/ppo_15m_grid_socnav.yaml`).

Both configs also use
`configs/benchmarks/route_clearance_certifications_v1.yaml`, preserving the
reviewed caveats for intentionally tight route geometry. ORCA uses
`socnav_missing_prereq_policy: fail-fast`: a missing native prerequisite ends
the campaign preflight rather than creating a fallback result row.

## Preflight Results

Both configs passed metadata-only preflight on source revision
`c24325e76a7c831941e0efe7ac8b25e231b9574b`. The packet records each
repository-relative source config and its full SHA-256:

| Property | Nominal | Stress |
|---|---|---|
| Scenarios | 4 | 48 |
| Planners | orca, ppo | orca, ppo |
| Seeds | 111-120 (10) | 111-120 (10) |
| Expected rows | 80 | 960 |
| Config hash | `47d684f55e8c1377` | `0375e182d186a8bc` |
| Source config SHA-256 | `3bf27cc362055e6874125f93b793c70f099ce6049641b60c1cb69974b3a55df7` | `e8f8b56097964568da4784054d23e1c590c14d32634c8ec6d465f735d1208dc6` |
| Route-clearance warnings | 2: 1 certified, 1 unresolved | 15: 15 certified, 0 unresolved |
| Native ORCA prerequisite | `rvo2` import passed | `rvo2` import passed |
| PPO checkpoint resolved | yes | yes |
| PPO checkpoint status | stageable_remote | stageable_remote |
| Submit safe | no (metadata-only) | no (metadata-only) |
| PPO checkpoint SHA256 | `2b30df81...` | `2b30df81...` |

The 15 stress warnings carry the reviewed Issue #1105 geometry caveats; they
are not uncaveated planner-attribution evidence. The nominal
`empty_map_8_directions_east` warning remains an unresolved map-level warning,
so a later campaign must preserve that caveat rather than infer a planner
effect from it.

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
- Route-clearance certification provenance, fail-fast ORCA prerequisite policy,
  and repository-relative route-override preview provenance

## Evidence

Compact preflight outputs archived at:
`docs/context/evidence/issue_6095_s10_discriminability_2026-07-22/`

The tracked packet keeps provenance paths repository-relative and uses LF line
endings for CSV artifacts. It is a portable config/preflight record, not raw
campaign evidence.

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

This is proposal-phase preflight evidence. No benchmark results or
discriminability conclusion exist because execution requires SLURM
(`compute_submit` is not authorized in this lane). Native ORCA/PPO execution
must complete without fallback or degraded rows before any result is promoted.
See issue #6095 for the full analysis contract.

## References

- Issue #6095: benchmark campaign specification
- Issue #1344: paired nominal/stress AMV protocol (parent config pattern)
- `docs/context/issue_691_benchmark_fallback_policy.md`
- `docs/context/artifact_evidence_vocabulary.md`
- `configs/baselines/ppo_15m_grid_socnav.yaml`
