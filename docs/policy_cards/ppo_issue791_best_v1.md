# Policy Card: `ppo_issue791_best_v1`

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1865>

## Summary

```yaml
policy_id: ppo_issue791_best_v1
policy_family: learned_baseline
card_status: current
registry_status:
  integration_status: implemented
  reproducibility_status: comparison_available
  benchmark_status: comparison_available
benchmark_track: grid_socnav_v1 model-promotion track plus scoped policy-search/paper-matrix reports
evidence_boundary: >
  Best current learned-only baseline for success-oriented comparison; not a safety promotion
  because the scoped paper-matrix collision rate is worse than ORCA.
not_for:
  - safety promotion
  - out-of-distribution generalization claim
  - unguarded replacement for collision-safer anchors
```

`ppo_issue791_best_v1` is the repository's current concrete learned-policy card example. It links a
policy-search candidate, PPO baseline config, model-registry artifact, learned-policy registry row,
adapter mapping, and recorded smoke/nominal/paper-matrix evidence.

## Source Links

| Source | Path |
| --- | --- |
| Learned-policy registry row | [`docs/context/policy_search/learned_policy_registry.md`](../context/policy_search/learned_policy_registry.md) |
| Candidate registry row | [`docs/context/policy_search/candidate_registry.yaml`](../context/policy_search/candidate_registry.yaml) |
| Candidate config | [`configs/policy_search/candidates/ppo_issue791_best_v1.yaml`](../../configs/policy_search/candidates/ppo_issue791_best_v1.yaml) |
| Baseline config | [`configs/baselines/ppo_15m_grid_socnav.yaml`](../../configs/baselines/ppo_15m_grid_socnav.yaml) |
| Adapter interface mapping | [`docs/context/issue_1618_learned_policy_adapter_interface.md`](../context/issue_1618_learned_policy_adapter_interface.md) |
| Model registry | [`model/registry.yaml`](../../model/registry.yaml) |
| Best-learning report | [`docs/context/policy_search/reports/2026-05-05_best_learning_policy.md`](../context/policy_search/reports/2026-05-05_best_learning_policy.md) |
| Promotion verdict | [`docs/context/issue_791_best_policy_verdict_2026_04_21.md`](../context/issue_791_best_policy_verdict_2026_04_21.md) |
| Narrow claim memory | [`memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md`](../../memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md) |

## Contracts

| Field | Value |
| --- | --- |
| Observation contract | PPO dict observation with `tracked_agents_no_noise` promotion metadata and predictive-foresight features from `model/registry.yaml` and `configs/baselines/ppo_15m_grid_socnav.yaml`. |
| Allowed observation keys | Listed in `model/registry.yaml` under the `benchmark_promotion.allowed_observation_keys` for `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`. |
| Predictive dependency | `predictive_foresight_enabled: true`, `predictive_foresight_model_id: predictive_proxy_selected_v2_full`. |
| Action contract | Deterministic unicycle velocity command through the PPO planner wrapper, with `v_max: 2.0` and `omega_max: 1.0`. |
| Adapter implementation | `robot_sf/baselines/ppo.py`. |
| Fallback policy | `fallback_to_goal: false`; missing or invalid model action should fail closed rather than silently switching to goal seeking. |

## Artifacts

| Field | Value |
| --- | --- |
| Model id | `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417` |
| W&B run | `ll7/robot_sf/ibo3aqus` |
| W&B artifact | `ll7/robot_sf/ppo_expert_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity-best-success:v9` |
| Public artifact source | GitHub release `artifact/models-2026-05-registry-v1` via `model/registry.yaml`. |
| Checkpoint checksum | `2b30df812bfcc737924b126b0763d69c567fe20716dc1c1eba8f56f926b49c1d` |
| Local checkpoint path | `output/model_cache/ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417/model.zip`; cache path only and may be absent in a fresh checkout. |
| Training config | [`configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity.yaml`](../../configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity.yaml) |
| Training data or split | In-repo PPO training through the training config above; model-registry and baseline notes identify the eval superset `ppo_full_maintained_eval_v1` caveat. No separate offline dataset artifact is declared for this policy card. |
| Benchmark track | `grid_socnav_v1` in `model/registry.yaml`; policy-search evidence also includes the named smoke, nominal-sanity, and scoped paper-matrix reports in this card. |
| Normalizer URI | `unknown`; no separate durable normalizer URI is declared in the current model registry row. |
| License/access | Local repository policy plus W&B/GitHub release artifact access as recorded in `model/registry.yaml`; no separate third-party policy license is declared for this local PPO baseline. |

## Evidence

| Scope | Evidence |
| --- | --- |
| Training/eval promotion | Best eval summary in `model/registry.yaml`: 70 episodes on `ppo_full_maintained_eval_v1`, success `0.929`, collision `0.071`, SNQI `0.353`, step `9,961,472 / 10,000,000`. |
| Policy-search smoke | `docs/context/policy_search/reports/2026-05-05_best_learning_policy.md`: smoke success `1.0000`, collision `0.0000`, near misses `0.0000`. |
| Nominal sanity | Same report: 18 episodes, success `0.2778`, collision `0.0000`, near misses `0.2222`; decision `revise`. |
| Scoped paper matrix | Same report: PPO success `0.2569`, collision `0.0903`, near misses `3.3403`, TTG norm `0.9305`; ORCA reference success `0.1806`, collision `0.0347`, near misses `4.9097`, TTG norm `0.9650`. |
| Benchmark interpretation | Comparison evidence exists for the named config/checkpoint/stage only. PPO improves success in the scoped comparison but has worse collision rate than ORCA, so it is not a safety promotion. |

## Known Failures And Caveats

- Training used the eval superset identified in `configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`; report those numbers as in-distribution benchmark-set performance, not OOD generalization.
- The nominal-sanity pass still has many non-success episodes from low-progress timeouts and
  intrusive near misses, according to the 2026-05-05 best-learning report.
- The scoped paper-matrix collision rate is worse than ORCA, so collision-safer planners remain the
  safety anchor.
- The local checkpoint path is an ignored cache path; durable recovery depends on the registry's
  W&B/GitHub release artifact metadata.
- The card does not establish a reusable normalizer artifact because no separate normalizer URI is
  declared in the current registry metadata.

## Review Notes

Update this card when any of these surfaces change:

- `docs/context/policy_search/learned_policy_registry.md`
- `docs/context/policy_search/candidate_registry.yaml`
- `model/registry.yaml`
- `configs/policy_search/candidates/ppo_issue791_best_v1.yaml`
- `configs/baselines/ppo_15m_grid_socnav.yaml`
- `docs/context/policy_search/reports/2026-05-05_best_learning_policy.md`
