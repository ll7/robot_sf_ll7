# Issue 602 Guarded PPO Safety-Aware Profile Note

## Goal

Formalize `guarded_ppo` as the current internal safety-aware planner-family representative, document
its benchmark contract, and test whether it is strong enough to matter on the canonical paper
surface.

## Canonical Internal Profile

- Planner key: `guarded_ppo`
- Algo config: `configs/algos/guarded_ppo_camera_ready.yaml`
- Canonical PPO model:
  `ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200`
- Benchmark comparison config:
  `configs/benchmarks/paper_experiment_matrix_v1_guarded_ppo_compare.yaml`

Matrix-level interpretation note:

- `baseline-ready-core` means the matrix is paper-facing and anchored to the core baseline set,
  but it may still include experimental challenger rows for comparison.
- It is intentionally separate from planner-level readiness labels such as `planner_group` and
  `benchmark_profile`.

## Planner-Facing Contract

`guarded_ppo` keeps PPO as the primary controller but adds a short-horizon near-field safety guard.
The configured contract is:

- primary policy: PPO network inference in `dict` observation mode
- command space: native `unicycle_vw`
- guard rollout: `guard_rollout_dt=0.2`, `guard_rollout_steps=6`
- hard pedestrian clearance: `0.58`
- first-step pedestrian clearance: `0.72`
- hard obstacle clearance: `0.30`
- minimum TTC: `0.70`
- fallback controller: internal `risk_dwa` profile

Observed benchmark metadata from the live run confirms:
- `canonical_algorithm: guarded_ppo`
- `policy_semantics: guarded_policy_network_inference`
- `execution_mode: mixed`
- `adapter_name: guarded_ppo_action_to_unicycle`
- kinematics projection rate remained very low: `0.00155`

## Guard / Intervention Semantics

The controller is not a learned safety policy in the SoNIC sense.
It is a guarded execution wrapper around a PPO policy:

- PPO proposes native robot actions.
- A short rollout checks near-field crowd and obstacle safety.
- If the guard predicts unsafe interaction, control falls back to the configured `risk_dwa`
  reactive controller.
- If PPO inference fails, `fallback_to_goal: true` allows a fail-open path toward simple goal
  pursuit instead of strict fail-closed stopping.

That makes the current implementation a lightweight safety-aware profile, not a full safe-RL or
uncertainty-aware planner.

## Benchmark Comparison

### Validation Command

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_guarded_ppo_compare.yaml \
  --label issue602_guarded_ppo_compare_retry \
  --log-level WARNING
```

### Canonical Artifact

- Final campaign:
  `output/benchmarks/camera_ready/paper_experiment_matrix_v1_guarded_ppo_compare_issue602_guarded_ppo_compare_retry_20260321_153203`

### Final Result

| Planner | Success | Collisions | SNQI | Runtime (s) | Near Misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ppo` | `0.2695` | `0.1773` | `-0.3664` | `76.9232` | `3.8156` |
| `orca` | `0.2199` | `0.0496` | `-0.2327` | `135.4919` | `4.4610` |
| `guarded_ppo` | `0.0071` | `0.0780` | `-0.2213` | `132.2272` | `1.9574` |

## Interpretation

- `guarded_ppo` is much safer than PPO on collisions and near misses.
- It is still not as safe as ORCA on collision rate.
- It almost completely collapses goal-reaching.

The resulting SNQI is slightly better than ORCA because the guard suppresses risky motion and near
misses aggressively, but that does not make it a better planner. The success collapse from `0.2695`
for PPO to `0.0071` for `guarded_ppo` is too severe to justify promotion.

The campaign also emitted a soft SNQI contract warning, which is another sign that the outcome mix
is too skewed to treat this as a stable benchmark-ready profile.

## Comparison To SoNIC-Style Safety-Aware Navigation

What `guarded_ppo` already covers:
- explicit safety intervention semantics
- benchmark-runnable internal safety-aware profile
- clear fallback controller and threshold contract
- practical evidence that the guard changes the policy behavior materially

What it still lacks relative to a SoNIC-style method:
- no learned uncertainty model
- no conformal or calibrated prediction wrapper
- no end-to-end safe-RL training objective
- no source-faithful external provenance to a published safety-aware planner family
- no convincing benchmark tradeoff against the current headline rows

So the repo can now claim partial internal safety-aware support, but not a strong external
safety-aware benchmark representative.

## Verdict

Treat `guarded_ppo` as an experimental documentation-backed safety-aware profile.

It is not baseline-ready and should not be promoted as a headline benchmark row.

## Recommended Claim Boundary

The benchmark stack has:
- internal experimental support for safety-aware execution via guarded PPO
- no strong benchmark-winning safety-aware planner yet
- no justified equivalence claim to SoNIC or similar external safe-RL methods
