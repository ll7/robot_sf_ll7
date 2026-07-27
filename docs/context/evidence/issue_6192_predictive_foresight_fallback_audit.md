<!-- AI-GENERATED (robot_sf#6192, 2026-07-23) — NEEDS-REVIEW -->
# Historical Evidence Audit: Silent Predictive-Foresight Fallback Contamination

**Issue**: [#6192](https://github.com/ll7/robot_sf_ll7/issues/6192)
**Date**: 2026-07-23
**Status**: Complete within the durable tracked-evidence scope
**Evidence tier**: diagnostic-only (tracked config and provenance analysis; no
tracked runtime outcome evidence)

## Scope

This audit classifies every claim-bearing benchmark campaign whose baseline config
enables `predictive_foresight_enabled: true`. The three allowed outcome classes are
defined in the issue body:

1. **confirmed-unaffected** — foresight disabled for that campaign, or verified model
   asset present at run time.
2. **confirmed-fallback** — retained logs contain the load/fallback warning.
3. **unverifiable** — foresight enabled but no retained model-load evidence.

**Rule**: Unverifiable is NOT retroactive contamination. Absence of evidence is not
evidence of fallback. Claims that materially depend on an unverifiable campaign may
need a rerun; that is a per-claim judgement.

---

## Model Asset Status

The predictive foresight model referenced by all active configs is
`predictive_proxy_selected_v2_full` (`model/registry.yaml:616-647`):

| Property | Value |
|---|---|
| `local_only` | `false` |
| `commit` | `cef93136b92ddca9b0c4436bc44049412461a2fd` |
| W&B run | `ll7/robot_sf/u40parjb` |
| GitHub release | `artifact/models-2026-05-registry-v1` |
| SHA256 | `a28aed6d6ad7e1ebf597277ade1cf908efa6da038d0a9fcfdf80c7c31d8d1be1` |
| Published | 2026-05-24T05:32:03+00:00 |

The model has a durable public release pointer since 2026-05-24. Before that date
it was available via W&B provenance (`ll7/robot_sf/u40parjb`) and local training
output paths documented in the registry.

---

## Durable Evidence Search

The audit searched repository-tracked campaign configs, release manifests, model
registry metadata, and context evidence for an explicit runtime record of either
successful model loading or the warning
`"Falling back to constant-velocity predictive planner behavior"`.

**Finding**: The tracked evidence contains no campaign receipt, log, or sidecar
recording either runtime outcome for a campaign that uses
`predictive_foresight_enabled: true`. Those campaigns are therefore unverifiable
from durable tracked evidence. Ephemeral host-local files and remote SLURM, W&B,
or CI logs were not used, and this audit does not assert that such records do not
exist.

---

## Baseline Configs with `predictive_foresight_enabled: true`

Six baselines under `configs/baselines/` enable the predictive foresight feature:

| # | Baseline Config | Status | Model ID | Predictive Model |
|---|---|---|---|---|
| B1 | `configs/baselines/ppo_15m_grid_socnav.yaml` | **Active, promoted** | `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417` | `predictive_proxy_selected_v2_full` |
| B2 | `configs/baselines/ppo_issue_791_eval_aligned_large_capacity.yaml` | Active (historical sibling) | Same as B1 | Same |
| B3 | `configs/baselines/ppo_issue_791_eval_aligned_large_capacity_cpu.yaml` | Active (CPU variant) | Same as B1 | Same |
| B4 | `configs/baselines/ppo_issue_791_eval_aligned_large_capacity_portable.yaml` | Active (portable variant) | Same as B1 | Same; uses `device: auto` |
| B5 | `configs/baselines/ppo_issue_791_horizon100_12178.yaml` | **Retired**; artifact missing (`model_id: retired_*_local_only`) | N/A (retired) | Same |
| B6 | `configs/baselines/ppo_issue_856_all_scenarios_12223.yaml` | **Retire-pending**; artifact missing | N/A (retired) | Same |

Additionally, two policy-search candidates enable it via `env_overrides`:
- `configs/policy_search/candidates/orca_residual_guarded_ppo_v0.yaml`
- `configs/policy_search/candidates/orca_residual_guarded_ppo_progress_v1.yaml`

---

## Claim-Bearing Campaign Classification

### Paper-Facing Campaigns

All campaigns listed as `paper_facing: true` in their config metadata. Each is
classified by the baseline it references.

#### Confirmed-Unaffected (foresight not enabled, or not on the affected planner path)

The following paper-facing campaigns reference only baselines that do NOT set
`predictive_foresight_enabled: true`. They are **clean; no action needed**.

| Campaign Config | Baseline(s) Used |
|---|---|
| `paper_experiment_matrix_v1_gap_prediction_compare.yaml` | gap_prediction planner |
| `paper_experiment_matrix_v1_gap_prediction_compare_schema_filtered.yaml` | gap_prediction planner |
| `paper_experiment_matrix_v1_mppi_social_compare.yaml` | mppi_social planner |
| `paper_experiment_matrix_v1_mppi_social_compare_schema_filtered.yaml` | mppi_social planner |
| `paper_experiment_matrix_v1_prediction_planner_v2_compare.yaml` | prediction_planner (separate model, not the PPO foresight) |
| `paper_experiment_matrix_v1_predictive_mppi_compare.yaml` | predictive_mppi planner |
| `paper_experiment_matrix_v1_risk_dwa_compare.yaml` | risk_dwa planner |
| `paper_experiment_matrix_v1_hybrid_portfolio_compare.yaml` | hybrid portfolio planners |
| `paper_experiment_matrix_v1_stream_gap_compare.yaml` | stream_gap planner |
| `paper_experiment_matrix_v1_prediction_planner_probabilistic_compare.yaml` | prediction_planner (probabilistic) |
| `paper_experiment_matrix_v1_ppo_v10_carry_forward.yaml` | `ppo_v10_carry_forward_27dbe5xu` (no foresight) |
| `paper_experiment_matrix_v1_social_navigation_pyenvs_hsfm_new_guo_only.yaml` | Social Force variants |
| `paper_experiment_matrix_v1_social_navigation_pyenvs_orca_only.yaml` | ORCA |
| `paper_experiment_matrix_v1_social_navigation_pyenvs_socialforce_only.yaml` | Social Force |

#### Unverifiable (foresight enabled, no retained model-load evidence)

The following paper-facing campaigns use baselines with
`predictive_foresight_enabled: true`. No tracked runtime outcome evidence confirms
whether the model loaded or fell back.

**Their reported PPO-arm results cannot be proven to have run with genuine foresight
features or silent constant-velocity fallback.**

| Campaign Config | Baseline | Paper-Facing | Notes |
|---|---|---|---|
| `paper_experiment_matrix_v1.yaml` | B1 (`ppo_15m_grid_socnav.yaml`) | Yes | **Core paper matrix** |
| `paper_experiment_matrix_all_planners_v1.yaml` | B1 | Yes | |
| `paper_experiment_matrix_7planners_v1.yaml` | B1 | Yes | |
| `paper_experiment_matrix_v1_extended_seeds_s10.yaml` | B1 | Yes | |
| `paper_experiment_matrix_v1_extended_seeds_s5.yaml` | B1 | Yes | |
| `paper_experiment_matrix_v1_release_smoke.yaml` | B1 | Yes | |
| `paper_experiment_matrix_v1_guarded_ppo_compare.yaml` | B1 | Yes | |
| `paper_experiment_matrix_v1_best_ppo_compare.yaml` | B1 | Yes | |
| `paper_experiment_matrix_v1_issue_821_extended.yaml` | B1 | Yes | |
| `paper_experiment_matrix_v1_guarded_ppo_tuning_compare.yaml` | B1 | Yes | |
| `paper_experiment_matrix_v1_guarded_ppo_tuning_compare_workers1.yaml` | B1 | Yes | |
| `paper_experiment_matrix_v2_h600_s30_extended.yaml` | B1 | Yes | |
| `paper_experiment_matrix_v2_h600_s30_extended_post1.yaml` | B1 | Yes | |
| `paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml` | B2 (large_capacity) | Yes | |
| `paper_experiment_matrix_v1_scenario_horizons_h500.yaml` | B2 | Yes | |
| `paper_experiment_matrix_v1_issue_791_horizon400_probe.yaml` | B2 | Yes | |
| `paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml` | B3 (cpu) | Yes | |
| `paper_experiment_matrix_v1_issue_857_horizon100.yaml` | B5 (retired) | Yes | **Retired baseline; artifact missing; not runnable** |

**Total paper-facing unverifiable campaigns: 18** (17 using active baselines + 1
using a retired baseline whose PPO checkpoint is also missing).

#### Not Paper-Facing but Claim-Adjacent

| Campaign Config | Baseline | Notes |
|---|---|---|
| `paper_experiment_matrix_v1_h600_extended_roster.yaml` | B2 | Not paper-facing; diagnostic |
| `paper_experiment_matrix_v1_issue_791_horizon600_probe.yaml` | B2 | Not paper-facing |
| `paper_experiment_matrix_v1_issue_791_horizon300_probe.yaml` | B2 | Not paper-facing |
| `paper_experiment_matrix_v1_issue_791_horizon200_probe.yaml` | B2 | Not paper-facing |
| `paper_experiment_matrix_v1_h600_hybrid_vs_orca_s30.yaml` | B2 | Not paper-facing |
| `paper_experiment_matrix_v1_issue_856_all_scenarios_compare.yaml` | B6 (retired) | Not paper-facing; baseline retired |

These are classified **unverifiable** for the same reason — no retained logs.

### Release Manifests

All 5 release manifests chain to `ppo_15m_grid_socnav.yaml` via their campaign
config and are therefore **unverifiable** by inheritance:

| Release | Campaign |
|---|---|
| `paper_experiment_matrix_v1_release_v0_1.yaml` | `paper_experiment_matrix_v1.yaml` |
| `paper_experiment_matrix_v1_release_smoke_v0_1.yaml` | `paper_experiment_matrix_v1_release_smoke.yaml` |
| `paper_experiment_matrix_7planners_v1_release_v0_0_2_scoped.yaml` | `paper_experiment_matrix_7planners_v1.yaml` |
| `paper_experiment_matrix_all_planners_v1_release_v0_0_2.yaml` | `paper_experiment_matrix_all_planners_v1.yaml` |
| `paper_experiment_matrix_v2_h600_s30_release_v0_0_3.yaml` | `paper_experiment_matrix_v2_h600_s30_extended.yaml` |

---

## Summary

| Outcome Class | Count | Campaigns |
|---|---|---|
| **Confirmed-unaffected** | 14 | Paper-facing campaigns that use baselines without `predictive_foresight_enabled: true` |
| **Confirmed-fallback** | 0 | No tracked runtime evidence records the fallback warning |
| **Unverifiable** | 24 (+ 5 releases) | All 18 paper-facing + 6 non-paper campaigns using baselines with `predictive_foresight_enabled: true`; no tracked runtime outcome evidence exists |

---

## Mitigating Factors

1. **The predictive model has a durable registry entry with W&B and GitHub release
   provenance.** Unlike the two retired PPO checkpoint entries (B5, B6), the
   `predictive_proxy_selected_v2_full` model is `local_only: false` and has a
   published release asset with a known SHA256. A run on a machine with network
   access and the registry-aware download path should resolve it successfully.

2. **The silent fallback requires a specific condition to trigger:** the
   `PredictiveForesightEncoder` model-load step must fail. This can happen when the
   model file is absent and network downloads fail (transient blip, no W&B access,
   no GitHub release access, wrong device, OOM). The fact that many of these
   campaigns ran on SLURM GPU nodes with W&B access and the model locally available
   from training reduces — but does not eliminate — the fallback probability.

3. **Fallback is not the only possible silent degradation.** Even when the model
   loads successfully, runtime divergence between the PPO training-time foresight
   features and evaluation-time foresight outputs could change behavior without
   errors. This audit does not address that separate concern.

---

## Claims That May Need Reruns

Per the issue rules, unverifiable campaigns are not automatically contaminated, but
claims materially depending on them may need reruns. The following claims are
affected:

- **PPO baseline performance in paper-facing tables**: all paper matrix PPO results
  are unverifiable. The canonical `ppo_15m_grid_socnav.yaml` baseline evaluation
  numbers (success_rate=0.929, collision_rate=0.071, SNQI=0.353) were measured
  during training (70 eval episodes, in-distribution). The scoped paper-matrix
  results from the best-learning report (success 0.2569, collision 0.0903) are
  also unverifiable.

- **PPO vs ORCA comparisons** in paper-facing tables that depend on the PPO arm.

- **Release 0.0.2 and 0.0.3 aggregate results** that include PPO arms.

- **Individual issue reports** citing paper-experiment-matrix PPO results for
  comparative claims.

**Recommendation**: Before paper submission or public release, validate one
representative campaign with the foresight model pre-seeded and `allow_fallback`
explicitly set to `false`, to establish a clean-reference benchmark for PPO
results. This is already being tracked by [#6189](https://github.com/ll7/robot_sf_ll7/issues/6189)
and [#6190](https://github.com/ll7/robot_sf_ll7/issues/6190).

---

## Caveats

- **No tracked runtime outcome evidence**: The "unverifiable" classification is
  driven by the durable evidence boundary, not evidence of fallback. The actual
  fallback rate is unknown and could be zero.
- **Durable-source boundary**: This report audits repository-tracked evidence only.
  Host-local runtime files and remote SLURM, W&B, or CI logs were not available,
  and no claim is made that they do not exist.
- **Config analysis only**: the classification relies on static config inspection.
  Runtime behavior may differ if configs were overridden or the code path changed
  between the campaign date and the current HEAD.
