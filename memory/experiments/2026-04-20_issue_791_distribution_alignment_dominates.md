---
name: Issue 791 - distribution alignment is the dominant lever
description: Wave-5/6 attribution result — eval-aligned training explains ~97% of the 0.586→0.929 PPO lift; curriculum/capacity/foresight add only a small foresight-gated residual.
type: project
---

Across the issue-791 Wave-4/5/6 campaign on the `socnav_struct + grid_socnav` PPO
stack, the lift from the 0.586 OOD plateau (jobs 11566/11610) to the 0.929 Wave-5
leader (job 11724) decomposes as:

- Vanilla baseline on eval-aligned scenarios (job 11799, no curriculum, default
  capacity, 10M): **0.900** success.
- Leader (job 11724, curriculum + large capacity + foresight, 10M): **0.929**.
- No-foresight variant at curriculum+largecap (job 11726): **0.900**.
- Reward-strong variant (job 11725): 0.914 — weaker than leader.

**Why:** the 0.586 plateau was an OOD generalization gap, not a capacity/exploration
ceiling. Training and evaluating on `configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`
(the eval superset itself) lifts all architectures by 0.25–0.50 absolute success.

**How to apply:**

- When analyzing 0.8–0.93 PPO success numbers in this repo, check whether the training
  scenario set equals the eval set before crediting architectural or reward changes.
- For future promotion candidates, start with the vanilla eval-aligned baseline as the
  reference point, not the old 0.586 OOD baseline.
- Do not frame curriculum/capacity/foresight as the primary driver in paper/issue text.
- The 0.929 number is **in-distribution by construction** — it requires a held-out
  OOD suite (see `docs/context/issue_791_ood_holdout_suite_design.md`) before any
  generalization claim.

Leader artifact (candidate, not yet promoted into
`configs/baselines/ppo_15m_grid_socnav.yaml`):
`output/model_cache/ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417/model.zip`

Full write-up: `docs/context/issue_791_wave6_results_and_benchmark_orca_block.md`.

2026-04-29/30 follow-up outcome for issue 857:

- Added a horizon-matched scenario surface at
  `configs/scenarios/sets/ppo_full_maintained_eval_v1_horizon100.yaml` using a new
  manifest-level `scenario_overrides` hook in `robot_sf/training/scenario_loader.py`.
- Added the seed-123 retrain clone
  `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_horizon100.yaml`.
- Added the horizon-400 benchmark reference
  `configs/benchmarks/paper_experiment_matrix_v1_issue_791_horizon400_probe.yaml`.
- Corrected `RobotState` timeout semantics so `max_episode_steps` now expires on the configured
  discrete step budget instead of one step late.
- Job `12178` trained the horizon-100 candidate. Its in-distribution eval reached
  `success_rate=0.6429`, but its camera-ready horizon-100 row in job `12205` fell to
  `success_mean=0.1489` with 120/141 timeouts and 0/141 binary collisions. Do not promote this
  retrain or queue seed replicas.
- Job `12206` reran the issue-791 Wave-5 leader (11724) on the same camera-ready matrix at horizon
  400. The leader recovered to `success_mean=0.8298` across 141 PPO episodes, with 117/141
  successes and 0/141 binary collisions. This is strong evidence that the 11724 leader's
  camera-ready drop is benchmark-horizon-bound, while horizon-100 retraining produced a degenerate
  conservative policy.
- Practical implication: keep 11724 as the canonical PPO leader for now, avoid claiming a
  horizon-100 PPO fix, and compare against horizon 400 as the paper-facing reference.
- Horizon influence sweep jobs completed on 2026-04-30:
  `12212` horizon 200 (`success_mean=0.7518`, `collisions_mean=0.1489`,
  `snqi_mean=-0.2208`), `12213` horizon 300 (`success_mean=0.8440`,
  `collisions_mean=0.1418`, `snqi_mean=-0.1912`), and `12214` horizon 600
  (`success_mean=0.8369`, `collisions_mean=0.1489`, `snqi_mean=-0.1752`). The curve jumps
  sharply from horizon 100 to 200 and is effectively saturated by horizon 300/400; extra budget
  beyond 400 does not materially lift success.

Full issue-857 write-up: `docs/context/issue_857_horizon_alignment_setup.md`.
