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
