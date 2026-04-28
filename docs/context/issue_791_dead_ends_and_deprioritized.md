# Issue-791 — dead ends and deprioritized branches

Date: 2026-04-20
Related: `docs/context/issue_791_wave6_results_and_benchmark_orca_block.md`

This note lists training directions that Wave-5/6 results disprove or render low-value,
so future agents do not waste SLURM budget repeating them.

## Dead ends (do not re-run)

### 1. Reward rescaling on top of curriculum+largecap

- Evidence: job 11725 (`..._eval_aligned_reward_strong.yaml`, 10M a30, 13:45 wall)
  → best success 0.914 / collision 0.086 at step 9.96M.
- Leader (11724) with unscaled weights reaches 0.929.
- **Stronger reward shaping does not compound with curriculum+largecap.** It delivers a
  −0.015 success delta and adds nothing to the safety column.
- Do not submit further reward-scaling sweeps on top of the leader recipe.

### 2. Warm-restart exploration boost at plateau (ent_coef=0.02, clip=0.15, target_kl=0.03)

- Evidence: job 11661 (previous campaign) early-stopped at every minibatch and
  degraded 0.586 → 0.471.
- Cause: larger target_kl with the current trust-region tuning triggers PPO KL guard.
- **Do not use aggressive exploration warm restarts as a plateau-escape mechanism.**

## Deprioritized (not disproven, but expected low information value)

### 3. Wave-6 Control M — large capacity × no curriculum (eval-aligned)

- Config: `configs/training/ppo/ablations/expert_ppo_issue_791_baseline_promotion_10m_env22_eval_aligned_large_capacity.yaml`.
- Rationale for deprioritization: 11799 (vanilla baseline, no curriculum, default
  capacity) reaches 0.900; 11724 (leader, curriculum + largecap) reaches 0.929. The
  residual +0.029 is already known to be foresight-gated (11726). Running control M
  would only partition that +0.029 between curriculum and capacity — useful for
  internal attribution but not decision-critical for promotion.
- **Keep the config; do not submit unless a secondary ablation is explicitly requested.**

### 4. Wave-6 stack-the-wins arm N — curriculum + largecap + n_steps=4096

- Config: `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_nsteps4096.yaml`.
- Rationale for deprioritization: 11723 (n_steps=4096, default cap) landed at 0.914 vs
  11724 (n_steps=2048, largecap) at 0.929. n_steps=4096 is not individually winning at
  default cap. Stacking both levers is unlikely to exceed 0.929 by a margin large
  enough to change the publication story.
- **Keep the config; reconsider only if 10M seed replicas (11869/11870) show the
  leader is highly seed-variable.**

## Keep doing

### 5. Eval-aligned training on full scenario superset

- Evidence: 11799 alone delivers +0.314 over the 0.586 OOD plateau.
- Distribution alignment is the dominant lever — any future experiment in this track
  should keep `scenario_config: ppo_full_maintained_eval_v1.yaml` unless explicitly
  probing a held-out split.

### 6. Predictive foresight

- Evidence: 11726 (no-foresight) lands at 0.900 vs 11724 (foresight) at 0.929.
- Foresight contributes +0.029 success on top of curriculum+largecap.
- **Do not disable predictive foresight in production training runs.**

### 7. Multi-seed anchoring at full budget

- Evidence: 3M-budget seed probes (11728, 11730) landed at 0.829 — well below 10M
  leader. 10M replicas (11869, 11870) are now running to close this gap.
- Continue the practice of 10M budget replicas for headline claims.
