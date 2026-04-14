# Issue 791 Promotion Campaign (128k / 256k / 1m)

## Objective

Run a full-length promotion campaign for the three issue-791 quality-gate improvements
(reward curriculum, asymmetric critic, attention head with zero-pedestrian hardening) against
an unchanged baseline. Determine which (if any) are ready for promotion into default training
pipelines.

---

## Completed Results

### Stage 1 (8k) — Stability gates

| Job   | Config                             | Success rate | Collision | Notes                     |
|-------|------------------------------------|-------------|-----------|---------------------------|
| 11434 | Reward curriculum stage1           | 0.0143      | 0.757     | Done                      |
| 11435 | Asymmetric critic stage1           | 0.171       | 0.829     | Done (randomize_seeds=True) |
| 11436 | Attention head stage1              | —           | —         | **NaN crash** (zero-ped bug) |
| 11438 | Attention head stage1 (post-fix)   | 0.0143      | 0.886     | Done                      |

### 32k Follow-up — Stability at longer horizon

| Job   | Config                      | Best success | Final success | Collision | Notes          |
|-------|-----------------------------|-------------|---------------|-----------|----------------|
| 11439 | Reward curriculum 32k       | 0.0429      | 0.0143        | 0.800     | Done           |
| 11440 | Asymmetric critic 32k       | 0.0143      | 0.0143        | 0.814     | Done           |
| 11441 | Attention head 32k          | 0.0143      | 0.0143        | 0.800     | Done           |

### 128k Promotion — Primary campaign (all complete)

| Job   | Config                          | Best success | Best step | Best collision | WandB run ID | Outcome       |
|-------|---------------------------------|-------------|-----------|----------------|--------------|---------------|
| 11462 | Baseline 128k                   | —           | —         | —              | —            | SLURM expired |
| 11467 | **Baseline 128k** (rerun)       | **0.186**   | 131072    | **0.686**      | —            | ✓ Done        |
| 11445 | Reward curriculum 128k          | 0.157       | 32768     | 0.814          | 2pwg82tz     | ✓ Done        |
| 11446 | Asymmetric critic 128k          | 0.171       | 114688    | 0.743          | yrsxbxwg     | ✓ Done        |
| 11447 | Attention head 128k             | 0.171       | 131072    | 0.700          | —            | ✓ Done        |

**128k verdict:** None of the three ablations beat the baseline at 128k. Baseline achieves
best success_rate=0.186 / collision=0.686. All ablations land below on both metrics. This
does not meet the +3–5% promotion threshold.

### 256k Promotion — Extended horizon

| Job   | Config                          | Best success | Best step | Best collision | WandB run ID | Outcome   |
|-------|---------------------------------|-------------|-----------|----------------|--------------|-----------|
| 11468 | Attention head **256k**         | **0.214**   | 229376    | 0.743          | lrxcp863     | ✓ Done    |
| 11474 | **Baseline 256k**               | **0.343**   | 262144    | **0.629**      | bv04bj9h     | ✓ Done    |
| 11475 | **Asymmetric critic 256k**      | **0.257**   | 262144    | **0.743**      | —            | ✓ Done    |

**256k verdict:** Asymmetric critic 256k (11475) peaked at **0.257 / 0.743** — below baseline
(0.343 / 0.629). Not promotion-ready at 256k. Attention head 256k (0.214) also below baseline.
All three ablations fail the +3–5% absolute promotion threshold at both 128k and 256k.

### 1m Env-22 — High-throughput exploration (22 parallel envs)

| Job   | Config                          | Best success | Best step | Best collision | Outcome       |
|-------|---------------------------------|-------------|-----------|----------------|---------------|
| 11470 | Attention head 1m env22         | —           | —         | —              | **OOM** (48G) |
| 11471 | Asymmetric critic 1m env22      | —           | —         | —              | **OOM** (48G) |
| 11469 | **Baseline 1m env22**           | **0.314**   | 917504    | **0.600**      | ✓ Done        |
| 11476 | Attention head 1m env22 (rerun) | —           | ~315k     | —              | **Cancelled** (2026-04-14) |
| 11477 | Asymmetric critic 1m env22      | —           | ~135k     | —              | **Cancelled** (2026-04-14) |

**1m env22 finding:** Baseline 1m env22 (0.314) peaked **below** the 256k baseline (0.343).
Additional compute via 22-env parallelism does not automatically outperform the 256k run for
the baseline. Jobs 11476 and 11477 were cancelled — both showed no improvement signal at 30%
budget (success ≤ 0.014), and the 1m env22 format does not benefit even the baseline.

OOM root cause: 22 subproc workers each load the predictive foresight model on CPU
(~2 GB/worker × 22 = ~44 GB), plus training overhead, exceeds the previous 48 G SLURM
allocation. All three issue-791 SLURM scripts bumped to `--mem=96G` on 2026-04-13.

---

## Key Observations

1. **Baseline wins at 128k and 256k.** No ablation clears the +3–5% absolute threshold
   required for promotion at either budget.

2. **Reward curriculum peaks anomalously early** (best checkpoint at step 32768 of 128k,
   then degrades). The curriculum advances after 4 episodes — too early. The policy
   converges under the simple first-stage reward and then regresses when the full reward
   kicks in. Fix applied in v2: `until_episodes` raised to 100 (~15k warmup steps, 12% of
   128k budget). V2 submitted as job 11510.

3. **Asymmetric critic is not promotion-ready at any budget tested.** 128k: 0.171 (vs
   baseline 0.186). 256k: 0.257 (vs baseline 0.343). The gap widened at 256k — not a
   convergence issue but likely a fundamental signal quality problem.

4. **Attention head scales with compute but does not beat the baseline at any budget tested.**
   128k → 0.171, 256k → 0.214. Baseline 256k → 0.343. The ablation gains absolute
   performance with more steps but the baseline gains more.

5. **1m env22 does not help even the baseline.** Baseline 1m env22 (11469) peaked at 0.314
   at step 917k — below the 256k baseline of 0.343. More compute via parallelism does not
   compound gains beyond the 256k regime. The 1m env22 format is not a useful extension
   for this campaign.

---

## Promotion Decision Criteria

| Outcome           | Criterion                                                            |
|-------------------|----------------------------------------------------------------------|
| Promote           | success_rate +3–5% absolute over matched-budget baseline; collision ≤ baseline |
| Conditional       | Mixed signals or still converging; extend to 256k before deciding    |
| Not ready         | Regression on success_rate vs baseline, or crash/NaN                |

---

## Active Jobs (as of 2026-04-14)

| Job   | Config                                | Budget | Partition | Status   |
|-------|---------------------------------------|--------|-----------|----------|
| 11510 | Reward curriculum v2 128k             | 128k   | a30       | Running  |
| 11512 | **Reward curriculum 10M env22**       | 10M    | a30       | Running  |
| 11514 | **Attention head 10M env22**          | 10M    | l40s      | Pending  |
| 11515 | **Asymmetric critic 10M env22**       | 10M    | l40s      | Pending  |

**Wall-time note:** a30 caps at 36h (~1.5M steps/run), l40s at 72h (~3M steps/run at
estimated throughput). All three 10M jobs will hit the wall and require resubmission with
`resume_model_id` pointing to the previous run's best WandB checkpoint. Eval checkpointed
every 524k steps so no progress is lost on timeout.

---

## Next Decisions (after active jobs complete)

1. **Reward curriculum v2 (11510) verdict:**
   Fix: `until_episodes` raised from 4 to 100 (~15k warmup steps, ~12% of 128k budget).
   → If v2 clears +3–5% over baseline 128k (0.186): promote and extend to 256k.
   → If v2 still shows early-peak regression: raise `until_episodes` to 200–500 or
      switch to a step-based trigger once the training API supports it.
   → If v2 shows no improvement over v1 at 128k: reward curriculum is not viable without
      a deeper redesign.

2. **Attention head and asymmetric critic — closed at current budgets:**
   Both failed at 128k and 256k. No further runs planned unless a fundamentally different
   architecture or hyperparameter hypothesis emerges.

---

## Config Artifact Lineage

```
configs/training/ppo/ablations/
├── expert_ppo_issue_791_baseline_promotion_128k.yaml          ✓ done (11467, best=0.186)
├── expert_ppo_issue_791_baseline_promotion_256k.yaml          ✓ done (11474, best=0.343)
├── expert_ppo_issue_791_baseline_promotion_1m_env22.yaml      ✓ done (11469, best=0.314)
├── expert_ppo_issue_791_reward_curriculum_promotion_128k.yaml ✓ done (11445, best=0.157 — early peak v1)
├── expert_ppo_issue_791_reward_curriculum_promotion_128k_v2.yaml  ← running (11510)
├── expert_ppo_issue_791_reward_curriculum_promotion_256k.yaml    (hold — gate on v2 128k result)
├── expert_ppo_issue_791_reward_curriculum_promotion_10m_env22.yaml  ← running (11512, a30)
├── expert_ppo_issue_791_asymmetric_critic_promotion_128k.yaml ✓ done (11446, best=0.171)
├── expert_ppo_issue_791_asymmetric_critic_promotion_256k.yaml ✓ done (11475, best=0.257) — FAIL
├── expert_ppo_issue_791_asymmetric_critic_promotion_1m_env22.yaml ✗ cancelled (11477)
├── expert_ppo_issue_791_asymmetric_critic_promotion_10m_env22.yaml  ← pending (11515, l40s)
├── expert_ppo_issue_791_attention_head_promotion_128k.yaml    ✓ done (11447, best=0.171)
├── expert_ppo_issue_791_attention_head_promotion_256k.yaml    ✓ done (11468, best=0.214) — FAIL
├── expert_ppo_issue_791_attention_head_promotion_1m_env22.yaml   ✗ cancelled (11476)
└── expert_ppo_issue_791_attention_head_promotion_10m_env22.yaml    ← pending (11514, l40s)
```

## SLURM Script Notes

All three issue-791 SLURM wrappers bumped to `--mem=96G` on 2026-04-13 to support 22-env
runs without OOM (previously 48G; 22 subproc workers × ~2 GB CPU predictive foresight model
per worker exceeded 48G at evaluation time).

Cleanup trap fixed on 2026-04-13: `rsync` is not available on l40s nodes. All three scripts
now fall back to `cp -r` when `rsync` is missing. Job 11504 (l40s test) exposed this —
artifacts were not synced on exit. Jobs submitted after this fix: 11507+.

`PROJECT_ROOT` fix on 2026-04-14: all three Auxme scripts previously set `PROJECT_ROOT` to
`SLURM_SUBMIT_DIR`, which resolves to the directory where `sbatch` is called — not the repo
root. Job 11486 failed because it was submitted from `SLURM/Auxme/`, making config paths like
`configs/training/...` resolve to `SLURM/Auxme/configs/...`. Fixed by resolving PROJECT_ROOT
via `git rev-parse --show-toplevel` from the submit dir. All three scripts updated 2026-04-14.

## Relevant Docs

- [Issue 791 Attention Head Gate](issue_791_attention_head_gate.md)
- [Issue 791 Reward Curriculum Gate](issue_791_reward_curriculum_gate.md)
- [Issue 791 Asymmetric Critic Gate](issue_791_asymmetric_critic_gate.md)
