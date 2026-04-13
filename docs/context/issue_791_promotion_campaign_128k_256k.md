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
| **11474** | **Baseline 256k**           | **0.257**   | 196608    | 0.686          | —            | **Running** (~221k/256k, 2026-04-13) |
| **11475** | **Asymmetric critic 256k**  | —           | —         | —              | —            | **Pending** (a30 QoS queue) |
| **11507** | **Asymmetric critic 256k**  | —           | —         | —              | —            | **Running** (l40s, 2026-04-13) |

**Revised finding:** baseline 256k is currently peaking at **0.257** (step 196k, collision
0.686), which **exceeds** attention head 256k (0.214). If this holds to completion, attention
head does NOT beat the 256k baseline and is not promotion-ready at this budget. Full verdict
pending 11474 completion. (Prior note about attention head beating the 128k baseline still
holds — the 128k baseline was 0.186 — but the matched-budget comparison is the deciding one.)

### 1m Env-22 — High-throughput exploration (22 parallel envs)

| Job   | Config                          | Reached      | Status          |
|-------|---------------------------------|--------------|-----------------|
| 11469 | Baseline 1m env22               | ~450k steps  | Running         |
| 11470 | Attention head 1m env22         | ~first eval  | **OOM** (48G)   |
| 11471 | Asymmetric critic 1m env22      | ~first eval  | **OOM** (48G)   |
| **11476** | **Attention head 1m env22** | —            | **Running** (96G fix) |
| **11477** | **Asymmetric critic 1m env22** | —          | **Running** (96G fix) |

OOM root cause: 22 subproc workers each load the predictive foresight model on CPU
(~2 GB/worker × 22 = ~44 GB), plus training overhead, exceeds the previous 48 G SLURM
allocation. All three issue-791 SLURM scripts bumped to `--mem=96G` on 2026-04-13.

---

## Key Observations

1. **Baseline wins at 128k.** No ablation clears the +3–5% absolute threshold required
   for promotion at 128k.

2. **Reward curriculum peaks anomalously early** (best checkpoint at step 32768 of 128k,
   then degrades). The curriculum advances after 4 episodes — too early. The policy
   converges under the simple first-stage reward and then regresses when the full reward
   kicks in. Needs a longer `until_episodes` or a softer blending schedule before it is
   worth extending to 256k.

3. **Asymmetric critic was still climbing at 128k end** (best at step 114688 / 128k = 90%
   through). Very likely under-converged. 256k run (11475) will clarify.

4. **Attention head scales with compute but likely does not beat the 256k baseline.**
   128k → 0.171, 256k → 0.214. The 256k result exceeds the *128k* baseline (0.186), but
   the *256k* baseline (11474) is currently peaking at 0.257 — above the attention head.
   Verdict pending 11474 completion.

5. **1m env22 is viable after the OOM fix.** 48G was the bottleneck; 96G should clear it.
   The baseline 1m env22 (11469) is still running and will provide the long-horizon reference.

---

## Promotion Decision Criteria

| Outcome           | Criterion                                                            |
|-------------------|----------------------------------------------------------------------|
| Promote           | success_rate +3–5% absolute over matched-budget baseline; collision ≤ baseline |
| Conditional       | Mixed signals or still converging; extend to 256k before deciding    |
| Not ready         | Regression on success_rate vs baseline, or crash/NaN                |

---

## Active Jobs (as of 2026-04-13)

| Job   | Config                          | Budget | Partition | Priority |
|-------|---------------------------------|--------|-----------|----------|
| 11469 | Baseline 1m env22               | 1M     | a30       | Reference (~630k/1M) |
| 11474 | Baseline 256k                   | 256k   | a30       | **Urgent** — nearly done (~221k/256k); verdict gate for attention head |
| 11475 | Asymmetric critic 256k          | 256k   | a30       | Pending (QoS queue) |
| 11507 | Asymmetric critic 256k          | 256k   | **l40s**  | High — parallel run, free slot used 2026-04-13 |
| 11476 | Attention head 1m env22         | 1M     | a30       | Medium — OOM rerun after mem fix |
| 11477 | Asymmetric critic 1m env22      | 1M     | a30       | Medium — OOM rerun after mem fix |

---

## Next Decisions (after active jobs complete)

1. **Attention head 256k verdict (after 11474 done):**
   Current data suggests baseline 256k (0.257 peak) > attention head 256k (0.214).
   → If confirmed: attention head is NOT promotion-ready at 256k budget. Consider 1m run.
   → If 11474 drops back and final best < 0.214: re-open promotion case.

2. **If asymmetric critic 256k (11475 or 11507) > baseline 256k:**
   → Asymmetric critic is promotion-ready. Consider stacking it with attention head.

3. **Reward curriculum v2 (config prepared, not yet submitted):**
   Config: `expert_ppo_issue_791_reward_curriculum_promotion_128k_v2.yaml`
   Fix: `until_episodes` raised from 4 to 100 (~15k warmup steps with 2 envs).
   → Submit once an a30 or l40s slot is free after the current 256k wave completes.
   → If v2 still shows early-peak regression, raise to 200–500 or switch to step-based.

4. **1m env22 results (11476, 11477, 11469):**
   → If 1m env22 attention head beats 256k attention head by a meaningful margin,
   scale the baseline to 1m env22 as well for a final head-to-head.

---

## Config Artifact Lineage

```
configs/training/ppo/ablations/
├── expert_ppo_issue_791_baseline_promotion_128k.yaml         ✓ done (11467)
├── expert_ppo_issue_791_baseline_promotion_256k.yaml         ← running (11474)
├── expert_ppo_issue_791_baseline_promotion_1m_env22.yaml     ← running (11469)
├── expert_ppo_issue_791_reward_curriculum_promotion_128k.yaml ✓ done (11445)
├── expert_ppo_issue_791_reward_curriculum_promotion_128k_v2.yaml  (ready — until_episodes=100 fix)
├── expert_ppo_issue_791_reward_curriculum_promotion_256k.yaml  (hold — run v2 128k gate first)
├── expert_ppo_issue_791_asymmetric_critic_promotion_128k.yaml ✓ done (11446)
├── expert_ppo_issue_791_asymmetric_critic_promotion_256k.yaml ← running (11475)
├── expert_ppo_issue_791_asymmetric_critic_promotion_1m_env22.yaml ← running (11477)
├── expert_ppo_issue_791_attention_head_promotion_128k.yaml   ✓ done (11447)
├── expert_ppo_issue_791_attention_head_promotion_256k.yaml   ✓ done (11468, best=0.214)
└── expert_ppo_issue_791_attention_head_promotion_1m_env22.yaml ← running (11476)
```

## SLURM Script Notes

All three issue-791 SLURM wrappers bumped to `--mem=96G` on 2026-04-13 to support 22-env
runs without OOM (previously 48G; 22 subproc workers × ~2 GB CPU predictive foresight model
per worker exceeded 48G at evaluation time).

Cleanup trap fixed on 2026-04-13: `rsync` is not available on l40s nodes. All three scripts
now fall back to `cp -r` when `rsync` is missing. Job 11504 (l40s test) exposed this —
artifacts were not synced on exit. Jobs submitted after this fix: 11507+.

## Relevant Docs

- [Issue 791 Attention Head Gate](issue_791_attention_head_gate.md)
- [Issue 791 Reward Curriculum Gate](issue_791_reward_curriculum_gate.md)
- [Issue 791 Asymmetric Critic Gate](issue_791_asymmetric_critic_gate.md)
