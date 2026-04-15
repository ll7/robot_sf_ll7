# Issue 791 Promotion Campaign (128k / 256k / 1m / 10m GPU Foresight)

## Objective

Run a full-length promotion campaign for the three issue-791 quality-gate improvements
(reward curriculum, asymmetric critic, attention head with zero-pedestrian hardening) against
an unchanged baseline. Determine which (if any) are ready for promotion into default training
pipelines.

2026-04-15 update: the campaign reopened after identifying that long-horizon PPO throughput was
being dominated by CPU-side predictive-foresight inference. The predictive-model device path now
correctly moves the loaded predictor to the requested runtime device, which enabled CUDA-backed
foresight smoke validation and restarted the long-run comparison under materially higher
throughput.

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

### 2026-04-15 GPU foresight validation and long-run restart

| Job   | Config / purpose                              | Outcome | Key evidence |
|-------|-----------------------------------------------|---------|--------------|
| 11544 | Asymmetric critic stage1 smoke, foresight on CUDA | Failed | CPU/CUDA tensor mismatch inside predictive model forward |
| 11560 | Asymmetric critic stage1 smoke, foresight on CUDA (post-fix) | Completed | `Using cuda device`, `fps=73`, no device mismatch |
| 11561 | Reward curriculum 10m submission attempt on a30 | Completed but invalid | Wrapper defaulted to stage1 config; not a promotion run |
| 11562 | Attention-head config via asymmetric wrapper, 10m env22 on a30 | Running | Long-run GPU-foresight comparison arm |
| 11566 | Reward curriculum 10m env22 on l40s | Running | Corrected resubmission with explicit 10m config |
| 11572 | Baseline 1m env22 on a30 | Failed | Transient SLURM allocation handshake failure (`Zero Bytes were transmitted or received`) |
| 11573 | Baseline 1m env22 on a30 (retry) | Failed | Same transient allocation handshake failure as 11572 |
| 11577 | Baseline 1m env22 on l40s | Pending | Re-routed baseline retry after repeated a30 handshake failures |
| 11578 | Baseline 3m env22 on a30 | Pending (dependency) | Pre-committed extension queued with `afterok:11577` |

GPU foresight bug root cause:

- `PredictionPlannerAdapter._build_model()` loaded the predictive checkpoint with
   `map_location=self._device`, but never moved the instantiated `PredictiveTrajectoryModel`
   module to that device.
- Checkpoint tensors were therefore copied into CPU module parameters, while inference inputs were
   moved to CUDA, producing the runtime failure seen in job 11544.
- Fix applied in `robot_sf/planner/socnav.py`: call `model.to(self._device)` after
   `load_predictive_checkpoint(...)` and before `model.eval()`.

GPU foresight smoke evidence:

- Job 11560 (`expert_ppo_issue_791_asymmetric_critic_stage1_gpu_foresight_smoke`) completed in
   2m10s with `fps=73` at 2 envs.
- This replaced the earlier CPU-foresight smoke behavior of roughly 8 to 11 fps and removed the
   CUDA tensor mismatch.
- The 10m env22 campaign was restarted only after this smoke path completed successfully.

---

## Key Observations

1. **Baseline wins at 128k and 256k.** No ablation clears the +3–5% absolute threshold
   required for promotion at either budget.

2. **Reward curriculum peaks anomalously early** (best checkpoint at step 32768 of 128k,
   then degrades). The curriculum advances after 4 episodes — too early. The policy
   converges under the simple first-stage reward and then regresses when the full reward
   kicks in. Fix applied in v2: `until_episodes` raised to 100 (~15k warmup steps, 12% of
   128k budget). V2 submitted as job 11510.

3. **Asymmetric critic underperforms the matched baseline at 128k and 256k, but the device-side
   foresight fix reopens the long-horizon question.** Earlier 1m/10m env22 conclusions were
   confounded by CPU-side foresight cost and wall-time inefficiency. The architecture remains
   below the baseline at completed medium budgets, but the restarted GPU-backed 10m run is now the
   correct test of whether the long-horizon regime changes that ordering.

4. **Attention head scales with compute but does not beat the baseline at completed budgets.**
   128k → 0.171, 256k → 0.214. Baseline 256k → 0.343. The long-horizon GPU-backed rerun is still
   useful as a final sanity check, but the current burden of proof remains on the ablation to
   close a large baseline gap.

5. **Old CPU-foresight env22 throughput conclusions are superseded.** The previous 1m/10m env22
   runs used CPU predictive foresight and underutilized the GPU nodes. The 1m baseline result
   (11469) still matters historically, but future env22 comparisons should treat the GPU-foresight
   restart as the canonical high-throughput path.

6. **Reward curriculum is the current leader among the restarted long runs.** As of 2026-04-15,
   job 11566 on l40s reached about 1.62M steps at roughly 389 to 391 fps with success_rate about
   0.386, collision_rate about 0.586, eval_episode_return about -2.59, and SNQI about -1.31.

7. **The restarted asymmetric/attention-head arm is viable but lagging.** As of 2026-04-15,
   job 11562 on a30 reached about 1.31M steps at roughly 155 to 186 fps with recent
   success_rate about 0.214, collision_rate about 0.771, eval_episode_return about -15.2, and
   SNQI about -2.07. It is learning, but currently trails reward curriculum on both throughput and
   quality.

---

## Promotion Decision Criteria

| Outcome           | Criterion                                                            |
|-------------------|----------------------------------------------------------------------|
| Promote           | success_rate +3–5% absolute over matched-budget baseline; collision ≤ baseline |
| Conditional       | Mixed signals or still converging; extend to 256k before deciding    |
| Not ready         | Regression on success_rate vs baseline, or crash/NaN                |

---

## Active Jobs (as of 2026-04-15)

| Job   | Config                                | Budget | Partition | Status   |
|-------|---------------------------------------|--------|-----------|----------|
| 11562 | **Attention-head config via asymmetric wrapper, 10M env22** | 10M | a30 | Running |
| 11566 | **Reward curriculum 10M env22**       | 10M    | l40s      | Running  |
| 11577 | **Baseline 1M env22**                 | 1M     | l40s      | Pending  |
| 11578 | **Baseline 3M env22**                 | 3M     | a30       | Pending (dependency) |

**Current throughput note:** with GPU foresight enabled, the restarted long runs are no longer in
the single-digit fps regime. Observed live throughput is about 155 to 186 fps for job 11562 on
a30 and about 389 to 391 fps for job 11566 on l40s. These values materially change the expected
wall-time behavior relative to the old CPU-foresight estimates.

---

## Validation / Proof Commands

Commands used to justify the current update:

```bash
sacct -j 11544,11560,11561,11562,11566,11572,11573,11577,11578 --format=JobID,JobName%45,Partition,State,Elapsed,ExitCode -X -n
squeue -u "$USER" -o "%.18i %.9P %.45j %.10T %.10M %.10l %R" | grep 'robot-sf-issue791'
tail -n 120 output/slurm/11560-issue791-asymmetric-critic.out
tail -n 60 output/slurm/11561-issue791-reward-curriculum.out
grep -E 'total_timesteps|success_rate|collision_rate|snqi|eval_episode_return|value_loss|fps ' output/slurm/11562-issue791-asymmetric-critic.out | tail -n 24
grep -E 'total_timesteps|success_rate|collision_rate|snqi|eval_episode_return|value_loss|fps ' output/slurm/11566-issue791-reward-curriculum.out | tail -n 24
```

## Next Decisions / Follow-ups

1. **Confirm job 11577 startup provenance.** The baseline job uses the reward-curriculum wrapper
   with an explicit `ISSUE791_TRAIN_CONFIG` override. Check the startup summary once it begins to
   confirm that it resolves to `ppo_expert_issue_791_baseline_promotion_1m_env22` and not the
   wrapper's stage1 default.

2. **Decide whether attention head needs its own dedicated 10m rerun.** Job 11562 currently runs
   the attention-head promotion config through the asymmetric wrapper/job naming path. If this arm
   becomes benchmark-relevant, submit a cleanly named wrapper or explicit attention-head script so
   accounting and artifact lineage stay unambiguous.

3. **Re-evaluate promotion thresholds using the GPU-backed long runs, not the old CPU-foresight
   env22 runs.** The reward-curriculum versus asymmetric comparison should be revisited after the
   restarted runs reach a materially larger shared budget.

4. **Consider making the 10m wrapper defaults safer.** Jobs 11561 and earlier submissions showed
   that wrapper defaults can silently fall back to stage1 configs when the override is omitted.
   A follow-up hardening change should either default the promotion wrappers to the promotion YAMLs
   or fail fast when a long-run submission lacks an explicit config override.

5. **Decide whether the baseline should be extended beyond 1m only if 11577 is competitive.** If
   the baseline remains clearly below the leading reward-curriculum run even with GPU foresight,
   further baseline scaling is low value. If it stays close, extend it before making a final
   promotion call.

---

## Config Artifact Lineage

```
configs/training/ppo/ablations/
├── expert_ppo_issue_791_baseline_promotion_128k.yaml          ✓ done (11467, best=0.186)
├── expert_ppo_issue_791_baseline_promotion_256k.yaml          ✓ done (11474, best=0.343)
├── expert_ppo_issue_791_baseline_promotion_1m_env22.yaml      ✓ done (11469, best=0.314, CPU foresight historical) / ← pending rerun (11572, CUDA foresight)
├── expert_ppo_issue_791_reward_curriculum_promotion_128k.yaml ✓ done (11445, best=0.157 — early peak v1)
├── expert_ppo_issue_791_reward_curriculum_promotion_128k_v2.yaml  ← running (11510)
├── expert_ppo_issue_791_reward_curriculum_promotion_256k.yaml    (hold — gate on v2 128k result)
├── expert_ppo_issue_791_reward_curriculum_promotion_10m_env22.yaml  ← running (11566, l40s, CUDA foresight)
├── expert_ppo_issue_791_asymmetric_critic_promotion_128k.yaml ✓ done (11446, best=0.171)
├── expert_ppo_issue_791_asymmetric_critic_promotion_256k.yaml ✓ done (11475, best=0.257) — FAIL
├── expert_ppo_issue_791_asymmetric_critic_promotion_1m_env22.yaml ✗ cancelled (11477)
├── expert_ppo_issue_791_asymmetric_critic_promotion_10m_env22.yaml  historical pending/cancelled path (11515); current long-run comparison uses the attention-head config instead
├── expert_ppo_issue_791_attention_head_promotion_128k.yaml    ✓ done (11447, best=0.171)
├── expert_ppo_issue_791_attention_head_promotion_256k.yaml    ✓ done (11468, best=0.214) — FAIL
├── expert_ppo_issue_791_attention_head_promotion_1m_env22.yaml   ✗ cancelled (11476)
└── expert_ppo_issue_791_attention_head_promotion_10m_env22.yaml    ← running via wrapper override (11562, a30, CUDA foresight)
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

GPU predictive-model device fix on 2026-04-15: the predictive foresight smoke run failed first
because the loaded `PredictiveTrajectoryModel` stayed on CPU while inference tensors were moved to
CUDA. `PredictionPlannerAdapter._build_model()` now calls `model.to(self._device)` after loading
the checkpoint. Smoke proof: job 11544 failed with a CPU/CUDA mismatch; job 11560 succeeded with
`Using cuda device` and `fps=73`.

Issue-791 wrapper hardening on 2026-04-15: `ISSUE791_TRAIN_CONFIG` is now required by
`issue_791_reward_curriculum.sl`, `issue_791_asymmetric_critic.sl`, and
`issue_791_attention_head.sl`. This prevents accidental stage1 fallback for promotion/long runs.

Auxme reliability helpers added on 2026-04-15:

- `scripts/dev/auxme_partition_status.sh` for live `a30`/`l40s` free-GPU and queue-pressure checks,
- `scripts/dev/sbatch_auxme_issue791.sh` for explicit-config submission with partition recommendation
   plus max-time-safe routing through `scripts/dev/sbatch_use_max_time.sh`.
- Reliability helper follow-up: ensure partition/qos are forwarded to `sbatch` as explicit args;
   routing hints used only for wall-time discovery are insufficient and can silently fall back to
   the script-local partition defaults.

## Relevant Docs

- [Issue 791 Attention Head Gate](issue_791_attention_head_gate.md)
- [Issue 791 Reward Curriculum Gate](issue_791_reward_curriculum_gate.md)
- [Issue 791 Asymmetric Critic Gate](issue_791_asymmetric_critic_gate.md)
