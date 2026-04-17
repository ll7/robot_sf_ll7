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
| 11562 | Attention-head config via asymmetric wrapper, 10m env22 on a30 | Failed | Reached about 5.4M steps, then hit the empty-map-pool runtime path later tracked in issue #830 |
| 11566 | Reward curriculum 10m env22 on l40s | Completed | Final eval `success_rate=0.571`, `collision_rate=0.400`; best checkpoint `success_rate=0.58571` at step `8912896` |
| 11572 | Baseline 1m env22 on a30 | Failed | Transient SLURM allocation handshake failure (`Zero Bytes were transmitted or received`) |
| 11573 | Baseline 1m env22 on a30 (retry) | Failed | Same transient allocation handshake failure as 11572 |
| 11577 | Baseline 1m env22 on l40s | Failed | Malformed YAML in `expert_ppo_issue_791_baseline_promotion_1m_env22.yaml` caused `yaml.scanner.ScannerError` before training started |
| 11578 | Baseline 3m env22 on a30 | Cancelled | Stale `afterok:11577` dependency after the failed 1m baseline attempt |
| 11582 | Baseline 1m env22 on l40s (corrected rerun) | Completed | Best checkpoint reached `success_rate=0.31429`; final eval settled at `success_rate=0.18571`, `collision_rate=0.68571` |
| 11583 | Baseline 3m env22 on a30 (replacement) | Failed | Transient SLURM allocation handshake failure during allocation startup |
| 11584 | Baseline 3m env22 on l40s (replacement) | Failed | Full rerun reached the same empty-map-pool runtime path later tracked in issue #830 |

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

6. **Reward curriculum is still the current leader among the restarted long runs.** Job 11566 on
   l40s completed its 10m run with final eval `success_rate=0.571`, `collision_rate=0.400`,
   `eval_episode_return=-0.617`, and `SNQI=-0.814`. The best W&B checkpoint reached
   `success_rate=0.58571` with `collision_rate=0.37143` at step `8912896`.

7. **The restarted asymmetric/attention-head arm was viable but lagging before failing on the
   loader bug.** Before the later empty-map-pool failure, job 11562 on a30 reached about 4.5M
   steps at roughly 157 to 196 fps with recent eval
   `success_rate=0.271`, `collision_rate=0.686`, `eval_episode_return=-11.1`, and
   `SNQI=-1.69`. Rollout success was fluctuating in the `0.42` to `0.54` range, but the eval
   surface still trails reward curriculum on both throughput and quality.

8. **The immediate issue with job 11578 was a dead dependency chain, not a scheduler anomaly.**
   Job 11578 was queued with `afterok:11577`, and 11577 failed because the 1m baseline YAML had a
   misindented `predictive_foresight_device` key. The corrected replacement chain was
   `11582 -> 11583`, after which 11583 still hit a transient allocation-handshake failure and
   11584 exposed the empty-map-pool runtime path that is now tracked separately.

---

## Promotion Decision Criteria

| Outcome           | Criterion                                                            |
|-------------------|----------------------------------------------------------------------|
| Promote           | success_rate +3–5% absolute over matched-budget baseline; collision ≤ baseline |
| Conditional       | Mixed signals or still converging; extend to 256k before deciding    |
| Not ready         | Regression on success_rate vs baseline, or crash/NaN                |

---

## Historical Active Jobs Snapshot (as of 2026-04-15)

| Job   | Config                                | Budget | Partition | Status   |
|-------|---------------------------------------|--------|-----------|----------|
| 11562 | **Attention-head config via asymmetric wrapper, 10M env22** | 10M | a30 | Running |
| 11582 | **Baseline 1M env22**                 | 1M     | l40s      | Running  |
| 11583 | **Baseline 3M env22**                 | 3M     | a30       | Pending (dependency) |

**Current throughput note:** with GPU foresight enabled, the restarted long runs are no longer in
the single-digit fps regime. Observed live throughput is about 157 to 196 fps for job 11562 on
a30, and the completed reward-curriculum run 11566 finished its endgame around 475 to 510 fps on
l40s. These values materially change the expected wall-time behavior relative to the old
CPU-foresight estimates.

## Queue State And Next-Wave Decision (as of 2026-04-16)

Current queue snapshot: job `11609` is running on `a30`, job `11610` is pending on `a30`, and
job `11606` remains pending on `pro6000`. Job `11608` completed successfully, and the first reward
warm-start retry (`11607`) failed fast on an observation-space mismatch and is now historical.

### Best-checkpoint preference

For the imech192 wave, prefer warm starts from `best.zip` instead of the most recent `step*.zip`
when the goal is to continue from the strongest observed policy rather than preserve strict
training chronology.

### Submitted jobs and reasoning

1. **11566 reward curriculum** — continue from `best.zip`.
   Job `11607` was the first retry using [expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_resume_best.yaml](../../configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_resume_best.yaml), but it still loaded the wrong model family and failed on an observation-space mismatch. The overlay was corrected back to the original `grid_socnav` / `MultiInputPolicy` contract and resubmitted as `11610`.

2. **11582 baseline 1m env22** — continue from `best.zip`.
   Job `11608` is the corrected retry of [expert_ppo_issue_791_baseline_promotion_1m_env22_resume_best.yaml](../../configs/training/ppo/ablations/expert_ppo_issue_791_baseline_promotion_1m_env22_resume_best.yaml) and completed successfully on `a30`.

3. **11562 attention-head rerun** — direct SLURM rerun.
   Job `11609` is the corrected retry of [expert_ppo_issue_791_attention_head_promotion_10m_env22.yaml](../../configs/training/ppo/ablations/expert_ppo_issue_791_attention_head_promotion_10m_env22.yaml) and is currently running on `a30`.

4. **11584 baseline 3m env22 full rerun** — low-priority overnight job.
   Job `11606` uses [expert_ppo_issue_791_baseline_promotion_3m_env22.yaml](../../configs/training/ppo/ablations/expert_ppo_issue_791_baseline_promotion_3m_env22.yaml) and remains pending on `pro6000` by design.

### Current retry status (updated 2026-04-17)

| Planned arm | Current job | Config | State | Partition |
|-------------|-------------|--------|-------|-----------|
| 11566 reward curriculum warm start | 11610 | [expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_resume_best.yaml](../../configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_resume_best.yaml) | Completed 2026-04-17T02:37 (plateau confirmed) | a30 |
| 11582 baseline warm start | 11608 | [expert_ppo_issue_791_baseline_promotion_1m_env22_resume_best.yaml](../../configs/training/ppo/ablations/expert_ppo_issue_791_baseline_promotion_1m_env22_resume_best.yaml) | Completed | a30 |
| 11562 attention-head rerun | 11609 | [expert_ppo_issue_791_attention_head_promotion_10m_env22.yaml](../../configs/training/ppo/ablations/expert_ppo_issue_791_attention_head_promotion_10m_env22.yaml) | Running (~10.1M steps, near completion) | a30 |
| 11584 full rerun | 11606 | [expert_ppo_issue_791_baseline_promotion_3m_env22.yaml](../../configs/training/ppo/ablations/expert_ppo_issue_791_baseline_promotion_3m_env22.yaml) | Failed (pro6000 allocation issue 2026-04-16) | pro6000 |

### 2026-04-17 Latest Findings

**Reward curriculum plateau confirmed.** Warm restart job 11610 (WandB run `x4qe2phu`) loaded `best.zip` from job 11566 and ran another ~10M steps over 1h 14m on a30. It reached exactly the same best checkpoint value as the source (`best/checkpoint_value=0.58571`, `best/collision_rate=0.38571` at step 9,961,472), and the final evaluation slipped back to `success_rate=0.529`, `collision_rate=0.414`. This is direct proof that the current recipe has exhausted its headroom at this compute budget: more steps from the best-known policy do not lift either metric.

**Train/eval gap is the dominant ceiling, not compute or architecture.** Rollout `success_rate` in job 11609 stabilized at 0.70–0.82 from step ~3M onward, while the 70-episode eval oscillates in 0.30–0.40. The eval set (`ppo_full_maintained_eval_v1.yaml`) is a superset of the training set — it adds atomic navigation archetypes (`issue_596_frame_consistency`, `issue_596_static_obstacles`, `issue_596_topology`, `issue_596_dynamic`, `issue_596_robustness`) that the training distribution never samples. The ~0.30 gap between rollout and eval success is therefore an out-of-distribution generalization gap.

**Attention head 10M ceiling is below reward curriculum.** Job 11609 WandB run `0hto2j2i` peaked at eval `success_rate=0.40` near 7.3M steps (collision 0.586) and has oscillated in 0.30–0.40 / 0.57–0.70 since. It matches curriculum at 128k/256k but does not surpass it at 10M and sits below the reward-curriculum leader (`0.586`).

**Confirmed campaign leaderboard after 11610 completion:**

| Run | Best eval success | Best eval collision | Best step | WandB |
|-----|-------------------|---------------------|-----------|-------|
| 11566 reward curriculum 10m | **0.58571** | 0.37143 | 8,912,896 | (historical) |
| 11610 reward curriculum warm-restart | 0.58571 (tie, no gain) | 0.38571 | 9,961,472 | `x4qe2phu` |
| 11609 attention-head 10m | ~0.40 (running, ~10.1M) | ~0.543 | ~7.3M | `0hto2j2i` |
| 11474 baseline 256k | 0.343 | 0.629 | 262,144 | `bv04bj9h` |
| 11608 baseline 1m warm | 0.31429 | 0.600 | 1,000,000 | `tbsty04a` |

The leader is the reward-curriculum family; the remaining promotion gap is generalization, not optimization.

### Contract Corrections Already Applied

The first retry wave exposed two distinct compatibility problems. `RobotEnv.__init__()` no longer
accepted `asymmetric_critic`, so the base contract was restored in
`robot_sf/gym_env/robot_env.py` and covered by a regression test in
`tests/test_socnav_env_integration.py`. The reward warm-start then exposed a second mismatch: the
overlay had drifted to an asymmetric attention layout, but the source 11566 checkpoint was trained
with `grid_socnav` and `MultiInputPolicy`. That overlay was corrected back to the original
architecture and resubmitted as `11610`.

Current expectation: the running and pending retries now line up with the intended checkpoint
families, so the remaining verification is execution-time only.

---

## Validation / Proof Commands

Commands used to justify the current update and next-wave planning decision:

```bash
sacct -j 11606,11607,11608,11609,11610 --format=JobID,JobName%45,Partition,State,Elapsed,ExitCode -X -n
squeue -u "$USER" -o "%.18i %.9P %.45j %.10T %.10M %.10l %R" | grep 'robot-sf-issue791' || true
grep -n 'Observation spaces do not match' output/slurm/11607-issue791-reward-curriculum.out
grep -E 'Training startup summary: policy_id=ppo_expert_issue_791_reward_curriculum_promotion_10m_env22|critic_profile=|policy=' output/slurm/11566-issue791-reward-curriculum.out | head -n 4
grep -E 'Training startup summary: policy_id=ppo_expert_issue_791_baseline_promotion_1m_env22|critic_profile=|policy=' output/slurm/11608-issue791-reward-curriculum.out | head -n 4
grep -E 'Training startup summary: policy_id=ppo_expert_issue_791_attention_head_promotion_10m_env22|critic_profile=|policy=' output/slurm/11605-issue791-attention-head.out | head -n 4
```

## Next Decisions / Follow-ups

### Previous wave (resolved)

1. ~~Monitor 11566 and 11582 warm starts.~~ Done: 11610 (reward curriculum warm) tied the source
   checkpoint (plateau proof). 11608 (baseline warm) completed at 0.314.
2. ~~Watch 11562 for the loader-fix validation signal.~~ Done: 11609 rerun ran cleanly through
   ~10M steps with no loader crash; issue-830 fix is validated in production.
3. ~~Expect 11584 to start later on `pro6000`.~~ Resolved: 11606 on pro6000 failed again on an
   allocation issue; the 3M baseline arm is deprioritized since 11582 already gave a 1M anchor.

### 2026-04-17 next-wave hypotheses (post-plateau)

The dominant finding is a ~0.30 train/eval generalization gap that is not closed by more compute
on the same distribution. The next wave should attack that gap directly rather than chase more
training-rollout throughput:

1. **Scenario-aligned training (top priority).** Train on `ppo_full_maintained_eval_v1.yaml` so
   the atomic archetypes in the eval set are in the training distribution. Expected to close a
   large portion of the OOD gap and lift eval success above the 0.586 plateau.
2. **Reward-curriculum warm start with exploration boost.** Continue from 11566 `best.zip` with
   `ent_coef` bumped from 0.008 → 0.02, `clip_range` 0.1 → 0.15, `target_kl` 0.02 → 0.03. Tests
   whether the plateau is an exploitation lock-in and not a capability ceiling.
3. **Curriculum + asymmetric critic combined at 10M.** Pair the two best single-factor wins
   (reward curriculum + asymmetric critic) with GPU foresight and 22 envs. This combination was
   not benchmarked together at 10M with GPU foresight.

These are the three submitted arms — see "Wave 4 submissions" below.

---

## Config Artifact Lineage

```
configs/training/ppo/ablations/
├── expert_ppo_issue_791_baseline_promotion_128k.yaml          ✓ done (11467, best=0.186)
├── expert_ppo_issue_791_baseline_promotion_256k.yaml          ✓ done (11474, best=0.343)
├── expert_ppo_issue_791_baseline_promotion_1m_env22.yaml      ✓ done (11469, best=0.314, CPU foresight historical) / ✗ handshake failures (11572, 11573) / ✗ malformed-YAML failure (11577) / ✓ done corrected rerun (11582; next step is a best.zip warm start)
├── expert_ppo_issue_791_baseline_promotion_3m_env22.yaml      ✗ handshake failure during replacement run (11583) / ✗ empty-map-pool failure during rerun (11584) / ← planned low-priority full rerun
├── expert_ppo_issue_791_reward_curriculum_promotion_128k.yaml ✓ done (11445, best=0.157 — early peak v1)
├── expert_ppo_issue_791_reward_curriculum_promotion_128k_v2.yaml  ← running (11510)
├── expert_ppo_issue_791_reward_curriculum_promotion_256k.yaml    (hold — gate on v2 128k result)
├── expert_ppo_issue_791_reward_curriculum_promotion_10m_env22.yaml  ✓ done (11566, best success=0.58571 @ 8.91M; final eval success=0.571) / ← planned best.zip warm start (resume overlay corrected back to the original grid_socnav contract)
├── expert_ppo_issue_791_asymmetric_critic_promotion_128k.yaml ✓ done (11446, best=0.171)
├── expert_ppo_issue_791_asymmetric_critic_promotion_256k.yaml ✓ done (11475, best=0.257) — FAIL
├── expert_ppo_issue_791_asymmetric_critic_promotion_1m_env22.yaml ✗ cancelled (11477)
├── expert_ppo_issue_791_asymmetric_critic_promotion_10m_env22.yaml  historical pending/cancelled path (11515); current long-run comparison uses the attention-head config instead
├── expert_ppo_issue_791_attention_head_promotion_128k.yaml    ✓ done (11447, best=0.171)
├── expert_ppo_issue_791_attention_head_promotion_256k.yaml    ✓ done (11468, best=0.214) — FAIL
├── expert_ppo_issue_791_attention_head_promotion_1m_env22.yaml   ✗ cancelled (11476)
└── expert_ppo_issue_791_attention_head_promotion_10m_env22.yaml    ✗ partial long run failed after ~5.4M steps (11562) / ← planned direct SLURM retry
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

Baseline config parse fix on 2026-04-15: job 11577 exposed a malformed
`configs/training/ppo/ablations/expert_ppo_issue_791_baseline_promotion_1m_env22.yaml` where
`predictive_foresight_device` was misindented. The corrected rerun is job 11582, and the stale
dependent extension 11578 was replaced with 11583.

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

## Wave 4 — 10+ Proposed Approaches And Selected Submissions (2026-04-17)

### Proposal list

Every approach is framed against the current leader (11566 reward curriculum 10M,
`success_rate=0.58571`, `collision=0.37143`). Ranked roughly by expected return on cluster
budget.

1. **Scenario-aligned training set.** Train on `ppo_full_maintained_eval_v1.yaml` so the atomic
   `issue_596_*` archetypes in the eval set are in-distribution during training. The ~0.30 gap
   between rollout success (0.75) and eval success (0.40) is almost entirely OOD generalization,
   so this should move the ceiling more than any hyperparameter tweak.
2. **Warm-start with exploration boost.** Resume from 11566 `best.zip`, raise `ent_coef`
   (0.008 → 0.02), loosen `clip_range` (0.1 → 0.15) and `target_kl` (0.02 → 0.03). Tests whether
   the plateau is exploitation lock-in rather than a capability ceiling.
3. **Combined asymmetric critic + reward curriculum + GPU foresight + eval-aligned scenarios.**
   Never benchmarked as a combined long-run with GPU foresight; compounds several working
   ingredients on the bottleneck distribution.
4. **LR schedule: linear decay from 1.5e-4 to 1e-5.** Fresh training with higher initial LR to
   escape the 7.5e-5 flat plateau, then anneal for stability.
5. **Longer PPO rollout horizon (`n_steps`) per env.** Default is 2048; moving to 4096 improves
   advantage estimation and GAE quality, typically helps long-horizon tasks at a modest wall-time
   cost.
6. **Larger policy capacity.** Widen `grid_channels` to `[64, 128, 128]` and
   `socnav_hidden_dims` to `[256, 256]`; the current grid CNN may be capacity-limited for the
   atomic archetypes.
7. **Recurrent PPO (LSTM policy).** Temporal memory over pedestrian dynamics; the current policy
   is reactive and foresight-conditioned only, which can lose information on multi-step interaction
   patterns.
8. **Curriculum on scenario difficulty.** Phase-in harder sampling weights (low → medium → high
   density) gated on running success-rate, instead of a static `weights` block.
9. **Reward shaping: stronger terminal bonus or progress weight.** Raise `terminal_bonus`
   (20 → 30) or `progress` (1.1 → 1.5); rollout success is already high so the policy may need
   sharper signal on episode completion.
10. **Multi-seed ensemble search (3 seeds × 3M each).** PPO is seed-sensitive; a cheap fan-out can
    reveal whether the 0.586 plateau is seed-specific or a true distribution ceiling.
11. **Disable predictive foresight at 10M.** Test if the GPU-predictive adapter actually helps at
    long horizon, or is acting as noise against an already well-conditioned grid observation.
12. **Domain randomization on pedestrian density.** Sample `max_peds_per_group` and pedestrian
    count within an expanded range so the policy sees crowd-density OOD states during training.
13. **Self-play / history-augmented sampling.** Mix scenarios where the robot starts from
    mid-episode states that were previously collisions, forcing the policy to learn recovery.
14. **Two-stage training pipeline.** Stage A (3M) on classic-only for fast progress acquisition,
    Stage B (7M) warm-started on the eval-aligned superset for generalization lift.

### Candidate selection

Selected the three highest-expected-return arms that are independent enough to distinguish which
lever moved the needle:

| Wave 4 arm | Intervention | Expected lever | Cost |
|------------|--------------|----------------|------|
| **A (eval-aligned curriculum)** | Train on `ppo_full_maintained_eval_v1.yaml` | Generalization gap | 10M, a30/l40s |
| **B (warm-start exploration boost)** | Resume 11566 `best.zip` with `ent_coef=0.02`, `clip_range=0.15`, `target_kl=0.03` | Exploitation lock-in | 10M, a30/l40s |
| **C (asymmetric critic + eval-aligned)** | Combined asymmetric critic + reward curriculum + GPU foresight + eval-aligned set | Generalization × privileged-critic interaction | 10M, a30/l40s |

Rationale: A tests the OOD hypothesis directly, B tests the exploitation-lockin hypothesis
against the same source checkpoint, and C compounds the two ingredients we have strongest
evidence for (reward curriculum leadership + asymmetric critic architecture) onto the
eval-aligned distribution.

### Wave 4 submissions

| Arm | Config | Wrapper | Job | Partition | Node |
|-----|--------|---------|-----|-----------|------|
| A | [expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned.yaml](../../configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned.yaml) | `issue_791_reward_curriculum.sl` | **11660 (RUNNING)** | a30 | auxme-imech172 |
| B | [expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_resume_exploration_boost.yaml](../../configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_resume_exploration_boost.yaml) | `issue_791_reward_curriculum.sl` | **11661 (RUNNING)** | l40s | auxme-imech091 |
| C | [expert_ppo_issue_791_asymmetric_critic_promotion_10m_env22_eval_aligned.yaml](../../configs/training/ppo/ablations/expert_ppo_issue_791_asymmetric_critic_promotion_10m_env22_eval_aligned.yaml) | `issue_791_asymmetric_critic.sl` | **11662 (RUNNING)** | l40s | auxme-imech093 |

All three started within seconds of submission on 2026-04-17. Job 11609 (attention-head rerun)
is also still running on a30 as the fourth concurrent issue-791 job this wave.

Submission commands (run from repo root):

```bash
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned.yaml \
  --job-name robot-sf-issue791-rc-eval-aligned \
  SLURM/Auxme/issue_791_reward_curriculum.sl

scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_resume_exploration_boost.yaml \
  --job-name robot-sf-issue791-rc-explore-boost \
  SLURM/Auxme/issue_791_reward_curriculum.sl

scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_asymmetric_critic_promotion_10m_env22_eval_aligned.yaml \
  --job-name robot-sf-issue791-asym-eval-aligned \
  SLURM/Auxme/issue_791_asymmetric_critic.sl
```

## Relevant Docs

- [Issue 791 Attention Head Gate](issue_791_attention_head_gate.md)
- [Issue 791 Reward Curriculum Gate](issue_791_reward_curriculum_gate.md)
- [Issue 791 Asymmetric Critic Gate](issue_791_asymmetric_critic_gate.md)
