# Issue-791 Wave-6 results + camera-ready benchmark blocked on rvo2

Date: 2026-04-20
Supersedes: append to `docs/context/issue_791_promotion_campaign_128k_256k.md`

This note consolidates the Wave-6 follow-up training results (jobs 11725, 11726, 11728,
11730, 11799) and explains why the camera-ready benchmark rerun (job 11798) finished
with `benchmark_success=false` before the new PPO leader was ever evaluated.

## TL;DR

- **Attribution resolved.** A vanilla baseline trained on the eval-aligned scenario
  superset (no reward curriculum, default capacity, job 11799) reaches 0.900 success.
  The Wave-5 leader (11724) reaches 0.929. **~97% of the +0.343 lift over the old
  0.586 plateau comes from distribution alignment alone**; curriculum + large capacity +
  foresight together add a residual +0.029.
- **Foresight is worth +0.029.** Dropping predictive foresight while keeping
  curriculum+largecap (job 11726) lands at 0.900, matching the vanilla baseline. The
  architectural levers stop contributing once foresight is removed.
- **Reward scaling is a dead end.** Job 11725 (reward_strong on top of leader recipe)
  matches 11723 (0.914) but stays below the 11724 leader (0.929). Stronger shaping did
  not compound with curriculum+largecap.
- **Benchmark 11798 halted on ORCA**: `rvo2` is not installed in the venv and
  `socnav_missing_prereq_policy: fallback` does not cover the rvo2 import check;
  `stop_on_failure=true` then halted the run before PPO was evaluated. 3 of 4 planners
  ran to completion (goal, prediction_planner, social_force). **PPO leader was never
  benchmarked.** Fix: `uv sync --extra orca` and resubmit.
- **Multi-seed leader probes (11728, 11730) used 3M-timestep configs, not 10M.** Best
  success at 3M budget for the leader recipe is 0.829 for both other seeds. Seed
  variance at full 10M budget is still unknown.

## Wave-6 results (all 10M unless noted)

| Job | Recipe | Partition | Wall | Best success | Best coll | Best step | Notes |
|----:|---|---|---:|---:|---:|---:|---|
| 11724 | curr + largecap + n_steps=2048 | l40s | 8:04 | **0.929** | 0.071 | 9.96M | **current leader**, SNQI 0.353 |
| 11723 | curr + default + n_steps=4096 | l40s | — | 0.914 | 0.086 | 9.96M | from earlier |
| 11725 | curr + largecap + `reward_strong` | a30 | 13:45 | 0.914 | 0.086 | 9.96M | Stronger shaping ≠ compounding |
| 11726 | curr + largecap **− foresight** | a30 | 10:22 | 0.900 | 0.100 | 9.96M | Foresight lever = +0.029 |
| 11799 | **vanilla baseline**, eval-aligned | l40s | 7:33 | 0.900 | 0.100 | 6.29M | **Control L — attribution** |
| 11728 | curr + largecap, seed 231, **3M** | a30 | 4:48 | 0.829 | 0.171 | 2.10M | Short-budget seed probe |
| 11730 | curr + largecap, seed 1337, **3M** | a30 | 4:47 | 0.829 | 0.171 | 2.62M | Short-budget seed probe |

### Attribution math

- Baseline at old OOD eval (11566/11610 era, pre-alignment): **0.586**
- Vanilla eval-aligned baseline (11799): **0.900** → distribution alignment alone: **+0.314**
- Leader (11724): **0.929** → curriculum + largecap + foresight on top: **+0.029**
- No-foresight variant at curr+largecap (11726): **0.900** → foresight alone: **+0.029**
- Reward-strong variant (11725): **0.914** → reward rescaling: **−0.015 vs leader**

Distribution alignment is overwhelmingly the dominant lever. The architectural /
curriculum stack contributes a small, foresight-gated residual.

### Caveats that must stay attached to these numbers

- Eval-aligned numbers are **in-distribution by construction** — the training and eval
  sets are the same (`configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`). No OOD
  claim is supported yet.
- Seed probes 11728/11730 were 3M budget, not 10M. We do **not** have leader-recipe
  seed variance at full budget. Report 0.929 as a single-seed point estimate until
  additional 10M seeds are run.
- The camera-ready benchmark that would place these numbers on the canonical comparison
  matrix has **not yet run successfully** (see next section).

## Benchmark 11798: why it failed

Config: `configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml`
(l40s, 23:35 wall).

Pipeline:

1. `goal`, `prediction_planner`, `social_force` — all `ok`, 141 episodes each.
2. `orca` — `partial-failure` (0 episodes evaluated). Error:
   `RuntimeError('rvo2 is required for the benchmark-ready ORCA planner. Install via
   uv sync --extra orca or set allow_fallback=True.')`
3. Campaign `stop_on_failure=true` halted; PPO leader and the other experimental
   planners were never evaluated.

Root cause: the `rvo2` Python extra is not installed in the repo venv. The orca
planner in adapter mode checks for `rvo2` at planner construction. The
`socnav_missing_prereq_policy: fallback` key the config already sets only applies to
socnav-related prerequisites, not to the rvo2 import check. So orca reports
`partial-failure` with 141 failed jobs and the `stop_on_failure` gate trips.

Artifacts: `output/benchmarks/issue_791/paper_experiment_matrix_v1_issue791-eval-aligned-leader-11724_20260418_211247/`
— campaign summary and 3 partial planner reports are present; no publication bundle
was emitted (`benchmark_success=false`).

### Fix options

| Option | Cost | Proof quality |
|---|---|---|
| A. `uv sync --extra orca` then resubmit | one sync, 10-15 min | **publication-grade**, full matrix runs |
| B. Set `allow_fallback=True` for orca in the config | zero | orca degrades to goal fallback — must be reported as caveat per fail-closed policy |
| C. Drop orca from this matrix | zero | shrinks baseline set; not comparable to v1 matrix; avoid for paper |

**Recommended: Option A.** Option B violates the fail-closed benchmark contract
(`docs/context/issue_691_benchmark_fallback_policy.md`); Option C breaks comparability
with the reference matrix. If rvo2 cannot be installed on the cluster (native build
issues), fall back to B only with an explicit caveat in the publication text.

## Proposed next steps (priority order)

1. **Unblock the benchmark.** Run `uv sync --extra orca` in the repo venv, re-verify
   `rvo2` imports, and resubmit:

   ```bash
   ISSUE791_BENCHMARK_CONFIG=configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml \
   ISSUE791_BENCHMARK_LABEL=issue791-eval-aligned-leader-11724-publication-retry \
   sbatch SLURM/Auxme/issue_791_benchmark.sl
   ```

2. **Multi-seed re-anchor at full budget.** The 3M seed probes (11728/11730) do not
   close the seed-variance question for the leader recipe. Submit at least two 10M
   seed replicas of `expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity.yaml`
   with seeds 231 and 1337 when slots open. Without this, 0.929 stays a single-seed
   point.

3. ~~**Build a held-out OOD suite.**~~ **Dropped (2026-04-20 maintainer decision).**
   The paper framing is "strong policy on a broad benchmark", not a generalization
   claim, so an OOD split is not required. See
   `docs/context/issue_791_ood_holdout_suite_design.md` for the decision record.
   The strength of the primary claim rests on benchmark-set breadth + seed-variance
   tightness, not on OOD coverage.

4. **Stop the reward-shaping branch.** 11725 shows reward rescaling does not compound
   with curr+largecap. Do not submit Wave-7 reward-shaping variants.

5. **Frame the narrative around benchmark-set performance, not generalization.**
   The paper claim is "a strong PPO policy evaluated on a broad social-navigation
   benchmark matrix". Do **not** use "generalize / transfer / unseen / novel" in
   external text. Internal attribution (distribution alignment dominates over
   curriculum / capacity / foresight) is useful for engineering context but should
   not be framed externally as closing an OOD gap.

6. **Optional: skip Wave-6 control M and stack-the-wins arm N.** Given 11725 did not
   compound and 11799 explains the bulk of the lift, running large-capacity-without-
   curriculum (control M) and curriculum+largecap+n_steps=4096 (arm N) now has lower
   expected information value. Keep the configs in the repo as documented follow-ups
   but deprioritize them behind steps 1–3.

## Promotion recommendation (interim, pending benchmark rerun)

- **Keep 11724 as the promotion candidate** but do not swap
  `configs/baselines/ppo_15m_grid_socnav.yaml` to it yet — wait until the benchmark
  rerun produces a clean publication bundle **and** at least one 10M seed replica lands
  within ±0.02 success of 0.929.
- The leader adapter config
  (`configs/baselines/ppo_issue_791_eval_aligned_large_capacity.yaml`) and the registry
  entry stay in place as the candidate-under-evaluation.

## Benchmark 0.929 → 0.121 gap: horizon mismatch (2026-04-20)

Partial benchmark run 11871 reported PPO leader at success 0.121 / collision 0.567
on the camera-ready matrix, vs success 0.929 on the in-distribution training eval.
Root cause is a **horizon mismatch**, not a policy regression:

- Benchmark: `horizon: 100` steps (10 s at `dt=0.1`) in
  [paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml](../../configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml#L50).
- Training scenarios: `max_episode_steps: 400–600` (40–60 s) across the eval superset
  manifests used by the leader (`ppo_full_maintained_eval_v1.yaml`).

The policy was optimized under 4–6× more time-per-episode than the benchmark allows,
so many benchmark episodes terminate at the horizon before the robot reaches the goal.
This explains low success without a proportional collision spike vs baselines.

**How to apply:** the narrow benchmark-set claim (see
`memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md`) rests on
benchmark numbers, so either (a) re-run the benchmark with a horizon matched to
training episode length, or (b) re-train the promotion candidate with an episode
length matched to `horizon: 100`. Do not report the 0.121 figure as a final result
until the horizon is aligned on one side.

## Wave-7: broad-training candidate (2026-04-20)

Submitted job **11885** (l40s) training the leader recipe on a broader scenario
manifest ([ppo_all_available_training_v1.yaml](../../configs/scenarios/sets/ppo_all_available_training_v1.yaml))
that superset-includes the maintained eval set plus verified-simple, safety-barrier
static, and TEB-topology slices. Config:
[expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity.yaml](../../configs/training/ppo/ablations/expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity.yaml).
Purpose: strengthen the "strong policy on a broad benchmark matrix" narrative by
exposing the same recipe to more scenario diversity during training.

Also re-submitted the benchmark as job **11886** with
`stop_on_failure: false` and both `uv sync --extra orca --extra sacadrl` installed,
so one failed planner (sacadrl TF missing / orca rvo2 missing) no longer halts the
entire campaign.

## References

- Wave-4/5 campaign log and artifact registry:
  `docs/context/issue_791_promotion_campaign_128k_256k.md`
- Benchmark fallback policy:
  `docs/context/issue_691_benchmark_fallback_policy.md`
- GitHub issue draft (for benchmark rerun):
  `docs/context/issue_791_benchmark_rerun_issue_body.md`
- Leader artifact:
  `output/model_cache/ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417/model.zip`
- Partial benchmark artifacts (11798):
  `output/benchmarks/issue_791/paper_experiment_matrix_v1_issue791-eval-aligned-leader-11724_20260418_211247/`
