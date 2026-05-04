# Issue #856 Implementation Verification

Date: 2026-05-02
Branch: `856-ppo-all-scenarios`
Base: `origin/main`
Verification skill: `.claude/skills/implementation-verification/SKILL.md`

This note records the pre-PR implementation verification for the issue-856 broad-training
campaign. It complements the campaign log at
[docs/context/issue_856_ppo_all_scenarios_full_budget.md](issue_856_ppo_all_scenarios_full_budget.md)
and is kept here so future agents can audit how the PR-ready state was reached.

## Branch Scope

7 files, +433 / −39. Touches only evidence/config/documentation surfaces for the completed
seed-123 broad-training control — no Python source or SLURM wrapper changes.

Inventory:

- 1 new baseline adapter:
  [configs/baselines/ppo_issue_856_all_scenarios_12223.yaml](../../configs/baselines/ppo_issue_856_all_scenarios_12223.yaml).
- 1 new benchmark matrix:
  [configs/benchmarks/paper_experiment_matrix_v1_issue_856_all_scenarios_compare.yaml](../../configs/benchmarks/paper_experiment_matrix_v1_issue_856_all_scenarios_compare.yaml).
- 2 updated context notes:
  [docs/context/issue_856_ppo_all_scenarios_full_budget.md](issue_856_ppo_all_scenarios_full_budget.md)
  and this verification record.
- 1 experiment-memory update:
  [memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md](../../memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md).
- `CHANGELOG.md` and [docs/README.md](../README.md) updates for discoverability and verdict
  preservation.

## Evidence Matrix

| Claim | Evidence surface | Validation | Result |
|---|---|---|---|
| Seed-123 (job 12223) trained 10M, best at step 9,961,472, success 0.900 / collision 0.100 / SNQI 0.226 | `output/slurm/issue791-reward-curriculum-job-12223/.../*_best.summary.json` | Numerical match with the context note | Pass |
| Camera-ready benchmark `12226` ran 7/7 successful, 987 episodes | `output/benchmarks/issue_856/.../reports/campaign_summary.json` | `runs=7`, all `status=ok`, 7×141=987 episodes, no warnings | Pass |
| Baseline adapter wires the 12223 checkpoint | [configs/baselines/ppo_issue_856_all_scenarios_12223.yaml](../../configs/baselines/ppo_issue_856_all_scenarios_12223.yaml) | Path exists locally; provenance captures WandB `ll7/robot_sf/ateif3c8` | Pass (worktree-local; not durable) |
| Benchmark matrix swaps only the PPO row vs eval-aligned compare | [paper_experiment_matrix_v1_issue_856_all_scenarios_compare.yaml](../../configs/benchmarks/paper_experiment_matrix_v1_issue_856_all_scenarios_compare.yaml) | YAML loads; only the `ppo` row uses `algo_config: configs/baselines/ppo_issue_856_all_scenarios_12223.yaml` | Pass |
| Camera-ready PPO row improves vs reference 12122 | `reports/campaign_table_experimental.md`: PPO row `success_mean=0.2199`, `collisions_mean=0.0922`, `snqi_mean=−0.3305` vs note-stated 12122 reference `0.2553 / 0.0851 / −0.2906` | Read campaign table | Numerical evidence available; broad-training arm is **worse** on all three metrics |

## Initial Gaps Found

1. **Branch was stale vs `origin/main`** — merge-base at `5b9bbc82`, ~30 commits behind including
   `robot_sf/adversarial/{__init__,bundle,certification,samplers,search}.py` and matching test
   updates. `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` failed
   [tests/adversarial/test_adversarial_search.py:243](../../tests/adversarial/test_adversarial_search.py#L243)
   on the stale branch (expects `not_available`, observed `failed`). Same test passes when those
   files are taken from `origin/main`. PR-blocker per
   [AGENTS.md](../../AGENTS.md).

2. **Camera-ready comparison conclusion missing** — the broad-training PPO row was recorded but
   not interpreted against 12122. Numerical deltas: success −0.0354, collisions +0.0071, SNQI
   −0.0399. The note's validation step 4 ("publication grade / parity / worse") was left undone,
   and the linked
   [memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md](../../memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md)
   was not updated.

3. **Replica seed-band write-back undefined** — the gating decision was recorded but the note
   didn't pre-commit a destination for the eventual seed-band result.

4. **Trained checkpoint and publication bundle are worktree-local only** — acknowledged in the
  note, but the durable-artifact policy under `AGENTS.md` was not yet resolved.

## Fixes Applied

- Merged `origin/main` into `856-ppo-all-scenarios`. Re-ran
  `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` and confirmed the adversarial test now
  passes alongside the rest of the parallel suite.
- Added a "Comparison vs job 12122" subsection to
  [docs/context/issue_856_ppo_all_scenarios_full_budget.md](issue_856_ppo_all_scenarios_full_budget.md)
  recording the three numerical deltas and the explicit verdict that broad-training underperforms
  on the camera-ready matrix at fixed budget.
- Updated
  [memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md](../../memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md)
  with the full-budget broad-training data point so the alignment-vs-diversity claim now
  references the 10M-budget control rather than only the 4.19M Wave-7 result.
- Pre-committed the seed-band write-back contract in the campaign note: replicas 12257 / 12258
  outcomes flow into the same context note plus the distribution-alignment memory entry.
- Recorded an explicit durable-artifact decision in the campaign note: the 12223 checkpoint is
  only promoted to W&B/Zenodo if the seed-band result becomes the publication-grade row;
  otherwise it stays as a local control whose provenance is the WandB run reference.

## How To Reproduce This Verification

1. `git fetch origin main`.
2. Inspect `git diff --stat origin/main...HEAD` and `git log origin/main..HEAD`.
3. Read [issue 856](https://github.com/ll7/robot_sf_ll7/issues/856),
   [docs/context/issue_856_ppo_all_scenarios_full_budget.md](issue_856_ppo_all_scenarios_full_budget.md),
   and the new configs.
4. `yaml.safe_load` the new baseline and benchmark configs to confirm validity.
5. Read `output/benchmarks/issue_856/.../reports/campaign_summary.json` and
   `campaign_table_experimental.md` for camera-ready evidence.
6. `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`.

## Related

- [docs/context/issue_856_ppo_all_scenarios_full_budget.md](issue_856_ppo_all_scenarios_full_budget.md)
- [memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md](../../memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md)
- [memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md](../../memory/decisions/2026-04-20_issue_791_narrow_benchmark_claim.md)
- [AGENTS.md](../../AGENTS.md)
