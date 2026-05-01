# Issue #856 - PPO all-scenarios full-budget campaign

Date: 2026-04-29

This note tracks the issue-856 follow-up that reruns the issue-791 leader recipe on the broader
training manifest at the full 10M budget.

## Goal

Measure whether the broad-training control closes the alignment-vs-diversity ambiguity left by the
eval-aligned issue-791 leader.

- Training manifest under test:
  `configs/scenarios/sets/ppo_all_available_training_v1.yaml`
- Leader reference:
  job 11724, success `0.929` on the eval-aligned in-distribution surface
- Camera-ready PPO reference:
  job 12122, `success_mean=0.2553`, `collisions_mean=0.0851`, `snqi_mean=-0.2906`,
  `time_to_goal_norm_mean=0.9274`

## Repo Changes For This Campaign

- Reused the existing seed-123 config:
  `configs/training/ppo/ablations/expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity.yaml`
- Added explicit seed-replica configs so the gated follow-up submissions stay reproducible:
  - `configs/training/ppo/ablations/expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity_seed231.yaml`
  - `configs/training/ppo/ablations/expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity_seed1337.yaml`

All three configs keep the same PPO hyperparameters, `grid_socnav` extractor, reward curriculum,
predictive foresight settings, `num_envs: 22`, and `randomize_seeds: true`.

## Submission Evidence

Seed-123 was submitted on 2026-04-29 through the maintained Auxme helper:

```bash
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity.yaml \
  --wandb-policy require \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

Observed helper output before submit:

- `a30`: `free_gpu=4`, `pending=0`, `slots_left=1`
- `l40s`: `free_gpu=0`, `pending=2`, `slots_left=1`
- Selected routing: `partition=a30`, `qos=a30-gpu`

Accepted job:

- Slurm job ID: `12172`
- Wrapper: `SLURM/Auxme/issue_791_reward_curriculum.sl`
- Effective wall time: `1-12:00:00`
- WandB policy: `require`
- Expected synced artifact root:
  `output/slurm/issue791-reward-curriculum-job-12172/`
- Expected stdout log:
  `output/slurm/12172-issue791-reward-curriculum.out`

## Replica Gate

Do not queue the seed-231 and seed-1337 configs until job 12172 completes and the best checkpoint
is evaluated.

Gate for proceeding with replicas:

- If the broad-training seed-123 result lands more than `0.05` below the eval-aligned leader on
  the in-distribution surface, stop at one seed and report the result as a negative control.
- Otherwise submit the two replica configs above and compare their camera-ready benchmark band
  against the 11724/12122 references.

## Validation Path After Job 12172 Lands

1. Run the best-checkpoint in-distribution evaluation with
   `configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`.
2. Add a benchmark adapter config under `configs/baselines/` for the trained artifact.
3. Rerun
   `configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml`
   with the PPO row swapped to the broad-training artifact.
4. Compare the PPO row against job 12122 and decide whether the broad-training arm is publication
   grade, parity-only, or clearly worse.
5. Update
   `memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md`
   once the outcome is known.

## Metadata Limitation

Issue #856 is the tracked GitHub issue for this work and the current branch is
`856-ppo-all-scenarios`. A Project #5 status update was attempted from this environment, but the
installed `gh` token lacks the `read:project` scope needed to read or mutate `projectV2` fields.
The issue state can still be tracked through the branch, this note, and the queued Slurm job.

## Related Surfaces

- GitHub issue: `https://github.com/ll7/robot_sf_ll7/issues/856`
- `docs/context/issue_791_wave6_results_and_benchmark_orca_block.md`
- `docs/context/issue_791_best_policy_verdict_2026_04_21.md`
- `memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md`
