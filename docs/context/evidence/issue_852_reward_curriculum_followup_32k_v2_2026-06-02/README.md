# Issue 852 32k reward-curriculum follow-up probe

This directory records the lightweight, reviewable provenance for the
`goal-slurm-experiment` 32k PPO training probe submitted on 2026-06-02.

Status: completed compute-node training probe; not campaign or paper-grade evidence.

## Submission

- Issue: #852
- Job: `12704` (`gse-852-32k`)
- Partition/QoS: `a30` / `a30-gpu`
- Node: `auxme-imech172`
- Worktree: `../robot_sf_ll7.worktrees/gse-852-32k-v2`
- Branch: `gse-852-32k-v2`
- Commit: `4cd4329568ff`
- Config:
  `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_followup_32k_v2.yaml`
- Launcher: `SLURM/Auxme/issue_791_reward_curriculum.sl`
- W&B run: <https://wandb.ai/ll7/robot_sf/runs/8ukuycz6>
- Local Slurm log:
  `output/slurm/12704-issue791-reward-curriculum.out`
- Synced local artifact root:
  `output/slurm/issue791-reward-curriculum-job-12704/`

Local `output/` paths are ignored and non-durable. The W&B run is the durable external pointer
for the model/artifact upload observed at completion.

## Result Summary

- SLURM accounting: `COMPLETED`, exit code `0:0`
- Runtime: 56m20s
- Total timesteps: 32768
- Seeds: `[123]`
- Eval schedule: `[8192, 16384, 24576, 32768]`
- Train throughput: about 10.65 env steps/sec
- Best checkpoint: step 16384 by success rate
- Best success rate: 0.1714
- Best collision rate: 0.8286
- Best SNQI: -2.1017
- Final success rate: 0.0714
- Final collision rate: 0.8286
- Final SNQI: -2.2305

Interpretation: the run completed and produced artifacts, but the 32k probe remains weak. The
best checkpoint does not meet convergence, collision remains high, and final success regressed
below the best checkpoint. Treat this as negative/diagnostic probe evidence rather than a promotion
candidate.

## Warnings

The log includes startup warnings that should be considered during interpretation:

- `uni_campus_big.svg` obstacle path `obstacle_13` has a self-intersection warning.
- Scenario switching reported observation-space bound mismatches and reused initial bounds for
  compatibility.

These did not stop training, but they are caveats for any downstream comparison.

## Files

- `summary.json`: compact metrics, paths, and artifact status for this probe.
- `SHA256SUMS`: checksums for key ignored local artifacts.
