# Slurm Launch Packet: Issue #4244

- Issue: https://github.com/ll7/robot_sf_ll7/issues/4244
- Title: training: pre-declared 7-arm comparison matrix (consolidation of the smoke-lane zoo — no new algorithms)
- Training config: `configs/training/ppo/ablations/expert_ppo_issue_2557_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_seed523_fixed.yaml`
- Intent: auto-prepared long GPU training prerequisite from the private Slurm queue.
- Submission owner: private `robot_sf_ll7-private-ops` queue and ledger.

This branch exists so Slurm submissions are never launched from the main public checkout.
Public code/config edits needed for the run should be made in this worktree before submission.
