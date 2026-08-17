# Slurm Launch Packet: Issue #1554

- Issue: 1554
- Title: job13198 constraints-first-analysis
- Training config: `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml`
- Intent: auto-prepared long GPU training prerequisite from the private Slurm queue.
- Submission owner: private `robot_sf_ll7-private-ops` queue and ledger.

This branch exists so Slurm submissions are never launched from the main public checkout.
Public code/config edits needed for the run should be made in this worktree before submission.
