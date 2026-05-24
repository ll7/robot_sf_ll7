# Issue #1428 ORCA-Residual Lineage Evidence

This bundle preserves the small reviewable contract evidence for the pre-SLURM
ORCA-residual behavior-cloning lineage packet.

- `diagnostic_contract.json`: required per-step and aggregate diagnostics for deciding whether the
  learned residual contributes beyond ORCA and the hard guard.
- `orca_residual_guarded_ppo_v0_smoke_summary.json`: compact smoke proof that the existing runtime
  residual surface still runs after the lineage packet was added. This is not learned-residual
  training evidence.

Large datasets, checkpoints, raw logs, videos, and Slurm outputs remain out of git. The launch
packet records pending durable artifact URIs that the follow-up Slurm job must replace with
concrete pointers.
