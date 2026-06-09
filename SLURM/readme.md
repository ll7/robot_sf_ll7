# SLURM

Canonical cluster workflow instructions for agents now live in `SLURM/AGENTS.md`.
Use that file for reusable submission policy, runtime checks, and required insight capture
between sessions.

For new long-running jobs, prefer `scripts/dev/sbatch_use_max_time.sh` over raw `sbatch`.
The wrapper queries the current partition and QoS limits and submits with the effective
max wall time by default. For private cluster overlays such as Auxme, see
[SLURM/Auxme/README.md](Auxme/README.md).

For post-run sizing decisions, use the SLURM + W&B audit workflow in
`docs/dev/slurm_resource_audit.md`. That runbook shows how to compare Slurm
allocations with W&B `history(stream="system")` metrics and explains why PPO
`num_envs` should not simply match reserved CPU count.

## Runtime and utilization evidence

Use the maintained audit workflow in `docs/dev/slurm_resource_audit.md` and the helper
`scripts/tools/collect_slurm_utilization.py` for CPU/GPU sizing evidence. Keep logs under
`output/slurm/`, and classify generated summaries before promoting any evidence into tracked docs.

Legacy helpers such as `SLURM/log_gpu_cpu_usage.py` and `SLURM/slurm_train.sl` are retained as
compatibility examples only. Do not use them as the starting point for new submissions; choose a
public template, the optional private operations overlay, or the LiCCA cluster-specific README plus
`SLURM/AGENTS.md` instead.
