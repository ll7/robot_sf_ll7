# SLURM Submission Workflow

[← Back to Documentation Index](../README.md)

Use the shared wrapper below for new batch jobs so the requested wall time tracks the
current partition and QoS policy instead of relying on stale hardcoded `#SBATCH --time`
lines.

## Default workflow

```bash
scripts/dev/sbatch_use_max_time.sh SLURM/Auxme/auxme_gpu.sl
```

The wrapper:

- reads `#SBATCH --partition` and `#SBATCH --qos` from the target script,
- queries Slurm for the partition `MaxTime`,
- queries QoS `MaxWall` when that metadata is available,
- uses the effective maximum of the selected profile as the default `--time`, and
- passes that value to `sbatch`, which overrides the script-local time directive.

This keeps new submissions aligned with the live cluster policy even if the script still
contains an older fallback value.

## Examples

Dry run before submitting:

```bash
scripts/dev/sbatch_use_max_time.sh --dry-run SLURM/Auxme/auxme_gpu.sl
```

Override partition or QoS discovery when testing a variant:

```bash
scripts/dev/sbatch_use_max_time.sh --partition a30 --qos a30-gpu SLURM/Auxme/auxme_gpu.sl
```

Force a shorter manual wall time when needed:

```bash
scripts/dev/sbatch_use_max_time.sh --time 08:00:00 SLURM/Auxme/auxme_gpu.sl
```

## Guidance

- Prefer the wrapper for long-running training jobs.
- Keep explicit short limits only for intentionally bounded jobs such as setup or quick
  interactive sessions.
- When adding a new batch script, include `#SBATCH --partition` and `#SBATCH --qos` so
  the wrapper can resolve the correct limit without extra flags.
- If Slurm tools are unavailable in the current shell, fall back to a manual `sbatch`
  command with an explicit `--time`.