# Issue 845: Slurm Utilization Probe

## Current Boundary

This issue needs live Slurm accounting evidence from representative jobs. The current local machine
context does not permit Slurm submission, so this branch prepares the reusable measurement path
without launching or modifying cluster jobs.

## Probe Command

Run on a Slurm login node with the representative job IDs:

```bash
uv run python scripts/tools/collect_slurm_utilization.py \
  <jobid-a> <jobid-b> \
  --output-json output/slurm/issue845_utilization.json \
  --output-md output/slurm/issue845_utilization.md
```

The collector records:

- `sstat -j <jobid>.batch --format=JobID,AveCPU,MaxRSS,MaxVMSize,AveRSS`
- `sacct -j <jobid> --format=JobID,JobName%40,Partition,State,Elapsed,AllocCPUS,AveCPU,TotalCPU,MaxRSS,ReqMem,MaxVMSize,ExitCode -P`
- `seff <jobid>` when available

## Interpretation

Use the generated Markdown report to compare:

- requested CPU allocation (`AllocCPUS`) against observed CPU time (`AveCPU`, `TotalCPU`),
- requested memory (`ReqMem`) against observed memory (`MaxRSS`, `MaxVMSize`),
- live batch-step resource values from `sstat` while jobs are still running,
- missing Slurm tools as environment blockers rather than utilization evidence.

## Follow-Up Boundary

Do not classify the root cause until at least two representative jobs have real accounting output.
If the evidence shows consistent over-requesting, split a concrete config or launcher tuning issue.
If it shows Python/runtime bottlenecks, split a profiling issue with the exact workload and job IDs.

## Validation

```bash
uv run pytest tests/tools/test_collect_slurm_utilization.py -q
uv run python scripts/tools/collect_slurm_utilization.py --help
git diff --check
```
