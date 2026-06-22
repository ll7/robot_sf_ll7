# Issue #3425 SLURM (Simple Linux Utility for Resource Management)-to-Claim Blocker

Issue: [#3425](https://github.com/ll7/robot_sf_ll7/issues/3425)

Status: blocked on SLURM-capable execution from the current machine. This note is not benchmark
evidence and does not demonstrate the requested submission -> finalizer -> durable evidence ->
summary -> claim-decision chain.

## Goal

Issue #3425 asks for one complete vertical research slice through the SLURM finalizer bridge and
campaign manifest workflow:

1. submit a small campaign,
2. finalize the run,
3. reconcile durable evidence,
4. produce a compact summary/table,
5. make a final claim decision.

## Current Blocker

The current local machine context forbids SLURM submission:

- `local.machine.md`: `allow_slurm_submission: false`
- `local.machine.md`: `slurm_submit_command: none`
- Shell preflight: `sbatch`, `squeue`, and `sacct` are not available on `PATH`

Repository submission policy also requires a SLURM-capable host before submit mode:

- `docs/dev/slurm_submission.md` requires `local.machine.md` to explicitly set
  `allow_slurm_submission: true`.
- The same workflow requires live queue/account checks before `sbatch`.

Because those preconditions are absent, submitting or simulating a successful SLURM-to-claim chain
from this machine would violate the issue contract.

## Smallest External Action

Run the vertical slice from a SLURM-capable host/account that can satisfy the repository submission
policy:

```bash
uv run python scripts/validation/run_research_campaign_manifest.py \
  configs/benchmarks/research_campaign_manifest.example.yaml \
  --output-dir output/research_campaign_packets/issue_3425_preflight
```

Then choose a tiny campaign whose expected runtime and artifact set are acceptable for cluster
execution, submit through the approved wrapper or queue workflow, and run the existing finalizer and
reconciliation tools:

```bash
scripts/dev/sbatch_use_max_time.sh <public-safe-cluster-script.sl>
uv run python scripts/tools/slurm_job_finalize.py <finalizer args>
uv run python scripts/tools/reconcile_slurm_evidence.py <reconciliation args>
```

The public follow-up should record only public-safe fields: issue, campaign id, branch/commit,
launcher/config path, public partition/cluster class when allowed, job id, finalizer manifest path,
durable pointer status, compact summary path, and the final decision (`promote`,
`keep diagnostic`, `block`, or `stop`).

## Current Decision

`block`: no SLURM job was submitted, no finalizer manifest was produced, no durable pointer was
validated, and no claim decision can be promoted from this machine.

The next valid proof step is external execution on a SLURM-capable host, not a local simulation of
success.
