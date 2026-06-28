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

Run the bounded empirical vertical-slice packet on a SLURM-capable host/account that can satisfy the
repository submission policy. The public packet is:

```bash
scripts/benchmark/run_issue3425_empirical_vertical_slice.sh
```

By default the script only generates the research campaign packet and camera-ready preflight
artifacts. The approved private execution route should call it from a clean public worktree with
`RUN_CAMPAIGN=1` after duplicate, queue, branch, commit, and local-machine submission checks pass.
The public campaign inputs are:

- `configs/benchmarks/issue_3425_empirical_vertical_slice_manifest.yaml`
- `configs/benchmarks/issue_3425_empirical_vertical_slice_smoke.yaml`

The slice intentionally uses three explicit Francis 2023 single-pedestrian scenarios, the `eval`
seed set `[111, 112, 113]`, and baseline-safe planners `goal`, `social_force`, and `orca`. This
avoids classic density-tier claims while issue #3725 remains open.

After execution, run the existing finalizer and reconciliation tools:

```bash
uv run python scripts/tools/slurm_job_finalize.py <finalizer args>
uv run python scripts/tools/reconcile_slurm_evidence.py <reconciliation args>
```

The public follow-up should record only public-safe fields: issue, campaign id, branch/commit,
launcher/config path, public partition/cluster class when allowed, job id, finalizer manifest path,
durable pointer status, compact summary path, and the final decision (`promote`,
`keep diagnostic`, `block`, or `stop`).

## Metric And Claim Gate

This slice must not promote planner rankings or paper-facing claims until the result is classified
with open prerequisites visible:

- #3724: collision and near-miss semantics freeze is still represented by open PR #3775 at the time
  this packet was prepared.
- #3482: `EpisodeEventLedger.v1` and comparator tooling are merged, but the durable frozen-trace
  backfill/application remains open.
- #3723 and #3699: Social Navigation Quality Index (SNQI) weights and normalization semantics remain
  decision-required.
- #3725: classic density-tier parameterization remains open; this packet avoids density-tier
  conclusions rather than resolving it.

Treat any completed run as `smoke` evidence at most until these gates are resolved or explicitly
caveated in the final decision.

## Local Readiness Gate (shared-PC-safe)

Before the slice is run on a SLURM-capable host, the inputs it will consume can be checked locally
with a CPU-only readiness gate. This does **not** submit SLURM work, run the reconciler end-to-end,
or promote any claim; it only reports whether the queue, submission manifest, and finalizer manifest
inputs are present and provenance-complete, and fails closed when durable pointers or manifest
linkage are missing:

```bash
uv run python scripts/validation/preflight_slurm_finalizer.py \
  --queue experiments/submission_queue.yaml \
  --submission-manifest <submission-manifest.yaml> \
  --finalizer-manifest <finalizer-manifest.json> \
  --json
```

Exit `0` means the slice inputs are ready; exit `1` lists each missing prerequisite together with
the smallest external action to unblock it; exit `2` means an input could not be parsed. The gate
reuses the canonical loaders in `scripts/tools/reconcile_slurm_evidence.py`, so it evaluates exactly
the inputs the real reconciliation consumes.

## Current Decision

`block`: no SLURM job was submitted, no finalizer manifest was produced, no durable pointer was
validated, and no claim decision can be promoted from this machine.

The next valid proof step is external execution on a SLURM-capable host, not a local simulation of
success.
