# SLURM Agent Playbook

Durable cluster safeguards for jobs submitted under `SLURM/` or in related training campaigns. Pair
this with [AGENTS.md](../AGENTS.md), [SLURM/readme.md](readme.md), and
[docs/dev/slurm_resource_audit.md](../docs/dev/slurm_resource_audit.md).

Auxme-cluster-specific details (node names, partitions, QoS profiles, per-user job limits, and
host-packing policy) live in the optional private operations overlay. See
[SLURM/Auxme/README.md](Auxme/README.md) for the public overlay contract. Never copy secrets or
host-specific policy into the repository.

## Configuration And Submission

- Use explicit, versioned configuration. Pass required environment and config inputs explicitly; do
  not rely on wrapper defaults for promotion or long-horizon submissions.
- Public wrappers validate their required environment and config inputs and fail with an actionable,
  non-secret message, including when the private operations overlay is absent.
- Prefer the repository wrappers under `scripts/dev/` over raw `sbatch`. For long jobs, prefer
  `scripts/dev/sbatch_use_max_time.sh` unless fixed wall time is required.
- Keep artifact output rooted in the configured artifact root and confirm synchronization on exit.
  Set `#SBATCH --output` to an absolute path in the job's results directory, such as
  `/path/to/job-results/%j-<description>.out`. Create the log directory before submission.
- Before submitting jobs, prepare one checkout per commit at
  `$EVIDENCE_ROOT/checkouts/<full-commit-sha>` on the shared home. Build its shared
  `.venv` once per commit with every extra the campaign needs, such as `orca`.
  Jobs use the checkout and virtual environment read-only and never run `uv sync`
  against the shared environment. Before using either, each job runs
  `git -C "$CHECKOUT" rev-parse --verify HEAD` with `CHECKOUT` set to that directory.
  Stop the job if the command fails or the full SHA differs from the commit in its
  submit receipt. Direct logs and outputs to the job's results path or `WORKDIR`,
  never into the checkout. Put caches and other temporary writes in `WORKDIR`.
  Before submission, verify that the launcher uses the prebuilt environment
  without syncing and routes its writes to these paths.

  Write results directly to a durable path on the shared home. Use a dedicated job-owned
  node-local `WORKDIR` for short-lived scratch. The job creates this directory and
  installs the cleanup trap before doing work.

  ```bash
  WORKDIR="${SLURM_TMPDIR:-/tmp/$USER}/job-${SLURM_JOB_ID}"
  mkdir -p -- "$WORKDIR" || exit 1
  RESULTS_VERIFIED=0
  trap '
    if [[ "$RESULTS_VERIFIED" == 1 ]]; then
      cd / && rm -rf -- "$WORKDIR"
    else
      printf "WORKDIR kept: %s\nReason: result copy and verification did not both succeed.\n" "$WORKDIR" >&2
    fi
  ' EXIT
  ```

  Verify that all expected results are present and readable at the durable destination
  before setting `RESULTS_VERIFIED=1`, even when no copy is needed. If a result was
  created in scratch, copy it to the durable evidence path or the shared home before
  the job ends. Set `RESULTS_VERIFIED=1` only after every result copy and its byte
  comparison or checksum verification succeed. This example packages all results in one
  temporary archive and sets `RESULTS_DEST` to the durable destination directory.

  ```bash
  cp -- "$WORKDIR/results.tar" "$RESULTS_DEST/results.tar" || exit 1
  cmp -- "$WORKDIR/results.tar" "$RESULTS_DEST/results.tar" || exit 1
  RESULTS_VERIFIED=1
  ```

  The trap and submit receipt cover only the job-owned `WORKDIR`. After successful result
  verification, confirm that `WORKDIR` is gone and record that in the submit receipt.
  If the job log reports that cleanup kept the directory, record
  `WORKDIR kept: <path>` with the logged path in the private submit receipt. Use that path to
  recover the results while scratch is still available. Scheduler-managed scratch may be
  reclaimed by the site after the job exits.

  After the campaign, the campaign owner removes old shared checkouts and their virtual
  environments once all jobs using them have finished. List this cleanup step in the
  submit receipt and record which versions it removed.

## Custody And Identity

- Record job ID, config path, commit SHA, seeds, and artifact root for every submission.
- Preserve logs and minimal reproduction evidence for failures.

## Failure Classification

- Distinguish infrastructure or transient failures (for example an allocation handshake failure)
  from model or algorithm failures. Classify infrastructure failures as such and follow the
  campaign's documented resubmission rule; do not treat them as model failure.
- Missing artifacts in the expected output path: verify the artifact root and cleanup/sync path
  before concluding a run produced no results; check `/tmp/<user>/<jobid>/results`-style roots.

## Evidence Boundary

- Do not claim performance improvement from short gate runs alone, and do not freeze one campaign's
  horizon stages as durable repository law; stage progression and baseline anchoring are campaign
  decisions recorded with the active campaign.
- Fail closed on unsupported claims: preserve logs and block promotion claims until finite,
  deterministic evaluation is restored.
- Promotion and paper-facing claims require the campaign's minimum evidence. Fallback or degraded
  evidence is never success evidence.

## Transient Campaign Guidance

Campaign-specific environment variables, wrappers, trackers, horizons, baselines, and progression
rules belong to the active campaign configuration and context record, not this file. Keep the
current owner and review condition here:

| Transient guidance | Owner | Review condition |
| --- | --- | --- |
| issue-791 wrapper env vars, WandB policy, horizon stages, resubmission rules | [`configs/training/`](../configs/training/) plus the active campaign's [context record via the context index](../docs/context/README.md) | Re-review when the issue-791 campaign closes or its wrapper contract changes |

## Insight Capture

Persist reusable findings that change durable practice before closing the task. Examples: reliable
load-recovery signatures, best `num_envs`/CPU pairings by host, stable eval-cadence ranges, and
proven mitigations. Do not create a documentation edit when a run produced no material, reusable
finding; record the negative result instead.
