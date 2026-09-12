# SLURM Submission Workflow

[← Back to Documentation Index](../README.md)

Use the shared wrapper below for new batch jobs so the requested wall time tracks the
current partition and QoS policy instead of relying on stale hardcoded `#SBATCH --time`
lines.

## Default workflow

```bash
scripts/dev/sbatch_use_max_time.sh <cluster-script.sl>
```

The wrapper:

- reads `#SBATCH --partition` and `#SBATCH --qos` from the target script,
- queries Slurm for the partition `MaxTime`,
- queries QoS `MaxWall` when that metadata is available,
- uses the effective maximum of the selected profile as the default `--time`, and
- passes that value to `sbatch`, which overrides the script-local time directive.

This keeps new submissions aligned with the live cluster policy even if the script still
contains an older fallback value.

## Post-run closeout

After a job reaches a terminal state, create the compact local closeout manifest with the
checked-in finalizer:

```bash
uv run python scripts/tools/slurm_job_finalize.py \
  --issue <issue-number> \
  --job-id <slurm-job-id> \
  --job-state <observed-state> \
  --control-plane-run-root <run-root> \
  --output <run-root>/slurm_finalization.json \
  --markdown-output <run-root>/slurm_finalization.md
```

Use `--expected-artifact <path>` repeatedly instead of `--control-plane-run-root` when the
run does not use the research-control-plane artifact set. Add `--optional-artifact <path>`
for non-required files, and pass `--post-campaign-stage-status <path>` only when the launcher
has emitted the validated `robot-sf-post-campaign-stage-status.v1` envelope. The finalizer
records the observed job state, artifact presence, checksums, and an issue-update summary; it
does not submit or poll jobs, upload files, or copy raw `output/` trees.

Interpret the result conservatively: `success` means the observed terminal state and required
local artifacts passed the helper's checks, while `missing_artifacts`, `failed`, `incomplete`,
`not_available`, or `manual_decision_required` require the corresponding follow-up rather than
being treated as benchmark success. A successful local closeout remains `pending_durable`
until a retrievable durable artifact URI is recorded with `--durable-uri`; local paths alone
are not durable evidence. Keep the issue/PR traceability checklist below in the public handoff,
and keep private host, account, QoS, and scratch details out of public comments.

## Failure classification (check-only)

Before deciding whether to resubmit, classify an operational failure from a sanitized
scheduler/launcher receipt and optional bounded log excerpt:

```bash
uv run python scripts/tools/classify_scheduler_failure.py \
  --check --receipt <sanitized-receipt.json> \
  [--log-excerpt <bounded-stderr-tail.log>] --format json
```

The classifier is versioned (`robot_sf.scheduler_failure_vocabulary.v1`) and always
read-only: it never submits, retries, cancels, or interprets scientific outcomes. Structured
scheduler, launcher, artifact, and job-manifest fields take precedence; log fingerprints are
secondary evidence at lower confidence. Every report records the class, evidence source,
confidence, retryability under the existing policy, owner, required remediation, and whether
outputs may still require harvest. Contradictory evidence stays `multiple_causes`; missing
evidence stays `unknown`; an application nonzero exit is never auto-retried as
infrastructure, and a completed scheduler state is never scientific success. Exit code 2
means the receipt was unreadable or malformed, not that the job failed.

## Array-index mapping verification (check-only)

Before launching array-based SLURM jobs or retries, verify that array task IDs map
bijectively and without bounds errors to campaign matrix rows:

```bash
uv run python scripts/validation/verify_slurm_array_mapping.py \
  --manifest path/to/campaign_manifest.json \
  --array-spec 0-99%10 \
  --step-chunk 1 \
  --output-dir output/benchmarks/campaign_1/
```

The verifier is read-only and fail-closed: it checks for gaps, duplicate row mappings,
off-by-one bounds, zero- vs one-based index mismatches, chunk tail truncations, shard
reorderings, resume/retry collisions, and invalid array concurrency specs. It exits with code
0 on success, 1 on mapping errors (or warnings under `--strict`), and 2 on invalid invocation.

## Receipt public projection (check-only)

Before a private scheduler, harvest, or custody receipt is quoted publicly, validate the proposed
sanitized projection against `scripts/tools/receipt_projection_policies.json`:

```bash
uv run python scripts/tools/validate_receipt_projection.py \
  --check --private <private-receipt.json> --public <proposed-public-receipt.json> --format json
```

Policies are explicit per field path (`keep`, alias transforms, `digest`, `omit`, `reject`).
Unknown or credential-named fields, private paths, hostnames, IPs, signed URLs, accounts, and
queue topology fail closed. Approved stable aliases replace private IDs and artifact roots, and
the public receipt binds `source_binding.receipt_sha256` to the canonical SHA-256 of the full
private receipt without publishing private bytes; repeated projection is byte-stable. Required
identity (source/config digests, environment class, row counts, terminal status, artifact
checksums, claim boundary) must survive, and over-redaction that erases it fails. Unsupported
classes are reported `unsupported_receipt_class`, never partially projected. Exit codes: 0 valid,
2 invalid or unsupported, 3 malformed input; the tool is check-only and changes no state.

## Scheduler-job reconciliation (check-only)

Before access ends, join a sanitized scheduler inventory projection with public issue/PR states,
immutable launch-manifest packet digests, expected row counts, and artifact owner/harvest metadata
so unbound, duplicate, stale, or ownerless jobs stay visible without exposing private infrastructure:

```bash
uv run python scripts/tools/reconcile_scheduler_jobs.py \
  --check --projection <sanitized-projection.json> [--public <public-snapshot.json>] --format json
```

Each output row binds one sanitized job alias (plus optional array index and parent lineage) to its
exact public owner, immutable input packet digest, artifact root, owner, harvest state, transfer
state, and next action. Rows classify as `owned_active`, `owned_terminal_unharvested`,
`owned_harvested`, `duplicate_candidate`, `orphan_unknown`, `stale_input`, `missing_output_owner`,
or `projection_unavailable`. Ownership is never inferred from mutable job names: missing or
unsanitized identity fails closed, equivalent active duplicates and conflicting owner bindings are
reported, and `scheduler_state == completed` never implies artifact completeness or result
validity. The report is byte-stable after documented volatile-field normalization (`generated_at`,
`snapshot_at`, `observed_at`, and peers are dropped). Exit codes: 0 every row active or harvested,
1 actionable rows, 2 malformed input. The tool is check-only: it never submits, cancels, relabels,
claims, comments, or deletes scheduler, GitHub, or artifact state.

## Staged source isolation verification

Before submitting compute-window or cluster jobs, verify that staged commands run
strictly from the immutable staged source without leakage from ambient `PYTHONPATH`,
user-site packages, sibling worktree checkouts, stale editable installations, or `.pth`
injections:

```bash
uv run python scripts/validation/verify_staged_source_isolation.py \
  --packet <staging-packet-or-bundle.json> \
  --format json
```

The verifier executes bounded import and startup probes in an isolated subprocess,
validates that all first-party imports resolve inside the staged source or an explicitly
declared companion (`--companion NAME=PATH`), checks git tree cleanliness, and emits a
sanitized `staged_source_isolation_receipt.v1` artifact without revealing private host paths.
Exit codes: `0` passed, `1` blocked, `2` malformed.

## Running-job monitor (check-only)

While a job is pending or running, reduce explicit sanitized observations without cancelling,
retrying, or harvesting anything:

```bash
uv run python scripts/tools/monitor_running_jobs.py \
  --check --projection <sanitized-projection.json> --once --format json
```

The projection lists only the intended job identities plus expected artifacts/rows and the
harvest request/artifact-root packet. The monitor verifies job, source, and immutable
submission-receipt identity before every state reduction, records transitions, observation
timestamps, evidence digests, and array summaries, and emits `running_job_harvest_handoff.v1`
naming the canonical `scripts/validation/harvest_terminal_job.py --check` command when a
terminal state is observed. For live polling, pass `--state-query "<read-only command with
{job_id}>"` with `--interval` and a hard `--max-wall-seconds`; expiry emits
`monitor_window_expired` with the current state instead of classifying the job terminal. The
query subprocess timeout is clamped to the remaining wall-clock budget. The monitor never
cancels, retries, submits, or harvests, and scheduler completion is never artifact or scientific
success.

## SLURM launcher static audit (check-only)

Before submitting or handing off SLURM scripts and wrappers, audit them for stale partitions,
missing job names, missing timeouts, unsafe log paths, hardcoded host/user paths, unbounded
arrays, conflicting GPU requests, stale module commands, missing preflight, and non-portable resume:

```bash
uv run python scripts/validation/audit_slurm_launchers.py \
  [<launcher-files>...] --check --format json
```

The audit tool is static, credential-safe, and read-only. It parses `#SBATCH` directives and
wrapper flags, compares them against sanitized capability classes, and reports violations
with exit code 0 (clean) or 1 (errors found).


## Training submission queue

Use `experiments/submission_queue.yaml` for reviewable planned training submissions that should be
safe for agents to dry-run and, on SLURM-capable hosts, auto-submit after all gates pass. The queue
is planned intent only: GitHub issues remain the backlog, and submitted/running/completed state
belongs in issue comments or `docs/context/issue_1544_slurm_experiment_state_ledger.md`.

Run a dry-run manifest before any submission:

```bash
uv run python scripts/dev/submit_training_jobs.py --dry-run
```

This validates queue entries, records branch/commit/dirty-tree state, builds the wrapper command,
checks local duplicate evidence such as existing output roots, and writes a timestamped manifest
under `output/slurm/submissions/`.

Submit eligible entries only from a SLURM-capable host:

```bash
uv run python scripts/dev/submit_training_jobs.py --submit
```

Submit mode additionally requires:

- `local.machine.md` must explicitly set `allow_slurm_submission: true`;
- the entry status is `ready_to_submit`;
- `auto_submit: true`;
- the config or launcher path exists;
- the output root is absent;
- live `squeue` and recent `sacct` checks do not show an equivalent job;
- no other active `gse-` training job is present;
- the existing wrapper exits successfully and returns a job id.

Equivalent submissions are duplicates when they match the same issue/objective lane, config or
launcher, seed set, commit or declared code version, target cluster, job name, or output root. A
duplicate blocks `--submit`; reruns should use a new queue id, changed output root, and documented
reason.

Final reports should include the generated manifest path, job id, public partition or cluster label,
command, branch, commit SHA, config, launcher, seed set, output/log paths, skipped entries,
monitoring command, and private ledger reference when applicable.

## Examples

Dry run before submitting:

```bash
scripts/dev/sbatch_use_max_time.sh --dry-run <cluster-script.sl>
```

Override partition or QoS discovery when testing a variant on a configured cluster:

```bash
scripts/dev/sbatch_use_max_time.sh \
  --partition <partition> --qos <qos> \
  --sbatch-arg --partition=<partition> \
  --sbatch-arg --qos=<qos> \
  SLURM/templates/gpu_training.sl
```

Force a shorter manual wall time when needed:

```bash
scripts/dev/sbatch_use_max_time.sh --time 08:00:00 SLURM/templates/gpu_training.sl
```

## Guidance

- Prefer the wrapper for long-running training jobs.
- Keep explicit short limits only for intentionally bounded jobs such as setup or quick
  interactive sessions.
- Before submitting, estimate the intended runtime from the config or preflight output: rows,
  scenarios, seeds, episodes, horizon, workers, GPU need, and expected artifacts. A benchmark or
  training run expected to finish in under 1 hour should default to local execution, not SLURM.
- Submit a sub-1-hour run to SLURM only when the point is compute-node proof: GPU-only execution,
  cluster-only dependencies, queue/runtime parity, or a maintainer-approved smoke. Label that job
  and any issue/PR follow-up as `smoke` or `probe`, not as completed campaign evidence.
- Do not let a GitHub `slurm` label alone justify submission. The label means cluster execution may
  be required; the config still needs a suitability check.
- For long-running training jobs, include predeclared early-stop criteria in the experiment card or
  launch packet before submission. The criteria must name the metric, threshold, check cadence,
  minimum runtime or timesteps, exact cancel condition, and diagnostic-preservation action.
- A cancelled Slurm run can be valid diagnostic evidence only when the stop rule was predeclared and
  the branch, commit, config, logs, manifest, local output root, and durable artifact/preservation
  status are recorded. Without that preservation trail, classify the run as failed, blocked, or
  inconclusive rather than proof.
- When adding a new batch script, include `#SBATCH --partition` and `#SBATCH --qos` so
  the wrapper can resolve the correct limit without extra flags.
- If Slurm tools are unavailable in the current shell, fall back to a manual `sbatch`
  command with an explicit `--time`.

## Shared SLURM Traceability Checklist

Use this single checklist for public issue/PR comments and private-ops ledger/handoff updates.
Do not duplicate fields into separate public/private lists; use the same checklist with public-safe
fields in public surfaces and private-only fields in private surfaces.

- Submission intent
  - issue/PR reference
  - experiment intent or hypothesis
  - expected evidence tier (`smoke`, `probe`, `campaign`, or diagnostic)
- Launch identity
  - command surface and launcher path
  - config path or exact snapshot
  - branch, commit, and dirty-tree status
  - partition and job name
- Route and health evidence
  - submitted job id
  - immediate route check outcome (`squeue`, `sacct`, earliest stderr/early log tail)
  - submitter finalizer command and exit/result
  - queue/duplicate gate status (satisfied or reason blocked)
- Public/Private trace
  - issue/PR comment id and posted fields (job id, partition, branch, commit, config, outputs)
  - private ops ledger/handoff reference (or equivalent private record)
- Output and artifacts
  - output root
  - manifest path, checkpoints, report path, and artifact status (`non_durable`, `durable`,
    `promoted`, or `discarded`)
- Completion and follow-up
  - run classification (`completed_needs_analysis`, `diagnostic`, `failed_preflight`, `blocked`,
    `partial_traceable`)
  - next action and rerun decision

Submission state rules:

- `sbatch`/`sacct` success or a job id is route evidence only.
- Use `submitted` only when the immediate health check succeeds and both traceability records are complete.
- If either route traceability step is missing, use `partial_traceable` and keep status explicitly blocked.
- Public issue/PR comments may include job id and partition for traceability, but must not include private host
  names, account/QoS details, scratch paths, or private retrieval mechanics.

## Output capacity preflight (check-only)

Before submitting a job whose results must be preserved, estimate the full output and
post-run transfer budget and compare it with a sanitized storage-capability projection:

```bash
uv run python scripts/tools/check_output_capacity_preflight.py --check \
  --packet path/to/capacity_packet.json \
  --storage-projection path/to/storage_capability_projection.json \
  --format json
```

The packet (`robot_sf.output_capacity_preflight_packet.v1`) declares the expected row
count and scaling, one component per output surface with `storage_class` (`task_output`,
`scheduler_log`, `temporary_scratch`, `durable_required`, `disposable_post_verification`)
and `output_kind` (rows, logs, checkpoints, harvest_manifest, checksums,
compression_workspace, temporary_workspace, other), and lower/expected/conservative-upper
per-row plus fixed bounds for bytes, files, and peak bytes. Every `empirical` component
must name a compatible `source_identity`; otherwise use `declared` bounds or an explicit
`unavailable` token. The projection (`robot_sf.storage_capability_projection.v1`) carries
sanitized source/destination free bytes, free inodes, and retention classes, the reserved
byte/inode/time safety margin, the access deadline, and the transfer route's rate bounds
plus rate uncertainty.

The verdict is fail closed. `capacity_ok` requires conservative upper bounds plus margin
to fit source and destination and the conservative transfer duration to fit before the
access deadline. `capacity_exceeded` reports conservative misses. `capacity_unknown` is
returned when row scaling, any component dimension, temporary workspace, inode use,
destination capacity, transfer rate, or access deadline is unbounded or unavailable, and
an unknown verdict never passes. The tool is check-only: it never deletes, compresses, or
mutates campaign artifacts. Exit codes are 0 `capacity_ok`, 2 `capacity_exceeded` or
`capacity_unknown`, and 3 malformed input. Fixtures for the passing, exceeded, and
unknown cases live under `tests/tools/fixtures/output_capacity_preflight/`.

## Capacity-aware / fill batches

When submitting a batch intended to fill spare cluster capacity (rather than a single prioritized
experiment), the batch may include several unrelated ready experiments, but it must satisfy these
public-safe preconditions before `sbatch`:

1. Live queue evidence: refresh `squeue --me` and partition-wide `squeue` so submissions reflect
   current load.
2. Bounded scope: declare a maximum job count, total GPU/CPU budget, or wall-time cap derived from
   visible spare capacity.
3. Duplicate checks: each job passes the standard duplicate gate (same issue/objective lane, config,
   seed set, commit, cluster, job name, output root).
4. Traceability: every job gets the shared traceability checklist (issue/PR comment or private-ledger
   handoff plus immediate health check).
5. Immediate health check: verify `squeue` acceptance after each submission; halt the batch on the
   first failure.
6. Polite scheduling: where the scheduler supports it, use `--nice` factors or equivalent to avoid
   displacing higher-priority work.
7. Avoid resource starvation: do not saturate a partition so heavily that other users' eligible jobs
   cannot start within a reasonable window.
8. Cluster-specific leave-one-way rule (imech192): when submitting on imech192, always preserve at
   least one GPUxCPU way free for other users unless the queue is empty or the maintainer explicitly
   overrides.

Capacity-aware batches use the same shared traceability checklist and submission-state rules as any
other SLURM submission. Private cluster mechanics (hostnames, QoS tuning, scratch paths) stay in the
private operations overlay and must not appear in public issue/PR comments.

## Multiple branches from one login node

When two active branches need to submit or monitor SLURM jobs from the same login node, prefer one
Git worktree per branch. Submit from the worktree whose branch, configs, and SLURM scripts should
be used by the job:

```bash
cd ~/git/robot_sf_ll7
mkdir -p ../robot_sf_ll7.worktrees
git fetch origin codex/193-feature-extractor-evaluation
git worktree add -b codex/193-feature-extractor-evaluation \
  ../robot_sf_ll7.worktrees/codex-193-feature-extractor-evaluation \
  origin/codex/193-feature-extractor-evaluation
cd ../robot_sf_ll7.worktrees/codex-193-feature-extractor-evaluation
scripts/dev/sbatch_use_max_time.sh SLURM/feature_extractor_comparison/run_comparison.slurm
```

This is safer than switching one checkout between branches while jobs are pending because SLURM
sets `SLURM_SUBMIT_DIR` to the directory where `sbatch` was called, and repository wrappers often
use that directory or resolve the Git root from it before reading configs.

This isolates branches, not file snapshots. Pending jobs normally read the worktree contents when
they start, so avoid incompatible edits to that worktree's configs or scripts while a queued job is
waiting.

If submission is performed by a private wrapper over SSH, create or refresh the owning worktree on
the submit host before `sbatch`, then record the host-side branch, commit, and clean status. A local
dry run proves the command shape, but it does not prove that the cluster can see the same worktree.

`local.machine.md` is gitignored. If the same login-node policy should apply to every local
worktree, symlink it from the original checkout:

```bash
ln -s ../../robot_sf_ll7/local.machine.md local.machine.md
```

Keep `.venv` branch-local unless the branches are known to have identical dependencies; most SLURM
scripts expect `.venv` under the submit worktree. See the durable workflow note:
[SLURM Multi-Worktree Branch Workflow](../context/slurm_multi_worktree_branch_workflow.md).

## Private Cluster Overlays

Cluster-specific hostnames, QoS policies, node-packing heuristics, local scratch paths, and
machine-only runbooks should live outside this public repository. The public repo keeps the
portable experiment contract: checked-in configs, generic wrapper behavior, artifact policy,
validation helpers, and reviewable evidence manifests.

Configure the optional private operations overlay with either an environment variable:

```bash
export ROBOT_SF_PRIVATE_OPS=/path/to/robot_sf_ll7-private-ops
```

or a gitignored local machine context entry:

```markdown
- private_ops_repo: /path/to/robot_sf_ll7-private-ops
```

When neither is set, `scripts/dev/private_ops.sh` falls back to a sibling checkout named
`robot_sf_ll7-private-ops` next to the current worktree's parent directory.

For worktrees, prefer one sibling private overlay shared by all checkouts:

```text
~/git/robot_sf_ll7/
~/git/robot_sf_ll7.worktrees/<branch>/
~/git/robot_sf_ll7-private-ops/
```

When a private overlay queue entry is blocked on a prerequisite that may have changed, reconcile the
blocker against current public issue comments, merged PRs, and the relevant source/tests before
submitting. If the prerequisite is satisfied, record that evidence in the private ledger or handoff;
if the queue and public state conflict without clear evidence, keep the job blocked.

## Auxme issue-791 private helper

For issue-791 wrappers on Auxme, use:

```bash
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22.yaml \
  --job-name robot-sf-issue791-reward-curriculum \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

This public helper delegates to the private operations overlay. The private implementation adds
pre-submit partition availability checks and recommendation logic based on current cluster pressure,
then submits through the public `sbatch_use_max_time.sh` in the active worktree.

Raw status table only:

```bash
scripts/dev/auxme_partition_status.sh
```

Machine-readable recommendation only:

```bash
scripts/dev/auxme_partition_status.sh --recommend
```

## Camera-ready benchmark campaigns

For camera-ready benchmark campaigns on a private cluster, prefer a generic launcher in the private
overlay rather than cloning an issue-specific public script:

```bash
CAMERA_READY_BENCHMARK_CONFIG=configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml \
CAMERA_READY_BENCHMARK_LABEL=issue999-preflight \
CAMERA_READY_BENCHMARK_MODE=preflight \
scripts/dev/sbatch_use_max_time.sh --dry-run <private-camera-ready-benchmark.sl>
```

Submit the full run by removing `--dry-run` and setting the intended artifact root:

```bash
CAMERA_READY_BENCHMARK_CONFIG=configs/benchmarks/paper_experiment_matrix_v1_issue_791_eval_aligned_compare.yaml \
CAMERA_READY_BENCHMARK_LABEL=issue999-camera-ready \
CAMERA_READY_BENCHMARK_OUTPUT_ROOT=output/benchmarks/issue_999 \
scripts/dev/sbatch_use_max_time.sh <private-camera-ready-benchmark.sl>
```

`CAMERA_READY_BENCHMARK_MODE=preflight` and `run` are both supported. The launcher requires an
explicit config and either `CAMERA_READY_BENCHMARK_LABEL` or `CAMERA_READY_BENCHMARK_CAMPAIGN_ID`
so queued jobs have a reviewable identity before they consume cluster time. Slurm logs stay under
`output/slurm/`; campaign outputs should stay under `output/benchmarks/...` unless a small
manifest, summary, or durable artifact pointer is intentionally promoted.

## Campaign input drift verification (check-only)

Before mutating the scheduler or submitting scarce compute-window jobs, verify that the live
submission packet has not drifted from the preflight receipt using the compare-and-swap binding
validator:

```bash
uv run python scripts/validation/validate_campaign_submission_binding.py --check \
  --preflight path/to/preflight_receipt.json \
  --submission path/to/live_submission_packet.json \
  --format json
```

The validator operates in check-only mode and fails closed: it recomputes and compares all
authority-bearing identities (Git commit SHA, working tree dirty state, config content SHA-256,
seed ordering, model checkpoint hash, Python/lock environment, expected row count, resource
allocations, output root, command tokens, admission claim, and duplicate execution state).
Permitted volatile fields (`observed_at_utc`, `submission_nonce`, `pid`, `hostname`, `host`,
`process_id`, `job_id_pending`) are tracked and reported in `volatile_fields_observed` without
causing false drift failures, while any unpermitted or unknown field divergence blocks submission
and exits with code 1.

## Expected-row ledger generation and verification (fail-closed)

Expand campaign packets into canonical byte-stable expected-row ledgers before launch:

```bash
uv run python scripts/validation/generate_campaign_row_ledger.py \
  --packet path/to/campaign_packet.json \
  --output path/to/campaign_expected_row_ledger.json \
  --check
```

The tool enforces schema `campaign_expected_row_ledger.v1.schema.json` across Cartesian grids, paired
arms, array tasks, and excluded-cells pruning. Every row receives a unique key
(`campaign_id::arm::scenario_id::seed::replicate`). Staging fails closed (`--check` exits with 1) on
duplicate identities, underspecified dimensions, unresolved aliases, mutable paths, or count mismatches.
Observed rows (`--observed path/to/rows.jsonl`) verify completion across 9 row states (`present`, `missing`,
`duplicate`, `unexpected`, `conflict`, `fallback`, `degraded`, `failed`, `provenance_invalid`).

## Campaign recovery and retry verification (check-only)

Before resuming an interrupted campaign or resubmitting uncompleted cells, verify recovery and retry
behavior under fail-closed contracts:

```bash
uv run python scripts/validation/verify_campaign_recovery.py \
  --fixture path/to/campaign_recovery_packet.json \
  --output path/to/campaign_recovery_receipt.json \
  --format json \
  --check
```

The verifier enforces schema `campaign_recovery_receipt.v1.schema.json` and evaluates:
- **Preservation of valid completed identities**: completed rows are never rerun or overwritten.
- **Fail-closed retry admission**: outcome-driven failures (collisions, task failure) cannot be retried
  away under infrastructure labels; retries are admitted only for documented infrastructure interruptions.
- **Authority input drift**: commit, config SHA-256, and model digest must match across attempts.
- **Degraded/fallback protection**: fallback executions cannot become clean successes through resume.
- **Lineage tracking**: scheduler job IDs and attempt indices are recorded across executions.
- **Supported runners**: canonical runners (`benchmark_matrix`, `slurm_array`) are verified; unknown runners
  report `status: "unsupported"`.
- **Ledger reconciliation**: final reconciled rows must match the expected-row ledger 1-to-1.



