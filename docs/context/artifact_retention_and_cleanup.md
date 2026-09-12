# Artifact Retention, Preservation, and Cleanup Guide

Plain-language summary: this guide explains which Robot SF outputs are disposable, diagnostic,
promoted, or protected; what counts as verified preservation; and when cleanup is safe. It is
derived from current owners and changes no retention, evidence, or release policy.

Status: operational guide. Canonical policy remains with the linked owners below.

## Canonical owners

| Concern | Owner |
| --- | --- |
| Evidence category vocabulary and `output/` rules | [artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md) |
| Tracked evidence bundles | [evidence/README.md](evidence/README.md) |
| Worktree lifecycle, leases, retirement | [../dev/worktree_lifecycle.md](../dev/worktree_lifecycle.md) |
| Worktree teardown and artifact preservation rules | [AGENTS.md](../../AGENTS.md) "Worktree Teardown And Artifacts" |
| Large result-tree manifests and verification | [chunk_manifest.py](../../scripts/tools/chunk_manifest.py) |
| Durable locality and failure-domain audit | [check_durable_artifact_locality.py](../../scripts/validation/check_durable_artifact_locality.py) |
| Read-only reclaim inventory | [check_worktree_capacity.py](../../scripts/dev/check_worktree_capacity.py) |
| Read-only worktree hygiene snapshot | [worktree_hygiene_snapshot.py](../../scripts/dev/worktree_hygiene_snapshot.py) |
| Preservation-aware retirement | [stale_worktree_reaper.py](../../scripts/dev/stale_worktree_reaper.py) |
| Active-worktree lease | [pr_gate_lease.py](../../scripts/dev/pr_gate_lease.py) |
| Source-host prune eligibility guard | [check_prune_eligibility.py](../../scripts/tools/check_prune_eligibility.py) |
| Environment and artifact restore verifier | [verify_restored_environment.py](../../scripts/tools/verify_restored_environment.py) |
| Post-access execution and artifact handoff | [generate_post_access_handoff.py](../../scripts/tools/generate_post_access_handoff.py) |

## 1. Retention classes in operational terms

These classes come from [artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md); the
operational meaning and cleanup default are restated here without changing the vocabulary.

| Class | Operational meaning | Cleanup default |
| --- | --- | --- |
| Local scratch | Test, smoke, demo, coverage, or temporary-conversion output produced during local work. | Disposable when the owning task is finished and no preserved artifact depends on it. |
| Exploratory output | Early run output used to inspect behavior or shape a hypothesis. | Disposable after its conclusion is recorded; it is not claim evidence. |
| Task-owned worktree output | Files under a worktree's `output/` that were produced by the task that owns the worktree. | Disposable after preservation review; never a durable dependency. |
| Tracked fixture | Small committed source-contract file used by tests or examples. | Never deleted as cleanup; remove only through a reviewed change. |
| Durable evidence copy | Small reviewable evidence promoted from generated output for later comparison. | Protected; keep the exact tracked path and its context note. |
| Release artifact | Immutable release bundle, archive, DOI, W&B artifact, or other publication target. | Protected; reference by URL or artifact URI plus checksum and version. |
| External artifact pointer | Data, runtime, or assets controlled outside this repository. | Protected as a pointer; do not delete the documented hydration target without a replacement. |
| Benchmark claim | Statement that a planner, config, metric, or scenario satisfies a benchmark contract. | Protected once the claim exists; `output/` alone never satisfies it. |
| Paper-facing claim | Statement intended to support manuscript, dissertation, release, or camera-ready language. | Protected; treat the frozen contract and its durable artifact as irreversible without a new ruling. |
| Legacy path | Pre-canonical repository path listed by the artifact-root migration map. | Migrate or verify with `get_legacy_migration_plan()` before removal; do not guess. |

`output/` is the git-ignored, worktree-local artifact root. It is not durable storage by itself.

## 2. Copies are not custody

An artifact moves through six distinct states. Only the later states count as preservation.

| State | What it is | What it does not prove |
| --- | --- | --- |
| Source output | The producer wrote a file under a local `output/` tree. | Nothing durable; the worktree or machine may disappear. |
| Retrieved copy | Someone copied the output to another local or session path. | Still local; no durability, no independent failure domain. |
| Verified durable copy | A copy at a durable location with a recorded command, commit, checksum, and hydration path. | Only that this copy is intact, not that a second independent copy exists. |
| Independent second copy | A second copy in a different failure domain (different host, account, or storage backend). | Only resilience to one failure domain; it is not a scientific validation. |
| Public release asset | An immutable published bundle, archive, DOI, W&B artifact, or release target. | Only what its declared contract and checksum state. |
| Compact public evidence | A small tracked pointer, manifest, or evidence copy with caveats and provenance. | Only the recorded scope; it is not the full raw data. |

Three common mistakes:

- **Scheduler completion is not preservation.** A terminal scheduler receipt or exit code 0 records
  execution, not artifact custody. Verify the produced rows, manifests, and checksums separately.
- **W&B run visibility is not custody.** A visible run or W&B artifact listing does not by itself
  bind a checksum, hydration command, retention decision, or license.
- **A checksum file alone is not an independent copy.** A checksum proves integrity of one copy; two
  copies in the same failure domain are still a single point of loss.

## 3. Gates before "preserved" or "cleanup-eligible"

State each gate explicitly before using either word.

| Gate | Question | Canonical check |
| --- | --- | --- |
| Active writer / lease | Does an active task still own the branch, worktree, or artifact? | `pr_gate_lease.py status`; `docs/dev/worktree_lifecycle.md` |
| Consumer / dependency | Does a benchmark, report, manuscript, or launcher still consume it? | Search the manifest, registry, or context note that references it. |
| Supersession | Is there an exact, repository-resolvable replacement pointer? | The replacement path or artifact URI must resolve; remove ambiguity first. |
| Failure domain | Are the surviving copies in genuinely different failure domains? | `check_durable_artifact_locality.py --check` reports `same_failure_domain` and `insufficient_redundancy`. |
| Rights / license | Are redistribution and retention permitted? | Asset-rights inventory and the evidence bundle policy. |
| Restore test | Can the artifact be hydrated and re-checked from the durable copy? | Hydrate into a scratch path and run the owning verification command. |
| Retention role | Is the intended lifecycle recorded? | `chunk_manifest.py manifest --retention-role {keep-latest,long-lived,short-lived,disposable,unspecified}` |

If a gate cannot be answered from current owners, stop and record the blocker instead of inferring a
policy.

## 4. Checked workflows

### 4.1 Inventory without deleting

```bash
uv run python scripts/dev/check_worktree_capacity.py --inventory --json
uv run python scripts/dev/worktree_hygiene_snapshot.py --repo-status --retirement-plan --json
```

Both helpers are read-only and never delete files.

### 4.2 Verify a large result tree

```bash
uv run python scripts/tools/chunk_manifest.py manifest --root <RESULT_TREE> --output <MANIFEST.json>
uv run python scripts/tools/chunk_manifest.py verify --root <RESULT_TREE> --manifest <MANIFEST.json> --json
```

`verify` fails closed with exact file or chunk locations on mutation, truncation, sparse/symlink/
hardlink/special-file, path, collision, and partial-manifest conditions.

### 4.3 Preserve

1. Produce or locate the result tree.
2. Verify it with `chunk_manifest.py` (or the owning schema check for small artifacts).
3. Promote a copy to a durable location (tracked evidence path, release asset, or documented
   external pointer) with command, commit, checksum, scope, and caveats.
4. Link the durable copy from the owning context note or evidence README.
5. Record whether an independent second copy exists and in which failure domain.

### 4.4 Restore-test

Hydrate the artifact from the durable copy into a scratch path and rerun the owning verification
command. A successful restore test is required before claiming preservation or cleanup eligibility.

```bash
uv run python scripts/tools/verify_restored_environment.py --check \
  --manifest <TRANSFERRED_MANIFEST> --root "$SCRATCH_ROOT" --format json
```

The verifier reconstructs the declared environment in a clean temporary root without source-host
dependencies, validates all checksums, row/config/checkpoint identities, and executes safe
read-only smoke assertions. The outcome is labelled restoration smoke, not scientific reproduction.

### 4.5 Check cleanup eligibility

```bash
uv run python scripts/dev/stale_worktree_reaper.py --path "$WORKTREE_PATH" --json
```

The reaper is a dry run by default. Removal additionally requires the verified merged identity:

```bash
uv run python scripts/dev/pr_gate_lease.py release --worktree "$WORKTREE_PATH"
uv run python scripts/dev/stale_worktree_reaper.py \
  --path "$WORKTREE_PATH" \
  --verified-merged-pr <PR_NUMBER> \
  --verified-branch <BRANCH> \
  --verified-head-sha <40-CHAR-HEAD-SHA> \
  --apply --json
git worktree prune
```

Any deletion, symlink/path alias, identity drift, new content, lease, lookup error, or lock failure
refuses removal. Normal worktree removal preserves the local branch and its commits; artifact
preservation remains the owner's responsibility before retirement.

### 4.6 Audit durable locality and failure domains

```bash
uv run python scripts/validation/check_durable_artifact_locality.py \
  --projection tests/validation/fixtures/durable_artifact_locality/compliant.json --check
```

The audit is fail-closed: it joins a sanitized locality packet to its locator-class artifacts
projection by artifact ID, version, and digest, and exits non-zero when an active durable-required
reference has no verified non-institutional custody or a release-facing reference lacks independent
failure-domain copies. It reads sanitized inputs only and never emits locator values.

### 4.7 Gate source-host artifact pruning on verified custody

```bash
uv run python scripts/tools/check_prune_eligibility.py --check \
  --source-manifest <SOURCE_MANIFEST> --destination-receipt <DESTINATION_RECEIPT> \
  --format json
```

The guard verifies durable destination custody, checksums, consumer coverage, and retention
dispositions before permitting deletion planning. Check mode performs zero file deletions;
an explicit `--apply` route enforces compare-and-swap revalidation before removing eligible bytes.

### 4.8 Generate complete post-access handoff

```bash
uv run python scripts/tools/generate_post_access_handoff.py --check \
  --inventory <COMPUTE_INVENTORY_JSON> --format json
```

The generator produces a deterministic, sanitized post-access handoff report in JSON or Markdown
summarizing workloads, scheduler receipts, artifact custody, and environment recreation states.
It redacts private paths, internal hosts, and credentials, rejects contradictory statuses and
orphan records, and enforces actionable next commands for incomplete runs.

### 4.9 Report a blocker

When two current owners disagree, when a cleanup command is not stable, or when a lifecycle state is
missing, stop and open a bounded issue describing the exact conflict. Do not invent a lifecycle
state, retention class, or destructive procedure.

## 5. Failure examples

- **Scheduler-complete but artifact-incomplete.** A job receipt says success, but the episode
  records or manifest rows are missing. Classify the artifact, not the scheduler: the run is not
  preserved until the produced rows verify.
- **W&B run without artifact custody.** The run page is visible, but no artifact URI, checksum,
  hydration command, or retention decision exists. Treat it as execution evidence only.
- **Two paths in one failure domain.** A copy under `output/` and a second copy on the same host or
  account are one loss event. Record the failure domain before calling the artifact independent.
- **Safely disposable task-owned output.** A smoke run writes only disposable scratch files, no
  benchmark or report depends on them, the owning task has finished, and the lease is releasable.
  This output may be cleaned without a preservation record.

## 6. Where to go next

- Evidence category rules: [artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md)
- Worktree lifecycle and retirement: [../dev/worktree_lifecycle.md](../dev/worktree_lifecycle.md)
- Durable artifact rules: [AGENTS.md](../../AGENTS.md) "Worktree Teardown And Artifacts"
- Tracked evidence policy: [evidence/README.md](evidence/README.md)
