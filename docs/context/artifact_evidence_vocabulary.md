# Artifact Evidence Vocabulary

**Status**: Canonical cross-issue vocabulary for issue bodies, PR validation notes, and benchmark
handoffs.

**Motivating issue**: [#1257](https://github.com/ll7/robot_sf_ll7/issues/1257)

**Canonical policy sources**:

- [AGENTS.md](../../AGENTS.md) durable artifact and proof-first validation rules.
- [docs/dev_guide.md](../dev_guide.md) artifact-root and PR-readiness guidance.
- [docs/context/evidence/README.md](evidence/README.md) tracked evidence-bundle policy.
- [Issue #691 Benchmark Fallback Policy](issue_691_benchmark_fallback_policy.md).
- [Issue #1062 Paper Evidence Archive Pointer](issue_1062_paper_evidence_archive.md).

## Purpose

Use this vocabulary when an issue, PR, benchmark report, or agent handoff names the expected
evidence category. The goal is to prevent local-only files from being promoted into durable,
benchmark, or paper-facing proof without the storage, hashes, and replay path needed to verify them
later.

`output/` is the git-ignored worktree artifact root. It is useful for local runs, smoke checks,
coverage, temporary exports, videos, and caches, but it is not a durable dependency by itself.

For operational retention classes, preservation proof, and cleanup-eligibility workflows, see the
[Artifact Retention, Preservation, and Cleanup Guide](artifact_retention_and_cleanup.md).
For the check-only guard that decides when one artifact or output identity may be deleted, see the
[Cleanup Eligibility Guard](cleanup_eligibility.md).
For the complete post-access restore and local-analysis sequence, see the
[post-access restoration and local-analysis runbook](../post_access_local_analysis_runbook.md).

Tool-specific contracts live in their own `docs/context/<tool>.md` note and are linked from a stable
location in this file (this section), not as a new top-level section per tool. Keeping the shared
vocabulary free of appended per-tool sections avoids parallel-merge conflicts and CI restarts when
several tooling PRs land together; the note-maintenance convention is documented in the
[Context Notes Workflow](README.md#per-tool-contract-notes).

## Vocabulary

| Category | Meaning | May cite `output/`? | Acceptable reference |
| --- | --- | --- | --- |
| Exploratory output | Early run output used to inspect behavior or shape a hypothesis. | Yes, with caveat. | Local path plus command, seed, commit, and "exploratory only" label. |
| Local scratch artifact | Disposable file produced by tests, smoke runs, demos, coverage, or temporary conversions. | Yes. | Local path in validation notes, explicitly marked disposable. |
| Tracked fixture | Small committed source-contract file used by tests or examples. | No, except as the generation source. | Repository path under `tests/`, `configs/`, `docs/context/evidence/`, `maps/`, or similar. |
| Durable evidence copy | Small reviewable evidence promoted from generated output for future comparison. | No, not as the durable location. | `docs/context/evidence/...` path with command, commit, checksum, and scope. |
| Release artifact | Immutable release bundle, archive, DOI, W&B artifact, or other durable publication target. | No, except local cache/hydration target. | Release URL or artifact URI plus checksum, version/tag, and hydration command. |
| External artifact pointer | Pointer to data/runtime/assets controlled outside this repository. | No, except local cache/hydration target. | Upstream URL, version, license/access note, expected checksum when available, and fail-closed behavior. |
| Benchmark claim | Statement that a planner, config, metric, or scenario satisfies a benchmark contract. | No. | Schema-checked episode records, summary/report files, provenance metadata, and reproducible command/config. |
| Paper-facing claim | Statement intended to support manuscript, dissertation, release, or camera-ready language. | No. | Frozen benchmark contract, release artifact or durable evidence copy, checksums, and explicit caveats. |

## Rules

- `output/` paths can support exploratory output and local scratch artifacts only.
- `output/` paths may appear in commands that regenerate or hydrate artifacts, but not as the sole
  evidence for durable, benchmark, or paper-facing claims.
- Benchmark and paper-facing claims must identify the contract being claimed, the exact command or
  config, the source commit, and the durable artifact or evidence path.
- Fallback, degraded, adapter, and not-available execution modes must be named explicitly. Fallback
  execution is a caveat or exclusion reason, not claim-grade benchmark success.
- External artifact pointers must include the unblock condition. If the asset/runtime cannot be
  hydrated, the dependent benchmark or planner path should fail closed with an actionable message.
- When evidence is expensive or too large to commit, track a manifest or pointer instead of copying
  raw episodes, videos, checkpoints, model caches, or logs into git.
- Substantial agent-produced evidence should include an
  [Agent Run Manifest](../agent_run_manifest.md) (`agent_run_manifest.yaml`) in the evidence bundle
  so the run that produced the evidence is auditable. Start from
  [`docs/templates/agent_run_manifest.yaml`](../templates/agent_run_manifest.yaml).

## Durable Artifact Locality Audit

`scripts/validation/check_durable_artifact_locality.py` joins the public `references`
inventory in a sanitized locality packet (`durable_artifact_locator_projection.v1`)
to its locator-class `artifacts` projection by artifact ID, version, and digest.
`--check` exits non-zero when an active durable-required reference has no verified
non-institutional locator, or a release-facing reference lacks its configured
independent failure-domain copies. It reads sanitized inputs only and never emits
locator values; historical inactive references stay recorded with outcome `inactive`
and never satisfy an active custody requirement.

Locator classes: `public_release`, `cloud_durable`, `personal_durable`,
`institutional_durable`, `institutional_cache`, `local_scratch`, `unknown`,
`unavailable`. Only the first three count as non-institutional custody. Stable
reason codes include `missing_projection_row`, `version_mismatch`, `digest_mismatch`,
`stale_verification`, `mutable_alias`, `institutional_only`, `cache_only`,
`non_durable_custody`, `no_verified_locator`, `same_failure_domain`, and `insufficient_redundancy`.

Validate with `uv run python scripts/validation/check_durable_artifact_locality.py --projection tests/validation/fixtures/durable_artifact_locality/compliant.json --check`.

## Chunk Manifests for Large Result Trees

[`scripts/tools/chunk_manifest.py`](../../scripts/tools/chunk_manifest.py) writes a
`chunk_manifest.v1` record for result trees too large to re-hash in one transfer window:
normalized relative paths, full-file digests for small members, fixed-boundary chunk digests for
large members, an order/worker-invariant `tree_sha256`, and a `manifest_id` semantic digest that
preservation and transfer receipts reference without rewriting producer manifests. `verify` fails
closed with exact file/chunk locations on mutation, truncation, sparse/symlink/hardlink/special
file, path, collision, and partial-manifest conditions.

## Compute-Window Readiness Dashboard

[`scripts/tools/compute_window_readiness_dashboard.py`](../../scripts/tools/compute_window_readiness_dashboard.py)
renders one deterministic JSON plus Markdown dashboard from versioned canonical input reports
(`robot_sf.compute_window_dashboard_input.v1`; sanitized fixtures under
`tests/tools/fixtures/compute_window_dashboard/`). Rows show campaign identity, owner,
priority/tier, resource class, prerequisite, source/config/checkpoint status, job state,
expected/observed rows, harvest/preservation/environment/restore state, copies, deadline fit,
and next owner, with implementation/compute/scheduler/artifact/evidence/review/claim kept
separate. No scientific score or admission decision is computed; stale, missing, contradictory,
duplicate, wrong-schema, or unsanitized input masks affected rows as explicit `unavailable`, and
output carries no private paths, hostnames, accounts, credentials, or signed URLs.

## Compute Staging Bundles

[`scripts/validation/build_compute_staging_bundle.py`](../../scripts/validation/build_compute_staging_bundle.py) binds one authorized workload's source/config/seed/checkpoint/lock identities into a deterministic `compute_staging_bundle.v1` receipt plus `SHA256SUMS`/inventory/transfer instructions; `--help` lists the stable fail-closed reason codes.

## Terminal-Job Harvest Receipts

[`scripts/validation/harvest_terminal_job.py`](../../scripts/validation/harvest_terminal_job.py) consumes one explicit `terminal_job_harvest_request.v1` plus a local artifact root and writes a deterministic `terminal_job_harvest.v1` public receipt, a private detailed receipt, and `SHA256SUMS`. Scheduler state and artifact completeness stay separate; every expected row gets one explicit disposition (present/duplicate/corrupt/failed/unavailable/missing) and execution mode (native/adapter/fallback/degraded); missing identity/contract/capacity, checksum, membership, stale source, and destination-verification conflicts fail closed. Validate with `uv run python scripts/validation/harvest_terminal_job.py --check --fixture <fixture-root> --format json`.

## Artifact Transfer Custody

[`scripts/validation/verify_artifact_transfer.py`](../../scripts/validation/verify_artifact_transfer.py)
consumes one existing `terminal_job_harvest.v1` or `compute_staging_bundle.v1` receipt and copies
only its manifest-declared members from an explicit source root to an explicit destination root
(local/fixture copy only). Every destination member is re-hashed before any copy, so verified
members are reported `already_verified` and never re-copied, interrupted `.transfer-partial` files
are cleaned and resumed, and conflicting bytes fail closed without being overwritten. `--apply`
writes a deterministic `artifact_transfer_custody.v1` receipt under the destination root with
per-file states, byte counts, capacity, and independently re-hashed destination bytes; `--check`
is read-only. Receipts carry normalized relative paths only, and live SSH/private-host transfer
stays routed through private operations: `uv run python
scripts/validation/verify_artifact_transfer.py --check --manifest <receipt> --source-root <root>
--destination-root <root> --format json`.

## Checkpoint Compatibility Audit

[`scripts/models/audit_checkpoint_compatibility.py`](../../scripts/models/audit_checkpoint_compatibility.py)
audits a sanitized overlay, canonical `--registry`/`--config` intake, or both into a deterministic
JSON plus Markdown inventory with nine terminal states and stable reason codes. An opt-in `--probe`
runs a bounded-subprocess loader check (hard timeout, no hidden fallback); `--check` exits 1 when an
active consumer's required model is not recoverable and load-verified and 2 for unknown input (see
the module docstring).

## Sanitized Lineage Index

[`scripts/tools/lineage_index.py`](../../scripts/tools/lineage_index.py) joins compact sanitized
records (`sanitized_lineage_input.v1`) into one deterministic `robot_sf.lineage_index.v1` JSON
plus Markdown index keyed by stable semantic identity (`kind:id`), never filename proximity or
timestamps. Records are `{"kind", "id", "refs": {"<kind>_ids": [...]}, ...scalars}` over
`issue`, `pull_request`, `commit`, `campaign`, `config`, `manifest`, `job`, `checkpoint`,
`model`, `environment`, `artifact`, `analysis`, and `claim`, with scalars `digest`, `owner`,
`attempt_index`, `relation`, `submission_receipt`, `predecessor_job_id`, `artifact_kind`,
`locator_class`, `claim_state`, and `not_applicable`; optional `private_projection` lists
withheld locators as `{"target", "digest", "withheld": true}` only. Each row is rooted at one
job attempt, or at an unreachable record, so retries and resumed shards stay separate rows
linked by `predecessor_job_id`, `attempt_index`, and `relation`; missing links classify as
`not_applicable`, `not_recorded`, `private_unavailable`, `conflict`, or `dangling`, and
duplicate IDs, contradictory digests, orphaned artifact pointers, projection drift, or absent
references fail closed. Query with `lineage_index.py query --input <path>` and
`--issue/--job/--campaign/--artifact-digest/--commit/--config` (exit 0 match, 2 otherwise).
Validate with `uv run python scripts/tools/lineage_index.py --input
tests/tools/fixtures/lineage_index/complete.json --check --format json`.

## Expiring-Resource Deadline Feasibility

[`scripts/validation/check_expiring_resource_feasibility.py`](../../scripts/validation/check_expiring_resource_feasibility.py)
evaluates the optional `expiring_resource` block of a campaign manifest (`expiring_resource_contract.v1`)
and returns one deterministic verdict: `fits_conservative`, `fits_expected`, `too_late`, or
`unknown`. It budgets expected/conservative runtime plus retrieval, verification, and preservation
reserves into a latest safe submission time, never guesses a scheduler start, and requires non-zero
retrieval/preservation reserves for durable-required outputs. Manifests without the block stay
non-applicable and non-blocking. See
[Expiring-Resource Deadline Feasibility](expiring_resource_deadlines.md); the case pack lives under
`tests/validation/fixtures/expiring_resource_feasibility/`.

## Learned-Policy Artifact Manifests

Learned local-policy checkpoints, normalizers, imitation datasets, and residual-policy artifacts
should specialize this vocabulary instead of creating a parallel evidence system. A learned-policy
artifact manifest is a compact pointer record. It is not the checkpoint, not a model registry, and
not benchmark evidence by itself.

Required fields:

| Field | Meaning |
| --- | --- |
| `policy_id` | Stable policy or component id, such as `learned_risk_model_v1`. |
| `artifact_role` | One of `checkpoint`, `normalizer`, `dataset_manifest`, `adapter_config`, or `launch_packet`. |
| `artifact_uri` | Durable URI, release URL, or tracked config/evidence path. Local `output/` paths are allowed only as regeneration or hydration targets, not as durable URIs. |
| `sha256` | Checksum for tracked fixtures, release artifacts, or local files promoted to durable storage. Use `pending` only before benchmark eligibility. |
| `training_config` | Repository path to the config or launch packet that produced or will produce the artifact. |
| `training_commit` | Git commit for the training, data-generation, or launch-packet contract. |
| `observation_schema` | Observation contract path or named schema, including `observation_t` and deployment-visible fields. |
| `action_schema` | Action-output family, bounds, frame, projection, and guard/fallback behavior. |
| `normalizer_uri` | Durable normalizer artifact URI or `not_required`; learned normalizers must state the fit split. |
| `license` | License or access note for the artifact and source data. |
| `split_contract` | Train/validation/test split contract or note. |
| `benchmark_eligibility` | One of `not_eligible`, `research_only`, `adapter_preflight`, or `benchmark_candidate`. |
| `fail_closed_behavior` | Action when the artifact, checksum, normalizer, or schema is missing or mismatched. |

Example manifest shape for the existing learned-risk launch lane:

```yaml
policy_id: learned_risk_model_v1
artifact_role: launch_packet
artifact_uri: configs/training/learned_risk_model_issue_1395_launch_packet.yaml
sha256: pending
training_config: configs/training/learned_risk_model_issue_1395_launch_packet.yaml
training_commit: e14e2f8bc2058d9f0e071219629915dd5b5dd5a8
observation_schema:
  contract: docs/context/policy_search/contracts/learned_local_policy_eligibility.md
  observation_t: current decision step
  deployment_fields:
    - trajectory_features.min_rollout_clearance_m
    - trajectory_features.mean_pedestrian_distance_m
    - trajectory_features.route_progress_delta
action_schema:
  family: auxiliary_cost
  role: rank otherwise-safe local commands only
  hard_guards_authoritative: true
normalizer_uri: not_required_for_launch_packet
license: repository-internal pre-SLURM launch packet; no checkpoint distributed
split_contract: docs/context/open_issues_training_split_audit_2026-05-30.md
benchmark_eligibility: adapter_preflight
fail_closed_behavior:
  missing_artifact: reject learned-policy benchmark row
  checksum_mismatch: reject learned-policy benchmark row
  missing_observation_or_action_schema: classify as not_eligible
  missing_normalizer: reject if the policy declares learned normalization
```

Benchmark-facing learned-policy claims must resolve any `pending` checksum or artifact URI first.
If a checkpoint, normalizer, dataset, or schema cannot be hydrated from the manifest, the dependent
adapter or benchmark row must fail closed with `not_available` or `failed` status. Do not silently
fall back to a non-learned planner and report that row as learned-policy success.

## Acceptable References

- Exploratory run:
  `output/benchmarks/h500_probe/summary.json`, generated by a named command at commit `abc123`,
  marked "exploratory; not benchmark-claim evidence."
- Local scratch validation:
  `output/tmp/parquet_export_smoke/parquet/`, generated during PR validation and classified as
  disposable.
- Tracked fixture:
  `tests/data/snqi/episodes_small.jsonl`, committed because tests need a tiny stable input.
- Durable evidence copy:
  `docs/context/evidence/issue_1023_scenario_horizons_preflight_2026-05-06/summary.json`, linked
  from the context note that explains the command, scope, and interpretation.
- Release artifact:
  `robot-sf-benchmark-release-0.0.2.tar.gz` from a GitHub release or DOI-backed archive, with the
  published SHA-256 and hydration command.
- External artifact pointer:
  CARLA `0.9.16` Docker image or SocNavBench/SDD asset pointer with license/access notes and the
  exact local cache path used only after hydration.

## Unacceptable References

- "Paper table generated from `output/benchmarks/latest/summary.json`" with no commit, command,
  checksum, or durable artifact.
- "Benchmark passed using fallback mode" without labeling the fallback/degraded status as a caveat.
- "Dataset exists under `output/SocNavBench`" without upstream source, license/access decision, or
  hydration instructions.
- "Planner is release-ready" based only on a local smoke output, with no schema-checked episodes or
  reproducible benchmark contract.

## Current Issue-Lane Examples

- [Issue #1243 Experiment Registry](https://github.com/ll7/robot_sf_ll7/issues/1243):
  registry entries may record exploratory output and local scratch paths, but any registry entry
  used to justify a benchmark or paper-facing claim must reference durable evidence, a release
  artifact, or an external artifact pointer.
- [Issue #1245 BenchmarkClaim Artifacts](https://github.com/ll7/robot_sf_ll7/issues/1245):
  claim payloads should use benchmark-claim and paper-facing-claim categories and reject local-only
  `output/` paths as sufficient evidence.
- [Issue #1231 Paper Handoff Fixture](https://github.com/ll7/robot_sf_ll7/issues/1231):
  the release archive/checksum is the release artifact; the hydrated `output/...` extraction is a
  local cache for tests, not the durable source.
- [Issue #1108 BC Warm-Start PPO Execution](https://github.com/ll7/robot_sf_ll7/issues/1108):
  SLURM logs, W&B run folders, and local `output/` paths are execution-run or exploratory evidence
  until a manifest, model registry entry, release artifact, or tracked evidence copy is published.
- [Issue #1686 Learned-Policy Artifact Manifests](https://github.com/ll7/robot_sf_ll7/issues/1686):
  `docs/context/policy_search/contracts/learned_local_policy_eligibility.md` defines the
  observation/action review contract, and
  `docs/context/open_issues_training_split_audit_2026-05-30.md` records the current training-lane
  split/provenance pointers that manifests should reference.
- [Issue #2923 Mechanism Trace v1 Schema](https://github.com/ll7/robot_sf_ll7/issues/2923):
  `tests/benchmark/fixtures/mechanism_trace.v1.example.json` is a tracked fixture for schema
  validation, while real mechanism reports still need durable trace inputs before they can support
  benchmark or paper-facing claims.
- CARLA runtime qualification issues
  ([#872](https://github.com/ll7/robot_sf_ll7/issues/872),
  [#1111](https://github.com/ll7/robot_sf_ll7/issues/1111),
  [#1169](https://github.com/ll7/robot_sf_ll7/issues/1169),
  [#1179](https://github.com/ll7/robot_sf_ll7/issues/1179)):
  Docker images, CARLA runtimes, and bridge assets are external artifact pointers until pinned,
  hydrated, and smoke-tested with fail-closed behavior.
- External data and map-conversion issues
  ([#1126](https://github.com/ll7/robot_sf_ll7/issues/1126),
  [#1134](https://github.com/ll7/robot_sf_ll7/issues/1134)):
  source datasets and converted assets need external artifact pointers or tracked fixtures before
  downstream benchmark claims can depend on them.

## Deferred Automation

The first slice is a human-readable vocabulary. Machine-readable enums, BenchmarkClaim validation
against local-only paths, and issue-form enforcement should be implemented in dedicated follow-up
issues rather than hidden inside this documentation change.

## Validation

For changes to this vocabulary:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

Spot-check the issue lanes above before changing the category names or the `output/` policy.
