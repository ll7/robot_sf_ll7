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
