# Issue #1627 Learned-Policy Transfer Benchmark Design - 2026-05-30

Date: 2026-05-30

Related issue:

- Issue #1627: <https://github.com/ll7/robot_sf_ll7/issues/1627>

Related anchors:

- Issue #691 benchmark fallback policy:
  `docs/context/issue_691_benchmark_fallback_policy.md`
- Issue #1618 learned local-policy adapter interface:
  `docs/context/issue_1618_learned_policy_adapter_interface.md`
- Issue #1619 learned-policy registry/source-claim work:
  <https://github.com/ll7/robot_sf_ll7/issues/1619>
- Issue #1620 external learned-policy ranking:
  <https://github.com/ll7/robot_sf_ll7/issues/1620>
- Learned local-policy eligibility checklist:
  `docs/context/policy_search/contracts/learned_local_policy_eligibility.md`
- Learned-policy registry:
  `docs/context/policy_search/learned_policy_registry.md`
- Episode schema:
  `robot_sf/benchmark/schemas/episode.schema.v1.json`

## Issue Requirements

Issue #1627 asks for a design note, not a runnable benchmark. The acceptance surface is:

- define transfer benchmark criteria for externally trained learned policies,
- specify source, artifact, observation/action, adapter, dependency, hardware, and missing-asset
  metadata,
- define integration-effort, source-reproducibility, Robot SF smoke, performance, and benchmark
  status outputs,
- list fail-closed outcomes for missing checkpoints, incompatible observations, unsupported
  dependencies, fallback, or degraded execution,
- link the design to Issues #1618, #1619, #1620, and the benchmark fallback policy,
- check compatibility with existing benchmark result schemas and at least one existing
  learned-policy candidate,
- propose a follow-up implementation issue only if the design is concrete enough.

## Goal

Define a benchmark surface for learned policies trained or released outside the current Robot SF
benchmark run, especially external repository/paper policies imported through source harnesses or
adapters. The benchmark answers a narrow question:

> What proof is required before an imported learned policy can appear in Robot SF comparisons, and
> what status should be reported when it cannot satisfy the contract?

This is a design note only. It does not run a benchmark, integrate a policy, stage external assets,
or turn source-side success into Robot SF benchmark evidence.

## Benchmark Stages

The transfer benchmark is a gated pipeline. A candidate may stop at any stage with a fail-closed
status, and later stages must not be inferred from earlier ones.

| Stage | Purpose | Required output | Success-capable status |
| --- | --- | --- | --- |
| `source_intake` | Verify source URL, license, paper, commit/release, dependency surface, and artifact pointers. | Intake metadata and source/provenance status. | No, metadata only. |
| `source_reproduction` | Run the smallest upstream command or explicitly record a blocked source harness. | Source-side run report or fail-closed blocker. | No, source success is not Robot SF success. |
| `adapter_metadata` | Map source observation/action/checkpoint contract to Issue #1618 and the eligibility checklist. | Structured learned-policy metadata, including forbidden fields and fallback policy. | No, metadata only. |
| `robot_sf_smoke` | Run one bounded Robot SF adapter smoke with deterministic inference and per-step diagnostics. | Episode JSONL plus adapter diagnostics for raw/adapted/post-guard actions. | Yes, only for smoke evidence. |
| `transfer_benchmark` | Run the declared Robot SF benchmark slice with the same status semantics as other planners. | Benchmark rows with availability and readiness fields. | Yes, only when every required contract is native or adapter-ready. |

The benchmark must report the highest completed stage and the first blocking stage. A source policy
that runs upstream but cannot satisfy Robot SF observation/action metadata remains
`source_reproduction` evidence, not a benchmark row.

## Required Metadata

Every candidate transfer record should include these fields. They can live in a YAML preflight spec,
adapter metadata payload, candidate registry entry, or future report row; Markdown should link to
the durable source of truth rather than duplicate large manifests.

| Group | Required fields |
| --- | --- |
| Candidate identity | `transfer_candidate_id`, `policy_family`, `source_kind`, `paper_or_source`, `upstream_repo_url`, `upstream_commit_or_release`, `source_license`, `local_assessment_note` |
| Artifact provenance | `checkpoint_uri`, `checkpoint_checksum`, `normalizer_uri`, `normalizer_checksum`, `artifact_license_or_access`, `artifact_manifest_status` |
| Runtime environment | `source_runtime`, `robot_sf_runtime`, `dependency_lock`, `hardware_requirements`, `device_policy`, `external_assets_required` |
| Training and split | `training_data_source`, `train_validation_test_split`, `robot_sf_overlap_assessment`, `normalization_fit_scope`, `privileged_training_inputs` |
| Observation contract | `observation_level`, `planner_observation_mode`, `observation_t`, `deployment_observable_fields`, `training_only_fields`, `forbidden_evaluation_fields`, `history_or_recurrent_state` |
| Action contract | `action_family`, `raw_action_shape`, `raw_action_frame`, `adapted_action_frame`, `action_bounds`, `kinematics_compatibility`, `projection_policy`, `guard_policy` |
| Adapter behavior | `adapter_path`, `deterministic_inference`, `batching_support`, `missing_checkpoint_policy`, `unsupported_observation_policy`, `unsupported_kinematics_policy`, `fallback_policy` |
| Diagnostics | `raw_model_action`, `adapted_action`, `post_guard_action`, `guard_applied`, `guard_or_fallback_reason`, `action_projection_metadata`, `adapter_status` |
| Benchmark routing | `benchmark_track`, `transfer_stage`, `execution_mode`, `readiness_status`, `availability_status`, `availability_reason`, `benchmark_success` |

The artifact fields should reuse `docs/context/artifact_evidence_vocabulary.md` for any concrete
checkpoint, normalizer, or dataset. Missing checksums or local-only machine paths make the candidate
`not_available` for benchmark use.

### Canonical Storage Boundary

The canonical transfer record should be a versioned `learned_policy_transfer_benchmark.v1` metadata
object. During design and validator work it may live as a YAML/JSON fixture or candidate preflight
spec. In runnable benchmark evidence, the same object should be serialized under
`algorithm_metadata.transfer_benchmark` so report rows carry the status proof that produced them.

`docs/context/policy_search/candidate_registry.yaml` remains the canonical list for implemented or
concrete runnable Robot SF candidates only. Source-only, blocked, monitor-only, or metadata-only
transfer candidates should stay in context notes, source-harness records, reject/monitor registries,
or transfer metadata fixtures until they have a concrete Robot SF config or adapter path plus a
validation command.

Large checkpoints, normalizers, datasets, and source tarballs remain artifact-registry concerns. The
transfer metadata should point to durable artifact evidence and checksums; it should not duplicate
large manifests or depend on worktree-local `output/` files.

## Status Semantics

The transfer benchmark reuses the canonical fallback-policy statuses rather than inventing a new
success taxonomy.

Decision rules:

- `availability_status=not_available`: a pre-run contract, source asset, dependency, checkpoint,
  normalizer, artifact registry entry, or required metadata field is absent or incompatible before a
  Robot SF episode can count.
- `availability_status=failed`: an adapter, schema validation, runtime dependency, model inference,
  or benchmark row was attempted and failed during execution or validation.
- `availability_status=partial-failure`: a multi-row or multi-job campaign has some valid rows and
  some failed rows. Do not use this for a single systematic contract violation.
- `execution_mode=unknown`: the source contract or adapter contract has not been declared enough to
  classify execution.
- `execution_mode=adapter`: a Robot SF adapter contract is declared or attempted, even if that
  adapter later fails validation.
- `execution_mode=mixed`: a fallback controller, guard, prior, or placeholder substitutes the final
  command that should have come from the learned policy.

| Condition | `execution_mode` | `readiness_status` | `availability_status` | `benchmark_success` |
| --- | --- | --- | --- | --- |
| Robot SF-native learned policy with complete local checkpoint and matching contract. | `native` | `native` | `available` | `true` only after smoke/benchmark proof. |
| External policy through a declared adapter with source, checkpoint, observation, and action proof. | `adapter` | `adapter` | `available` | `true` only after smoke/benchmark proof. |
| External source command runs but Robot SF adapter metadata is incomplete. | `unknown` | `degraded` | `not_available` | `false` |
| Checkpoint, normalizer, dependency, model registry entry, or source asset is missing. | `unknown` | `degraded` | `not_available` | `false` |
| Observation uses forbidden evaluation-time fields or mismatched observation level. | `adapter` or `unknown` | `degraded` | `not_available` | `false` |
| Action must be projected without declared frame/bounds or kinematics compatibility. | `adapter` | `degraded` | `failed` | `false` |
| Fallback controller, guard, prior, or placeholder produces the final command instead of the model. | `mixed` | `fallback` | `not_available` | `false` |
| Runtime exception, invalid tensor shape, invalid action, or schema validation failure. | `adapter` or `unknown` | `degraded` | `failed` | `false` |

`robot_sf_bench run` and any transfer benchmark launcher should eventually return non-zero for
`fallback`, `degraded`, `partial-failure`, `failed`, or `not_available`, matching Issue #691. That is
a runner/reporting integration requirement, not part of the first metadata-validator follow-up.

## Metrics And Outputs

Transfer outputs should separate effort/reproducibility evidence from performance. A blocked
source harness is still useful evidence, but it is not policy performance.

| Output area | Fields |
| --- | --- |
| Integration effort | `source_files_touched_count`, `adapter_files_touched_count`, `new_dependencies_count`, `manual_asset_steps_count`, `source_harness_commands_count`, `blocked_reason_count`, `integration_effort_tier` |
| Source reproducibility | `source_command`, `source_exit_status`, `source_runtime_status`, `source_checkpoint_status`, `source_dependency_status`, `source_result_artifact` |
| Robot SF smoke | `smoke_command`, `smoke_episode_count`, `smoke_exit_status`, `smoke_adapter_status`, `diagnostics_complete`, `raw_adapted_post_guard_logged` |
| Benchmark performance | Existing episode metrics such as `success`, `collision`, `min_distance`, `near_misses`, `time_to_goal`, `comfort_exposure`, and SNQI-style summaries when available. |
| Interpretation | `transfer_claim_level`, `evidence_boundary`, `non_claims`, `next_reopen_condition` |

Recommended `integration_effort_tier` values:

- `small`: metadata-only or local registry wiring; no external runtime.
- `medium`: source harness plus small adapter, no new system dependency.
- `large`: external runtime, custom adapter, checkpoint hydration, or dependency isolation.
- `not_currently_actionable`: missing source, license, checkpoint, or runnable command.

Recommended `transfer_claim_level` values:

- `source_metadata_only`
- `source_reproduced`
- `adapter_smoke_only`
- `robot_sf_benchmark_row`
- `not_available`

## Existing Schema Compatibility

`robot_sf/benchmark/schemas/episode.schema.v1.json` already allows additional properties and has an
`algorithm_metadata` object with status/config/observation and `planner_kinematics` fields. A first
implementation can therefore attach transfer metadata under:

```yaml
algorithm_metadata:
  transfer_benchmark:
    schema_version: learned_policy_transfer_benchmark.v1
    transfer_candidate_id: ...
    transfer_stage: ...
    source_reproducibility: ...
    adapter_contract: ...
    artifact_status: ...
    diagnostics_status: ...
```

The top-level episode fields `observation_mode`, `observation_level`, `benchmark_track`, and
`track_schema_version` should remain the aggregation fence. Transfer metadata explains the imported
policy boundary; it must not change the meaning of scenario metrics.

A future implementation may add a dedicated JSON Schema for the transfer metadata object, but that
should be a narrow follow-up after this design is reviewed.

## Candidate Fit Checks

### `ppo_issue791_best_v1`

This is not an external import, but it proves the metadata shape works for a Robot SF-native learned
policy:

- candidate: `configs/policy_search/candidates/ppo_issue791_best_v1.yaml`
- baseline config referenced by the candidate:
  `configs/baselines/ppo_15m_grid_socnav.yaml`
- checkpoint from the baseline config:
  `model_id: ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`
- observation mode from the baseline config: `dict`
- action contract from the baseline config: `unicycle`, `v_max: 2.0`, `omega_max: 1.0`
- fallback policy from the baseline config: `fallback_to_goal: false`
- transfer status: `execution_mode=native`, `readiness_status=native` only when the model registry
  artifact hydrates and the PPO adapter emits valid actions.
- claim boundary: benchmark-set performance only; the config explicitly warns against OOD
  transfer/generalization claims.

The follow-up metadata fixture should be metadata-only. Validator tests should not require hydrating
the PPO model artifact, running inference, or proving benchmark performance.

### CrowdNav HEIGHT

This is the current source-harness-first example for an external learned family:

- source note: `docs/context/policy_search/issue_1394_crowdnav_height_source_harness.md`
- source URL: <https://github.com/Shuijing725/CrowdNav_HEIGHT>
- source commit recorded: `65451bcdd1f3fbebaf6e96a0de73aaa56d74ca05`
- source status: blocked by missing legacy `gym`
- checkpoint status: missing local checkpoint; upstream README points to an external Google Drive
  folder for `237400.pt`
- transfer stage: `source_reproduction`
- benchmark status: `execution_mode=unknown`, `readiness_status=degraded`,
  `availability_status=not_available`, `benchmark_success=false`

The fields above are sufficient to represent a real imported-policy candidate without adding it to
the runnable candidate registry or pretending that source-harness failure is performance evidence.
The follow-up fixture should use CrowdNav HEIGHT only as a blocked source-first metadata example,
not as a benchmark-success or Robot SF adapter proof.

## Minimal Candidate Set

The first implementation should not start with every external learned-policy family. Use the
smallest set that exercises all status paths:

| Candidate | Why include | Expected status |
| --- | --- | --- |
| `ppo_issue791_best_v1` | Native learned-policy control row with model registry provenance and fail-closed fallback disabled. | `native` / `available` when artifact hydration succeeds. |
| `crowdnav_height_igat_family` | Existing source-harness-first external family with concrete source and checkpoint blockers. | `not_available`, source reproduction blocked. |
| One future adapter-ready external candidate | Only after source reproduction, checkpoint hydration, and Issue #1618 metadata are proven. | `adapter` / `available` only after smoke proof. |

Do not include Arena-Rosnav, NavDP/NoMaD, VLA/foundation policies, or Decision Transformer lanes in
the first runnable transfer benchmark unless their source-side notes produce a concrete checkpoint
and Robot SF observation/action contract.

## Follow-Up Implementation Scope

The design is concrete enough for one narrow implementation issue, opened as Issue #1761:

- add a minimal `learned_policy_transfer_benchmark.v1` metadata schema or validator for the
  `algorithm_metadata.transfer_benchmark` object,
- require the first validator pass to cover candidate identity, artifact status,
  observation/action summary, transfer stage, execution/readiness/availability statuses,
  availability reason, `benchmark_success`, and evidence pointers,
- add one metadata-only fixture for `ppo_issue791_best_v1`,
- add one blocked source-first metadata fixture for CrowdNav HEIGHT using the #1394 evidence,
- add validator tests showing fallback, degraded, and not-available metadata rows cannot set
  `benchmark_success=true`,
- do not run external training, hydrate external checkpoints, wire a new adapter path, modify
  `robot_sf/benchmark/planner_command_contract.py`, or update benchmark runner/report-writer
  behavior in that implementation issue.

Runner and report-writer enforcement should be a later follow-up after the metadata object and
fixtures are reviewed.

## Non-Claims

- Source-side reproduction is not Robot SF benchmark success.
- A source paper's reported metrics are not Robot SF metrics.
- A Robot SF adapter smoke is not a full transfer benchmark.
- Fallback, guard-dominated, placeholder, degraded, or missing-checkpoint execution is not a
  successful transferred-policy row.
- No candidate should enter `docs/context/policy_search/candidate_registry.yaml` without a concrete
  Robot SF config or adapter path and validation command.

## Validation

Design compatibility checks used:

```bash
rg -n "execution_mode|readiness_status|availability_status|benchmark_track|algorithm_metadata" \
  robot_sf/benchmark/schemas/episode.schema.v1.json \
  robot_sf/benchmark/algorithm_metadata.py \
  robot_sf/benchmark/planner_command_contract.py \
  docs/context/issue_691_benchmark_fallback_policy.md \
  docs/context/issue_1618_learned_policy_adapter_interface.md
rg -n "ppo_issue791_best_v1|fallback_to_goal|model_id|obs_mode|action_space" \
  configs/policy_search/candidates/ppo_issue791_best_v1.yaml \
  configs/baselines/ppo_15m_grid_socnav.yaml
rg -n "CrowdNav HEIGHT|source harness blocked|missing_local_checkpoint|ModuleNotFoundError" \
  docs/context/policy_search/issue_1394_crowdnav_height_source_harness.md
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
