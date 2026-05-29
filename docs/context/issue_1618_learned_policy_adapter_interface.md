# Issue #1618 Learned Local Policy Adapter Interface

Date: 2026-05-29

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1618>

## Scope

This note defines the adapter contract for learned local-navigation policies in Robot SF. It turns
the learned-policy eligibility checklist into a runtime boundary that future PPO, guarded PPO,
ORCA-residual, LiDAR-only, imitation, learned-risk, or external learned-policy work can implement
without coupling each policy directly to benchmark runners.

This is a design contract only. It does not train a policy, promote a checkpoint, add a new
benchmark row, or claim any learned policy is benchmark-ready.

## Existing Surfaces

- `robot_sf/baselines/interface.py` defines generic planner observation/action metadata and the
  `PlannerProtocol.step(...)` lifecycle used by benchmark adapters.
- `robot_sf/benchmark/algorithm_metadata.py` maps existing algorithms to observation and action
  contracts, including `ppo`, `guarded_ppo`, external learned wrappers, and policy-stack planners.
- `robot_sf/benchmark/planner_command_contract.py` validates planner metadata against requested
  observation mode, observation level, and robot kinematics before benchmark execution.
- `docs/context/policy_search/contracts/learned_local_policy_eligibility.md` is the intake
  checklist for leakage, observation, action, provenance, and diagnostics.
- `scripts/validation/check_learned_policy_eligibility.py` validates structured checklist metadata.
- `configs/policy_search/candidates/ppo_issue791_best_v1.yaml` and
  `configs/baselines/ppo_15m_grid_socnav.yaml` are the cleanest current concrete learned-policy
  mapping: candidate ID, model registry pointer, dict observation mode, deterministic inference,
  predictive-foresight dependency, unicycle action output, and `fallback_to_goal: false`.
- `configs/training/lidar/lidar_ppo_mlp_eligibility_issue_1615.yaml` is the cleanest current
  planned/no-checkpoint spec that declares runtime observation keys, action semantics, and
  fail-closed policy before training.
- `robot_sf/baselines/ppo.py` is the current concrete learned-policy adapter, but its legacy
  fallback-to-goal behavior must be disabled for benchmark-strength learned-policy evidence.
- `configs/training/orca_residual/orca_residual_bc_issue_1428.yaml` records useful residual-policy
  diagnostics and the `fallback_degraded_rows_count_as_success: false` boundary.

## Adapter Boundary

Use `LearnedLocalPolicyAdapter` as the conceptual interface for future implementations. A concrete
adapter may be a protocol, abstract base class, or wrapper around the existing planner protocol, but
it should expose these responsibilities:

| Responsibility | Required behavior |
| --- | --- |
| Declare schema | Return observation, action, checkpoint, device, determinism, batching, and diagnostics metadata before execution. |
| Load model | Resolve model registry IDs, checkpoint URIs, normalizers, and optional feature extractors without writing bulky artifacts to git. |
| Validate runtime request | Fail before benchmark episodes when observation level, observation keys, kinematics, or checkpoint provenance do not match the declared contract. |
| Reset | Reset recurrent state, history buffers, stochastic generators, and per-episode diagnostics from an explicit seed. |
| Infer | Convert one Robot SF observation at decision step `t` into a raw model action in deterministic mode by default. |
| Adapt action | Convert raw model output into a Robot SF command with explicit frame, units, bounds, clamp/projection order, and kinematics compatibility. |
| Guard or project | Apply safety guards or fallback controllers only as a reported post-policy step, never as hidden model success. |
| Report diagnostics | Emit raw/adapted/post-guard actions, guard status, projection metadata, fallback reason, and adapter status for every smoke or benchmark run. |
| Close | Release model handles, workers, files, and device memory. |

The adapter should remain callable from benchmark runners through the existing planner surface:
`reset(seed=...)`, `step(obs) -> dict[str, float]`, `close()`, and optional metadata methods. The
learned-policy contract adds stricter metadata and diagnostics, not a separate benchmark runner.

## Required Metadata

Every learned local-policy adapter or runnable candidate spec must record:

- `policy_id` and implementation path.
- Source paper/repository/model lineage, license boundary, and whether the adapter is source-backed
  or Robot SF-native.
- Training config, evaluation config, train/validation/test split assumptions, and normalization
  provenance.
- Checkpoint or artifact source, including durable URI or explicit `no checkpoint yet` status.
- Observation mode, benchmark observation level, exact observation keys, tensor shapes, frame,
  ordering, history length, and normalization.
- Deployment-observable, training-only, and forbidden evaluation-time fields.
- Action output family, frame, units, bounds, raw-to-adapted mapping, kinematics compatibility, and
  projection policy.
- Device handling and deterministic inference mode.
- Batching support, or an explicit `single_step_only` statement.
- Per-step diagnostic fields required by
  `docs/context/policy_search/contracts/learned_local_policy_eligibility.md`.
- Missing model, unsupported observation, unsupported kinematics, guard activation, degraded, and
  fallback reporting policy.
- Evidence status: source-only, metadata-eligible, adapter-ready, smoke-tested, or benchmark-ready.

These fields can live in a YAML eligibility spec, model registry entry, candidate registry entry, or
adapter metadata payload. Do not duplicate large checkpoint metadata in Markdown when a registry
entry or artifact manifest is the source of truth; link to it instead.

## Status Semantics

Adapters should use the canonical benchmark status fields where benchmark rows or smoke summaries
are emitted:

| Field | Allowed values | Learned-policy interpretation |
| --- | --- | --- |
| `execution_mode` | `native`, `adapter`, `mixed`, `unknown` | Whether the learned policy ran as Robot SF-native code, through a source-backed adapter, through mixed semantics, or with unresolved runtime semantics. |
| `readiness_status` | `native`, `adapter`, `fallback`, `degraded` | Whether the policy satisfied its intended contract, needed an adapter, only ran through fallback, or failed to satisfy the clean contract. |
| `availability_status` | `available`, `partial-failure`, `failed`, `not_available` | Whether the runtime contract was available for success-capable evidence, partially failed, failed, or was not available. |

Use these learned-policy status combinations as shorthand:

| Case | Status payload | Benchmark interpretation |
| --- | --- | --- |
| Robot SF-native policy ran with matching checkpoint, observation contract, and action contract. | `execution_mode=native`, `readiness_status=native`, `availability_status=available` | May be counted only with smoke or benchmark proof. |
| Source-backed policy ran through an explicit Robot SF adapter. | `execution_mode=adapter`, `readiness_status=adapter`, `availability_status=available` | May be counted only with source/provenance and runtime proof. |
| Required checkpoint, package, artifact, observation level, or metadata is missing before execution. | `availability_status=not_available` | Exclusion or blocker, not success. |
| Adapter ran with a weakened contract, substituted input, partial dependency, or unverified transform. | `readiness_status=degraded` or `availability_status=partial-failure` | Caveat only, not benchmark-strength evidence. |
| A guard, prior, or fallback controller produced the final command instead of the learned policy. | `readiness_status=fallback` | Caveat only; report separately from model success. |
| Runtime error, invalid output, unsupported shape, or validation failure. | `availability_status=failed` | Failure evidence with actionable reason. |

This follows `docs/context/issue_691_benchmark_fallback_policy.md`: fallback and degraded execution
must not be classified as successful benchmark evidence unless the task explicitly measures that
mode.

## Example Mapping: PPO Issue-791 Leader

The current concrete learned-policy example is the issue-791 PPO leader:

| Contract field | Current mapping |
| --- | --- |
| Candidate entry | `configs/policy_search/candidates/ppo_issue791_best_v1.yaml` |
| Baseline config | `configs/baselines/ppo_15m_grid_socnav.yaml` |
| Adapter implementation | `robot_sf/baselines/ppo.py` |
| Model registry | `model/registry.yaml` entry `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417` |
| Observation mode | `dict` via the PPO baseline config |
| Action family | Unicycle command with `v_max: 2.0` and `omega_max: 1.0` |
| Device handling | `device: auto`, with predictive-foresight device explicitly set to `cuda` |
| Determinism | `deterministic: true` |
| Fallback policy | `fallback_to_goal: false`; the policy must fail closed instead of silently switching to goal-seeking |
| Caveat | Training used the eval superset; report as benchmark-set performance, not OOD generalization |

This is the best concrete interface mapping because it connects a policy-search candidate, baseline
config, model registry entry, adapter code, observation/action declaration, and fail-closed fallback
policy.

## Planned Mapping: LiDAR PPO MLP Gate

The `ppo_lidar_mlp_gate_v1` launch-packet candidate is the clean planned/no-checkpoint example
because it has a structured eligibility spec but no checkpoint yet:

| Contract field | Current mapping |
| --- | --- |
| Candidate spec | `configs/training/lidar/lidar_ppo_mlp_eligibility_issue_1615.yaml` |
| Launch packet | `configs/training/lidar/lidar_learned_policy_launch_packet_issue_1615.yaml` |
| Observation mode | `DEFAULT_GYM` |
| Observation level | `lidar_2d` |
| Runtime keys | `drive_state` and `rays` |
| Forbidden runtime inputs | SocNav structured pedestrian state, occupancy grids, future trajectories, collision labels, and outcome labels |
| Action family | Velocity command in robot local frame, linear m/s plus angular rad/s |
| Projection policy | Clip raw model output to Robot SF differential-drive bounds and log projection metadata |
| Checkpoint status | No checkpoint yet; issue #1662 owns the first smoke training follow-up |
| Missing checkpoint policy | Fail closed as `not_available` before benchmark episodes are written |
| Evidence status | `metadata-eligible` / `eligible_for_research_only`, not benchmark-ready |

For this candidate, the first implementation issue should not start by changing benchmark metrics.
It should materialize a smoke config, validate the eligibility metadata, run the bounded smoke, and
record compact artifacts exactly as described in `docs/context/issue_1615_lidar_learned_policy_plan.md`.

## Current PPO Adapter Implication

`robot_sf/baselines/ppo.py` already exposes model loading, deterministic prediction, dict
observations, device selection, metadata, and action conversion. It also has legacy
`fallback_to_goal` recovery paths for robustness. For learned-policy evidence, adapter configs must
set `fallback_to_goal: false` or otherwise report fallback commands as `fallback`, not as model
success. `configs/baselines/ppo_issue_791_eval_aligned_large_capacity.yaml` already follows this
pattern.

## Follow-Up Boundary

- Issue #1662 is the concrete LiDAR PPO MLP smoke follow-up.
- Issue #1619 and PR #1670 cover the learned-policy registry surface.
- Future adapter implementation issues should name one concrete policy family, one checkpoint or
  source-reproduction target, one observation/action contract, and one smoke command.

Do not open a generic learned-policy adapter issue without a runnable candidate and a proof path.

## Validation

Docs-only changes to this contract should run:

```bash
rg -n "LearnedLocalPolicy|fallback|not_available|ppo_lidar_mlp_gate_v1" \
  docs/context/issue_1618_learned_policy_adapter_interface.md \
  docs/context/policy_search/README.md \
  docs/context/README.md
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```

Code changes that implement this contract must add targeted tests around schema validation,
unsupported observation/kinematics failure, missing checkpoint failure, deterministic inference, and
per-step diagnostic emission before using PR readiness as proof.
