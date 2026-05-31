# External Learned-Policy Intake Contract

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1870>

## Purpose

Use this contract when an external learned local-navigation family is screened for possible Robot
SF adapter work. It is an intake and routing contract only: it does not download checkpoints,
implement policies, reproduce upstream training, or reclassify the full learned-policy registry.

`docs/context/policy_search/learned_policy_registry.md` remains the source of truth for the durable
current-state row. This note defines the stage evidence that a row or family-specific assessment
must cite before the registry status can move.

## Stage Status Enum

Use this small per-stage status vocabulary in source-assessment notes, issue comments, or intake
checklists:

| Status | Meaning |
| --- | --- |
| `not_started` | No review evidence has been recorded for the stage. |
| `needs_evidence` | The candidate may be viable, but the required proof is missing or incomplete. |
| `passed` | The named stage has direct evidence and a link to the command, artifact, or source note. |
| `blocked` | A known missing dependency, asset, license answer, contract, or runtime prevents progress. |
| `rejected` | The candidate is incompatible with the current Robot SF local-planner contract. |
| `not_applicable` | The stage is irrelevant for this candidate and the reason is stated. |

These values are not a second registry. Durable roll-up status belongs in
`learned_policy_registry.md` using `integration_status`, `reproducibility_status`, and
`benchmark_status`.

## Intake Stages

Move through the stages in order. Do not skip a stage unless the assessment records why it is
`not_applicable`.

| Stage | Required evidence | Registry effect |
| --- | --- | --- |
| Source screen | Upstream paper/repository or local source note, intended task, sensor assumptions, action output, and dependency class. | May justify a `monitor_only` row, but not adapter work by itself. |
| License check | License text, asset/checkpoint license when different, and compatibility note for source reuse versus clean-room reimplementation. | Missing or incompatible license keeps `integration_status: monitor_only` or `rejected`. |
| Checkpoint check | Durable checkpoint URI, checksum or source manifest, normalizer/statistics availability, and access constraints. | Missing checkpoint keeps benchmark status `blocked` unless the candidate is explicitly a design/reference family. |
| Observation/action contract | Mapping to Robot SF observations, forbidden/training-only fields, action frame/units/bounds, and projection or guard behavior. | Required before `adapter_needed` can become `staged` or `implemented`. |
| Source-side smoke | Upstream or source-harness inference command that runs the named policy/checkpoint in its native environment. | Source-side success is `not_benchmark_evidence`; it can only support source confidence or adapter planning. |
| Robot SF adapter | Adapter path, config pointer, metadata fields, fail-closed behavior, and raw/adapted/post-guard action logging plan. | Adapter-only status is `not_benchmark_evidence` until the Robot SF smoke runs through that adapter. |
| Robot SF smoke | Minimal Robot SF command that loads the adapter and checkpoint and records expected diagnostics without fallback-only success. | Supports `smoke_only`, not benchmark ranking or paper-facing claims. |
| Benchmark suite | Declared benchmark config/stage, candidate-registry row, promotion gate, artifact provenance, and fail-closed result handling. | Only this stage can support `comparison_available` benchmark evidence. |

## Registry Mapping

Use this roll-up mapping when updating `learned_policy_registry.md`. Keep detailed stage evidence in
the family note or issue; keep only the current state and anchors in the registry row.

| Intake roll-up | Minimum completed stage | `integration_status` | `reproducibility_status` | `benchmark_status` |
| --- | --- | --- | --- | --- |
| `source_screened` | Source screen | `monitor_only` | `monitor_only` or `source_harness_required` | `blocked` or `not_benchmark_evidence` |
| `license_or_checkpoint_blocked` | Source screen, with license or checkpoint blocker | `monitor_only` or `rejected` | `blocked` or `source_harness_required` | `blocked` or `rejected_for_current_adapter` |
| `contract_blocked` | Observation/action contract attempted | `monitor_only` or `adapter_needed` | `source_harness_required` or `monitor_only` | `blocked` or `rejected_for_current_adapter` |
| `source_smoke_only` | Source-side smoke | `monitor_only` or `adapter_needed` | `source_smoke_proven` | `not_benchmark_evidence` |
| `adapter_only` | Robot SF adapter | `staged` or `adapter_needed` | `launch_packet` or `source_smoke_proven` | `not_benchmark_evidence` |
| `robot_sf_smoke_only` | Robot SF smoke | `staged` or `implemented` | `smoke_proven` | `smoke_only` |
| `benchmark_suite_complete` | Benchmark suite | `implemented` | `comparison_available` | `comparison_available` |

When the table above conflicts with a stricter existing row, keep the stricter row. For example, a
candidate with a successful source-side smoke but a missing Robot SF observation/action contract
still remains blocked for Robot SF benchmark use.

## Example: Arena-Rosnav Stack

`arena_rosnav_stack` is already tracked in
`docs/context/policy_search/learned_policy_registry.md` and assessed in
`docs/context/policy_search/issue_1758_arena_rosnav_source_assessment.md`.

Current intake interpretation:

- Source screen: `passed`; the source family is identified as a ROS Noetic/Gazebo/Flatland stack.
- License, checkpoint, and source-side smoke: `needs_evidence` or `blocked` until a named Rosnav
  agent, durable source assets, and source command are proven.
- Observation/action contract and Robot SF adapter: `not_started`; no single Robot SF-compatible
  policy checkpoint or adapter contract is claimed.
- Robot SF smoke and benchmark suite: `blocked`.

Registry roll-up stays `integration_status: monitor_only`,
`reproducibility_status: source_harness_required`, and `benchmark_status: blocked`. This prevents
the source-side assessment from becoming benchmark evidence.

## Benchmark Evidence Boundary

Source-side success is not Robot SF evidence. Adapter import success is not benchmark evidence.
Fallback-only, degraded, guard-dominated, or missing-checkpoint execution must be reported as a
caveat or blocker unless the issue explicitly exists to measure that mode.

A policy may support benchmark claims only after the benchmark-suite stage names the runnable
Robot SF candidate, config, checkpoint/artifact provenance, observation/action contract, smoke
proof, benchmark stage, promotion gate, and fail-closed status.

## Validation

For docs-only changes to this contract or linked registry rows, use the cheap validation path:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
uv run python scripts/validation/validate_policy_search_registry.py
test -e docs/context/policy_search/learned_policy_registry.md
test -e docs/context/policy_search/issue_1758_arena_rosnav_source_assessment.md
```
