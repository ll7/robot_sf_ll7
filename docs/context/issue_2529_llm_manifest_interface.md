# Issue #2529 LLM-to-Scenario Manifest Interface (2026-06-12)

Status: interface/proposal-only. This contract gates execution but does not make any
benchmark or safety claim.

Related surfaces:

- [Issue #2524 Adversarial Scenario Manifest Generation](issue_2524_adversarial_manifests.md)
- [Issue #2468 Adversarial Scenario Generation Roadmap](issue_2468_adversarial_generation_roadmap.md)
- [adversarial scenario manifest module](../../robot_sf/adversarial/scenario_manifest.py)

## Contract goal

Security and planning teams requested that LLMs propose adversarial ideas only through
explicit `adversarial_scenario_manifest.v1` payloads so planner execution can fail-closed
before simulation.

The contract is:

- A helper model may emit only a structured payload.
- The payload must validate deterministically against the manifest schema.
- Any manifest that is not `VALID` must never execute; it is marked rejected and routed for human review.

## Prompt contract

All LLM calls that target scenario proposal must return a single serialized object
matching this template with no markdown wrappers and no prose:

```yaml
schema_version: adversarial_scenario_manifest.v1
execution_status: generated_only
evidence_boundary: "diagnostic-only: ..."
source:
  scenario_template: <string>
  search_space: <string>
  map_id: <string>
  scenario_name: <string>
  config_path: <string>
  search_space_path: <string>
generator:
  family: <string>
  generator_id: <string>
  seed: <int>
  candidate_index: <int>
candidate_controls:
  start:
    x: <float>
    y: <float>
  goal:
    x: <float>
    y: <float>
  spawn_time_s: <float>
  pedestrian_speed_mps: <float>
  pedestrian_delay_s: <float>
  scenario_seed: <int>
```

Validation hard-fail conditions for the current validator path:

- malformed YAML/JSON payloads,
- wrong `schema_version`,
- missing or non-finite fields,
- out-of-bound controls for the selected search space,
- negative timing or non-positive speed,
- duplicate control hash under the same batch.

Accepted artifacts from LLM output are not direct execution rows.

## Deterministic gate and fail-closed path

Execution path:

1. Parse payload as YAML/JSON.
2. Call `validate_manifest_payload` from `robot_sf.adversarial.scenario_manifest`.
3. Call with deterministic `SearchSpaceConfig` loaded from the declared search-space fixture.
4. Continue only when result status is `valid`.
5. For `invalid`, `degenerate`, or any missing execution precondition, mark rejected and stop.

`validate_manifest_payload` uses pure, deterministic checks:

- schema-level check (`schema_version`, `candidate_controls` typing),
- typed control conversion,
- bounded-space checks when search-space is supplied,
- duplicate-control hashing via `compute_control_hash`,
- status classification (`valid`, `invalid`, `degenerate`).

No execution path reads prompt text directly.

## Examples

### Accepted example manifest

```yaml
schema_version: adversarial_scenario_manifest.v1
execution_status: generated_only
evidence_boundary: diagnostic-only
source:
  scenario_template: crossing_ttc.yaml
  search_space: crossing_ttc_space.yaml
  map_id: classic_cross_trap
  scenario_name: crossing_ttc_template
  config_path: configs/scenarios/templates/crossing_ttc.yaml
  search_space_path: configs/adversarial/crossing_ttc_space.yaml
generator:
  family: llm_to_manifest
  generator_id: structured_prompt_adapter_v1
  seed: 2026
  candidate_index: 0
candidate_controls:
  start:
    x: 1.0
    y: 2.0
  goal:
    x: 5.0
    y: 2.0
  spawn_time_s: 0.0
  pedestrian_speed_mps: 1.0
  pedestrian_delay_s: 0.0
  scenario_seed: 7
```

### Rejected example manifest

```yaml
schema_version: adversarial_scenario_manifest.v1
execution_status: generated_only
evidence_boundary: diagnostic-only
candidate_controls:
  start:
    x: 1.0
    y: 2.0
  goal:
    x: 5.0
  spawn_time_s: 0.0
  pedestrian_speed_mps: 1.0
  pedestrian_delay_s: 0.0
  scenario_seed: 7
generator:
  family: llm_to_manifest
  generator_id: structured_prompt_adapter_v1
  seed: 2026
  candidate_index: 1
```

Failure mode for the rejected example:
`candidate_controls.goal must define x and y` → status `invalid`, no execution.

## Human review / scientific interpretation boundary

The manifest gate is a deterministic safety interface only.

- Humans/reviewers own final interpretation of whether a valid candidate is scientific or
  benchmark-relevant.
- This contract does not claim LLM proposals are valid adversarially complete,
  high-utility, or safety-strong.
- `adversarial_scenario_manifest.v1` validity is a necessary precondition, not a benchmark claim.

## Deterministic evidence check

The accepted and rejected example files are stored in `tests/adversarial/fixtures/issue_2529/` and are
validated in `tests/adversarial/test_adversarial_scenario_manifest.py` using
`validate_manifest_payload`.
