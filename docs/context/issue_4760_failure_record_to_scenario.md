# Issue #4760: Failure Record to Scenario Conversion

## Overview

This document defines a prototype conversion path that turns structured failure records into executable `robot_sf_ll7` scenario hypotheses. Generated scenarios are **hypotheses only** and require human review before any execution or evidence claims.

## Input Schema

Failure records use YAML format with the following schema:

```yaml
schema_version: failure-record.v1
failure_record:
  id: "<unique-identifier>"
  source: "<source-description>"
  date: "<ISO-8601-date>"
  environment: "sidewalk|shared_space|road_edge|crossing|event"
  actors:
    - type: "<actor-type>"
      count: <integer>
      description: "<optional-description>"
  triggering_condition: "<description-of-trigger>"
  failure_mode: "collision|near_miss|stuck|blocked_path|unsafe_fallback|pedestrian_disruption"
  contextual_factors:
    - "<factor-1>"
    - "<factor-2>"
  required_manual_review: true
  claim_boundary: "scenario hypothesis only; not evidence"
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | yes | Must be `failure-record.v1` |
| `id` | string | yes | Unique identifier for the failure record |
| `source` | string | yes | Origin of the failure record (e.g., incident report, test log) |
| `date` | string | yes | ISO-8601 date of the incident |
| `environment` | string | yes | One of: `sidewalk`, `shared_space`, `road_edge`, `crossing`, `event` |
| `actors` | list | yes | List of actors involved; controls pedestrian count in scenario |
| `triggering_condition` | string | yes | What triggered the failure |
| `failure_mode` | string | yes | One of: `collision`, `near_miss`, `stuck`, `blocked_path`, `unsafe_fallback`, `pedestrian_disruption` |
| `contextual_factors` | list | no | Additional context (e.g., `dense_crowd`, `temporary_obstacle`) |
| `required_manual_review` | bool | yes | Always `true` for generated scenarios |
| `claim_boundary` | string | yes | Must state "scenario hypothesis only; not evidence" |

## Output Schema

Generated scenarios follow the `robot_sf.scenario_matrix.v1` schema with additional metadata:

```yaml
schema_version: robot_sf.scenario_matrix.v1
scenarios:
  - name: "<generated-name>"
    map_file: "<path-to-svg-map>"
    simulation_config:
      max_episode_steps: <integer>
      ped_density: <float>
    single_pedestrians: []
    robot_config: {}
    metadata:
      generated_from_failure_record: "<record-id>"
      generation_method: deterministic_template_v1
      required_manual_review: true
      claim_boundary: "scenario hypothesis only; not executed evidence"
      generated_assumptions:
        - "<assumption-1>"
        - "<assumption-2>"
      invalidity_warnings:
        - "<warning-1>"
      expected_failure_modes:
        - "<expected-mode-1>"
      authoring:
        status: draft
        source_issue: "#4760"
        generated_by: scripts/tools/convert_failure_record_to_scenario.py
        benchmark_evidence: false
    seeds:
      - 101
      - 102
      - 103
```

### Output Metadata Fields

| Field | Description |
|-------|-------------|
| `generated_from_failure_record` | Source failure record ID |
| `generation_method` | Always `deterministic_template_v1` for this prototype |
| `required_manual_review` | Always `true` |
| `claim_boundary` | Clear statement that this is not evidence |
| `generated_assumptions` | List of assumptions made during conversion |
| `invalidity_warnings` | Known limitations or warnings |
| `expected_failure_modes` | Failure modes the scenario is designed to expose |

## Mapping Rules

The converter uses deterministic templates based on environment and failure mode:

### Environment + Failure Mode Mappings

| Environment | Failure Mode(s) | Target Template | Map Family |
|-------------|-----------------|-----------------|------------|
| `event` | `blocked_path`, `stuck`, `unsafe_fallback` | event-disruption | `event_disruption` |
| `sidewalk` | `blocked_path`, `near_miss` | sidewalk-blocked | `ammv_sidewalk` |
| `shared_space` | `blocked_path`, `near_miss` | shared-space-blocked | `ammv_shared_space` |
| `crossing` | `near_miss`, `collision` | crossing-stress | `classic_crossing` |
| `road_edge` | `stuck`, `unsafe_fallback` | road-edge-stress | `road_edge` |

### Actor Mapping

- `actors` list length controls `single_pedestrians` count
- Each actor becomes a pedestrian with deterministic ID (`h1`, `h2`, ...)
- Actor `count` field is recorded in metadata but not expanded (single pedestrians only for v1)

### Contextual Factor Mapping

Contextual factors become scenario metadata, not simulator behavior:

| Factor | Metadata Effect |
|--------|-----------------|
| `dense_crowd` | Sets `ped_density` hint in metadata |
| `temporary_obstacle` | Added to `generated_assumptions` |
| `communication_loss` | Added to `invalidity_warnings` |
| Unknown factors | Added to `invalidity_warnings` as "unmapped factor" |

## Limitations

1. **No LLM generation**: This prototype uses deterministic templates only.
2. **Single pedestrians only**: Actor counts > 1 are noted but not expanded into groups.
3. **Map assumptions**: Converter assumes canonical maps exist for each template.
4. **No validation of executability**: Generated scenarios may reference POIs that do not exist in the target map.
5. **No evidence claims**: Generated scenarios are hypotheses until executed and reviewed.

## Human Review Requirement

Before using a generated scenario:

1. Verify the map file exists and contains required POIs.
2. Review `generated_assumptions` for validity.
3. Check `invalidity_warnings` for deal-breakers.
4. Execute the scenario in a controlled environment.
5. Record execution results separately; do not treat the generated scenario as evidence.

## Usage

```bash
# Convert a single failure record
uv run python scripts/tools/convert_failure_record_to_scenario.py \
  --record configs/failure_records/examples/ammv_sidewalk_blocked_path.yaml \
  --output-yaml output/failure_record_scenarios/ammv_sidewalk_blocked_path.scenario.yaml

# Validate the generated scenario
uv run python scripts/tools/validate_scenario.py \
  output/failure_record_scenarios/ammv_sidewalk_blocked_path.scenario.yaml
```

## Example Workflow

1. Author a failure record in `configs/failure_records/examples/`.
2. Run the converter to generate a scenario hypothesis.
3. Review the generated YAML for assumptions and warnings.
4. Execute the scenario manually to verify it runs.
5. If useful, promote through normal scenario certification workflow.

## Related Files

- `scripts/tools/convert_failure_record_to_scenario.py` - Converter implementation
- `configs/failure_records/examples/` - Example failure records
- `tests/tools/test_convert_failure_record_to_scenario.py` - Test suite
