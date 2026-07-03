# Issue #3977 Public-Requirement Scenario Context

[Issue #3977](https://github.com/ll7/robot_sf_ll7/issues/3977) added four deterministic
scenario proxies for public-requirement diagnostics. They are authored fixtures that exercise
loader, metadata, and detector contracts; they are not public-acceptance evidence, safety
certification evidence, benchmark gates, or paper/dissertation claim evidence.

## Scenario Families

The Phase-1 scenario set is tracked by
[`configs/scenarios/sets/issue_3977_public_requirements.yaml`](../../configs/scenarios/sets/issue_3977_public_requirements.yaml)
and currently includes:

| Family | Scenario file | Diagnostic condition |
| --- | --- | --- |
| Safe braking | [`issue_3977_safe_braking.yaml`](../../configs/scenarios/single/issue_3977_safe_braking.yaml) | A pedestrian steps into the robot path after a proximity release. |
| Visibility and intent | [`issue_3977_visibility_and_intent.yaml`](../../configs/scenarios/single/issue_3977_visibility_and_intent.yaml) | A pedestrian starts or stops near a robot turn where intent should be legible. |
| Emergency reaction | [`issue_3977_emergency_reaction.yaml`](../../configs/scenarios/single/issue_3977_emergency_reaction.yaml) | A sudden obstacle proxy becomes conflict-relevant near the robot corridor. |
| Speed limit | [`issue_3977_speed_limit.yaml`](../../configs/scenarios/single/issue_3977_speed_limit.yaml) | A route-following scenario exposes an authored sidewalk speed cap. |

## Metadata Contract

Each scenario uses one scenario-level `metadata` block with:

- `metadata.public_requirement.schema_version == public-requirement-scenario.v1`
- `metadata.public_requirement.claim_boundary == authored_scenario_proxy_not_human_subject_evidence`
- `metadata.public_requirement.event_contract.type`
- `metadata.archetype`
- `metadata.flow`
- `metadata.behavior`
- `metadata.purpose`

Sibling scenario files should keep those descriptive fields as siblings of
`metadata.public_requirement`, matching `issue_3977_safe_braking.yaml` and
`issue_3977_speed_limit.yaml`. The focused loader test in
[`tests/scenarios/test_issue_3977_public_requirement_scenarios.py`](../../tests/scenarios/test_issue_3977_public_requirement_scenarios.py)
guards the manifest and metadata shape.

## Detector Contract

The diagnostic detector output schema is `public-requirement-events.v1`. The authored event types
covered by this Phase-1 set are:

- `pedestrian_steps_in_front`
- `turn_or_start_stop_near_pedestrian`
- `sudden_obstacle_proxy`
- `speed_limit_monitor`

These events are deterministic contract probes. They may support local debugging and regression
tests, but they do not establish public-requirement satisfaction, safety certification, release
readiness, benchmark ranking, or manuscript-facing conclusions.
