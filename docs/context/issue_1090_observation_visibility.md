# Issue #1090 Observation Visibility

Related issue: [#1090](https://github.com/ll7/robot_sf_ll7/issues/1090)

## Decision

Add opt-in field-of-view, range, and static-occlusion filtering at the SocNav structured observation
builder. This is the planner-facing boundary used by benchmark planners, while the simulator state
continues to expose full ground truth to metrics and episode recording.

## Why This Boundary

Filtering in `robot_sf/sensor/socnav_observation.py` keeps the feature close to the observation
contract and avoids changing simulator dynamics, pedestrian state, metric inputs, or map state. It
also covers the benchmark path because `robot_sf/benchmark/map_runner.py` builds environments with
`ObservationMode.SOCNAV_STRUCT`.

Scenario parsing lives in `robot_sf/training/scenario_loader.py` so benchmark YAML can express the
settings directly. The original scenario block remains part of `scenario_params`, and runtime
settings are copied into `algorithm_metadata.observation_visibility`.

## Scope Boundary

Implemented:

* FOV cone filtering around robot heading.
* Optional maximum range filtering.
* Static obstacle line-of-sight filtering using map obstacle polygons.
* Targeted tests that verify planner-facing observations hide pedestrians while simulator
  `ped_pos` remains intact.
* A reusable smoke config:
  `configs/scenarios/single/observation_visibility_blind_corner_smoke.yaml`.

Deferred:

* Dynamic pedestrian-to-pedestrian occlusion: follow-up
  [#1124](https://github.com/ll7/robot_sf_ll7/issues/1124). This needs a separate issue because
  body extent, depth ordering, and planner-contract semantics are easy to overclaim.

## Validation Plan

Targeted validation should include:

* `rtk uv run pytest tests/test_socnav_observation.py tests/training/test_scenario_loader.py -q`
* A representative benchmark run with
  `configs/scenarios/single/observation_visibility_blind_corner_smoke.yaml`
* Inspection that the generated episode JSONL contains
  `algorithm_metadata.observation_visibility`
* Full PR readiness after merging the latest `origin/main`
