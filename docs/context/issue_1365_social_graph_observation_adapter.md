# Issue #1365 Social Graph Observation Adapter 2026-05-20

Date: 2026-05-20

Related issue:
- `robot_sf_ll7#1365` enhancement: add social-RL pedestrian graph observation adapter

Related context:
- `docs/context/issue_601_crowdnav_feasibility_note.md`
- `docs/context/issue_627_sonic_wrapper_followup.md`
- `docs/context/issue_770_igat_st2_attention_assessment.md`
- `docs/benchmark_planner_family_coverage.md`
- `robot_sf/sensor/socnav_observation.py`
- `robot_sf/sensor/social_graph_observation.py`

## Goal

Provide a reusable Robot SF observation adapter for graph/social-RL candidate screening without
adding another source-policy wrapper or changing the benchmark observation mode.

The adapter translates the existing SocNav structured observation into deterministic graph-style
arrays with explicit masks, caps, robot-relative features, optional static-obstacle tokens, and a
small stateful history stack.

## Implemented Surface

New module:

- `robot_sf/sensor/social_graph_observation.py`

Public entry points:

- `SocialGraphObservationConfig`
- `SocialGraphObservationAdapter`
- `build_social_graph_observation`
- feature-name constants for robot, pedestrian, and static-obstacle rows

The adapter accepts both nested `ObservationMode.SOCNAV_STRUCT` observations and the flattened
SocNav keys used by Stable-Baselines-compatible benchmark paths. It emits:

- `robot_features`
- `pedestrian_features`
- `pedestrian_mask`
- `pedestrian_count`
- `pedestrian_history`
- `static_obstacle_features`
- `static_obstacle_mask`
- `static_obstacle_count`
- `edge_index`
- `edge_type`

## Contract Boundary

This is a deployment-observation adapter only.

It intentionally uses current-timestep Robot SF data:

- robot position, heading, velocity, radius, and current goal,
- visible/current pedestrian positions and velocities already exposed through SocNav observations,
- optional static obstacle line segments supplied by the caller.

It rejects future-like or label-only fields such as `future_positions` so candidate wrappers cannot
quietly use training labels or privileged future trajectories through this path.
This enforcement is key-name based; callers remain responsible for provenance when data arrives
under ordinary current-timestep field names.

The first implementation does not claim source-family parity for CrowdNav++, HEIGHT, DS-RNN, RGL,
SoNIC, GenSafeNav, or SAGE. Those methods still need source-harness or model-specific parity proof
before benchmark claims. This adapter only removes one repeated local risk: hidden differences in
pedestrian ordering, masks, caps, and relative feature packing.

## Candidate Consumption Notes

CrowdNav-family candidates can consume the adapter as an in-repo pre-normalization layer when a
future wrapper needs a Robot SF graph input. Candidate-specific wrappers still own:

- source-specific normalization,
- policy-specific node/edge feature ordering,
- action projection into Robot SF command space,
- checkpoint and source-harness provenance,
- and benchmark-readiness metadata.

For HEIGHT, SoNIC, and GenSafeNav, existing model-only wrappers remain source-specific. This shared
adapter is most useful for future candidate comparison, contract tests, and wrapper prototypes that
need deterministic Robot SF pedestrian graph state before rebuilding a source tensor.

## Validation

Targeted commands:

```bash
.venv/bin/python -m ruff check robot_sf/sensor/social_graph_observation.py tests/test_social_graph_observation.py
PYTEST_NUM_WORKERS=8 uv run pytest tests/test_social_graph_observation.py -q
PYTEST_NUM_WORKERS=8 uv run pytest tests/test_social_graph_observation.py tests/test_socnav_observation.py tests/test_socnav_observation_mode.py -q
```

Observed targeted result:

- Ruff passed for the new module and tests.
- `tests/test_social_graph_observation.py`: covered by the broader SocNav run below after final
  Copilot-review hardening.
- `tests/test_social_graph_observation.py tests/test_socnav_observation.py
  tests/test_socnav_observation_mode.py`: 16 passed after final Copilot-review hardening.

Covered behavior:

- empty-pedestrian zero masks and fixed shapes,
- deterministic nearest ordering with geometric tie-breaks under input permutation,
- velocity tie-breaks for exact duplicate pedestrian positions,
- cap truncation and masks,
- fail-closed handling when `robot.velocity_xy` is absent, so angular speed is not used as a
  lateral velocity,
- optional static-obstacle tokens and typed edges,
- stateful history fill, shift, and reset,
- flat vs nested SocNav parity,
- fail-closed rejection of future-like deployment fields.

## Follow-Up Boundary

Safe next steps:

- use the adapter in a future source-specific wrapper prototype after source-harness evidence is
  available,
- add a benchmark metadata row only when a concrete planner entrypoint consumes the adapter,
- extend feature rows only with current-timestep deployment fields unless a separate training-only
  path is created.

Not included here:

- no new planner algorithm key,
- no benchmark promotion,
- no policy integration,
- no training labels or prediction targets.
