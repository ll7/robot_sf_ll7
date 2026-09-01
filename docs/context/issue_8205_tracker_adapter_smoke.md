# Issue #8205: Observation-Derived Tracker Goal-Belief Adapter

Status: implemented as an opt-in observation-only bridge; smoke evidence only (2026-09-01).

## Purpose and ownership

`robot_sf.sensor.pedestrian_tracking.PedestrianTracker` remains the owner of association,
track lifecycle, frame normalization, and sensor-history semantics. The follow-up adapter in
`robot_sf.prediction.tracker_goal_belief_adapter` is only a typed bridge from a validated
`PedestrianTrackingResult` to the canonical actor-side `GoalBeliefV1` contract from #8063.

The package-level adapter names are lazy exports from `robot_sf.prediction`; the direct module
path is the stable implementation owner. The lazy export avoids importing the sensor tracker
while `robot_sf.prediction` is initializing.

## Runtime contract

```python
from robot_sf.prediction import TrackerGoalBeliefAdapter, TrackerGoalBeliefAdapterConfig

adapter = TrackerGoalBeliefAdapter(TrackerGoalBeliefAdapterConfig(enabled=True))
result = tracker.update(snapshot)
channel = adapter.adapt(result)
beliefs = channel.beliefs
```

- `TrackerGoalBeliefAdapterConfig()` is default-off. The disabled channel carries no timing,
  tracks, or beliefs, so it does not alter Gym observations, actions, spaces, checkpoint inputs,
  or reset metadata.
- The enabled channel carries one `GoalBeliefV1` per tracker ID, sorted by the serialized textual
  `track_id`, with `source=observation_only`, `coordinate_frame=global_xy`, the tracker timestamp
  and step, one current decision-point history row, and the adapter configuration hash.
- Until a separate observation-only candidate provider exists, every belief is explicitly
  `mode=unavailable` with all candidate mass assigned to `unknown_candidate_probability=1.0`.
  The adapter does not infer a destination, force, route, or posterior.
- Tracker covariance, association confidence, lifecycle counters, blockers, and the conservative
  history projection are available under channel diagnostics. These diagnostics contain no
  simulator pedestrian ID, route assignment, true goal, or PySocialForce state.
- `adapter.reset(token)` starts a new adapter epoch and places the caller-owned reset token in
  enabled belief/channel metadata. The adapter itself retains no track or simulator state.

## History and missing-data policy

`pedestrian_tracking.v1` stores an oldest-to-newest history-validity mask alongside position and
velocity history, but it does not expose row-level velocity provenance. A historical velocity can
therefore be estimated or predicted even when the current velocity is later available. The adapter
uses a stateless, fail-closed projection until the tracker contract provides that provenance:

- a currently visible row with currently available position and velocity becomes one `observed`
  current decision-point row;
- an unavailable current velocity, or a current invisible/lost/prediction-only row, becomes one
  `invisible` row with no position or velocity vectors;
- every enabled belief contains exactly one current row; prior position or velocity history is
  withheld, and the projection records either `current_row_only_velocity_unavailable` or
  `current_row_only_tracker_v1_velocity_provenance_unavailable` as appropriate.

Occlusion and reacquisition are represented across successive channels as current-row masks. They
are not reconstructed as multi-step actor history. Full history is explicitly deferred to a
separately reviewed tracker-contract extension with per-row velocity provenance.

## Boundary and non-goals

The adapter accepts only the validated `PedestrianTrackingResult` type. It does not accept a
generic simulator object and does not call the existing simulator-state
`planner_goal_posterior_channel_from_state` path. The latter remains a separate opt-in path and
is not an actor input to this adapter.

The deterministic fixture covers stable row reorder, same-step robot-frame velocity normalization,
brief occlusion and reacquisition across successive channels, reset isolation and epoch pairing,
missing velocity and recovery, serialized textual ordering, disabled tracking, malformed input, and
deterministic serialization. The smoke receipt records implementation integrity only; it contains no
tracking-quality, prediction-quality, planner, benchmark, safety, or paper-grade result.

## Validation

```bash
uv run pytest tests/prediction/test_tracker_goal_belief_adapter.py -q
uv run ruff check robot_sf/prediction tests/prediction/test_tracker_goal_belief_adapter.py
uv run ruff format --check robot_sf/prediction tests/prediction/test_tracker_goal_belief_adapter.py
```

The durable receipt is
[`docs/context/evidence/issue_8205_tracker_adapter_smoke_receipt.v1.json`](evidence/issue_8205_tracker_adapter_smoke_receipt.v1.json).
