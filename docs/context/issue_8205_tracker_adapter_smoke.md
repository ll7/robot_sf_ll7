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
- The enabled channel carries one `GoalBeliefV1` per tracker ID, sorted by stable `track_id`,
  with `source=observation_only`, `coordinate_frame=global_xy`, the tracker timestamp and step,
  explicit visibility masks, and the adapter configuration hash.
- Until a separate observation-only candidate provider exists, every belief is explicitly
  `mode=unavailable` with all candidate mass assigned to `unknown_candidate_probability=1.0`.
  The adapter does not infer a destination, force, route, or posterior.
- Tracker covariance, association confidence, lifecycle counters, blockers, and the conservative
  history projection are available under channel diagnostics. These diagnostics contain no
  simulator pedestrian ID, route assignment, true goal, or PySocialForce state.
- `adapter.reset(token)` starts a new adapter epoch and places the caller-owned reset token in
  enabled belief/channel metadata. The adapter itself retains no track or simulator state.

## History and missing-data policy

The tracker result stores an oldest-to-newest validity mask but does not expose a per-history-row
velocity-validity mask. The adapter therefore uses a fail-closed projection:

- a contiguous history whose span is provable from `age_steps` is converted into observed rows;
  invalid rows inside that active span are emitted as `invisible` rows with no vectors;
- padded rows before the first valid row are omitted;
- timestamp/step gaps, malformed history, or an unavailable current velocity reduce the payload
  to a current conservative row and record a `current_row_only_*` projection blocker;
- a current invisible or prediction-only tracker row never promotes predicted position/velocity
  into an observed actor row.

This is an integration contract, not a claim that the current tracker history is sufficient for a
future estimator. Extending field-level history masks belongs with the tracker contract and must
be reviewed separately.

## Boundary and non-goals

The adapter accepts only the validated `PedestrianTrackingResult` type. It does not accept a
generic simulator object and does not call the existing simulator-state
`planner_goal_posterior_channel_from_state` path. The latter remains a separate opt-in path and
is not an actor input to this adapter.

The deterministic fixture covers stable row reorder, same-step robot-frame velocity normalization,
brief occlusion and reacquisition, reset isolation, missing velocity, disabled tracking, malformed
input, and deterministic serialization. The smoke receipt records implementation integrity only;
it contains no tracking-quality, prediction-quality, planner, benchmark, or paper-grade result.

## Validation

```bash
uv run pytest tests/prediction/test_tracker_goal_belief_adapter.py -q
uv run ruff check robot_sf/prediction tests/prediction/test_tracker_goal_belief_adapter.py
uv run ruff format --check robot_sf/prediction tests/prediction/test_tracker_goal_belief_adapter.py
```

The durable receipt is
[`docs/context/evidence/issue_8205_tracker_adapter_smoke_receipt.v1.json`](evidence/issue_8205_tracker_adapter_smoke_receipt.v1.json).
