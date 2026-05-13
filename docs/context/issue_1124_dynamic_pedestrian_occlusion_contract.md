# Issue 1124 dynamic pedestrian occlusion contract

Date: 2026-05-12

## Scope

This note records the first opt-in dynamic pedestrian occlusion contract for planner-facing SocNav observations.

Implemented files:

- `robot_sf/sensor/socnav_observation.py`
- `robot_sf/gym_env/unified_config.py`
- `robot_sf/training/scenario_loader.py`
- `tests/test_socnav_dynamic_occlusion.py`
- `tests/test_socnav_observation.py`
- `tests/training/test_scenario_loader.py`

## Contract

Dynamic occlusion is disabled by default and enabled only through:

```yaml
observation_visibility:
  enabled: true
  dynamic_occlusion: true
```

The setting is exposed in benchmark metadata through `ObservationVisibilitySettings.to_metadata()` as `dynamic_occlusion`.

Planner-facing visibility semantics:

- Pedestrians are modeled as circular blockers with radius `SimulationSettings.ped_radius`.
- The helper tests center-line visibility from robot center to target pedestrian center.
- A target pedestrian is hidden when a nearer currently-visible pedestrian disk intersects that line segment.
- Already-hidden pedestrians do not act as blockers for farther pedestrians.
- Simulator ground-truth state, metrics, and raw pedestrian arrays are not mutated; filtering applies only to planner-facing SocNav observations.

## Limits

This is an abstraction, not a calibrated perception model.

- It does not model partial body visibility.
- It does not model detector probabilities, false positives, tracking delay, or sensor-specific angular resolution.
- Results with `dynamic_occlusion=true` should not be compared against default observations without noting the opt-in visibility setting.

## Remaining work

- Run targeted tests and a visibility smoke before PR handoff.
- Add scenario examples only if needed for a benchmark profile.
- Keep reports explicit about `observation_visibility.dynamic_occlusion`.
