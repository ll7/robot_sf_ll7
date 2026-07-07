# Footprint-orientation diagnostics for elongated AMMV bodies

> **Diagnostic proxy only.** This feature is a lightweight, CPU-only diagnostic.
> It is **not a full SE(2) planner implementation**, not a collision-checking
> runtime, not a benchmark or paper-facing result, and not calibrated against
> real-vehicle swept-volume data. See the `claim_boundary` in
> `configs/diagnostics/footprint_orientation_v1.yaml`.

## Plain-language summary

The robot in this repository is normally modeled as a **circle** (`radius_m`).
Real Autonomous Micromobility Vehicles (AMMVs) — kick scooters, cargo bikes,
delivery robots, shuttle pods — are **elongated**, so whether a route is
traversable depends on the body's yaw (heading), not just its centerline
clearance. This diagnostic compares route clearance under circular,
rectangular, and elongated footprint assumptions on the same route, so that a
route a circular body "clears" can be checked against what a scooter-, cargo-,
or shuttle-pod-class body would actually do.

Issue: #4762.

## What it reports

For each scenario and footprint model the diagnostic reports two **separate**
quantities:

- `centerline_clearance_m` — minimum distance from the route centerline (a
  Shapely `LineString`) to the obstacle polygons, ignoring the footprint. This
  is what the existing circular route-clearance contract reasons about.
- `footprint_aware_clearance_m` — minimum distance from an oriented rigid
  footprint, sampled along the route and oriented along the local tangent, to
  the obstacle polygons (`0.0` when overlapping).

It also reports a `status` of `clear` or `collision` (boundary contact counts
as collision, conservative fail-closed). Reporting both numbers separately is
the point: it makes circular-vs-elongated pass/fail outcomes directly
comparable on the same route.

## Footprint models

Defined in `configs/diagnostics/footprint_orientation_v1.yaml`:

| id | kind | radius / length × width |
| --- | --- | --- |
| `circular` | circular | r = 0.35 m |
| `scooter_like` | rectangular | 1.3 × 0.55 m |
| `cargo_bike_like` | rectangular | 2.2 × 0.8 m |
| `shuttle_pod_like` | rectangular | 3.0 × 1.4 m |

A circular footprint uses the analytic centerline-minus-radius margin (exact).
A rectangular footprint is sampled along the route at `sample_step_m` (0.1 m)
and a rigid rectangle is oriented along the local tangent at each sample.

## Scenario families

Five self-contained synthetic fixtures (no external map assets needed) that
surface three distinct mechanisms:

| family | mechanism | what it surfaces |
| --- | --- | --- |
| `narrow_passage` | width-driven | a 0.9 m corridor: circular clears, shuttle-pod collides |
| `pedestrian_crossing` | width-driven | pedestrian block offset from centerline |
| `occluded_corner` | turn-overrun length-driven | an elongated body oriented along the incoming leg overruns a turn and pokes a wall a circular body clears |
| `recovery_after_avoidance` | width-driven | recovery-zone block offset from centerline |
| `blocked_path_turn_around` | forward-reach length-driven | an elongated body's forward reach pokes a dead-end wall before the centerline reaches it |

## Run it

```bash
# JSON to stdout
python scripts/diagnostics/run_footprint_orientation_diagnostic.py

# Markdown report to a file
python scripts/diagnostics/run_footprint_orientation_diagnostic.py \
  --format markdown --output footprint_diagnostic.md
```

## API

```python
from robot_sf.nav.footprint_diagnostic import (
    load_footprint_orientation_config,
    parse_footprints,
    parse_diagnostic_parameters,
    build_diagnostic_scenarios,
    build_diagnostic_report,
)

payload = load_footprint_orientation_config(
    "configs/diagnostics/footprint_orientation_v1.yaml"
)
footprints = parse_footprints(payload)
params = parse_diagnostic_parameters(payload)
scenarios = build_diagnostic_scenarios()
report = build_diagnostic_report(
    scenarios, footprints, params["sample_step_m"], params["max_samples"]
)
```

## Relationship to existing route clearance

The camera-ready route-clearance check
(`robot_sf/benchmark/camera_ready/_route_clearance.py`) is a **fail-closed
benchmark gate** for the circular footprint: it refuses to run a route whose
centerline is closer to an obstacle than the robot radius. This diagnostic is
**not** a gate and **not** a modification of that contract. It is a
research/diagnostic tool that adds the missing orientation-aware
(rectangular/elongated) view and compares it against the circular baseline on
the same route. It does not change any benchmark, metric, or schema.

## Limitations (honest caveats)

- The rectangular footprint is oriented along the **local route tangent**. It
  does not model yaw control or the body's actual heading during a turn.
- It does **not** simulate the full swept volume of a body pivoting through a
  corner; consecutive samples are independent oriented rectangles. A corner
  sample therefore orients along an interpolated tangent, not the full
  turn-sweep.
- It samples at a fixed spacing (`sample_step_m`); sub-step features can be
  missed.
- Obstacles are static; there is no dynamic-obstacle or pedestrian modeling.
- Boundary contact is reported as `collision` (conservative); this is a
  diagnostic flag, not a penetration-depth measure.
- `footprint_aware_clearance_m` is `0.0` (not a signed penetration depth) when
  a rectangular footprint overlaps an obstacle.

## Sources (motivation only)

- SE(2) Navigation Mesh — https://arxiv.org/abs/2607.01454
- CoFL-S: Spatially Queryable Sector Flow Fields — https://arxiv.org/abs/2607.02222

Both are referenced as `motivation_only` in the config; this repository does
not implement either method.
