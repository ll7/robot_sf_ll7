# Issue #4759: Event-Disruption Scenario Family

**Status:** prototype — schema draft, experimental ODD-stress proxy

**Issue:** <https://github.com/ll7/robot_sf_ll7/issues/4759>

## Motivation

The July 4, 2025, Waymo fleet disruption in San Francisco exposed a class of
deployment-context stressors that are not captured by ordinary navigation
benchmark scenarios: dense crowd exits, blocked roads, abnormal pedestrian
behavior, unexpected debris, and degraded fallback context (remote assistance
unavailable, limited battery horizon). These are not edge cases in the
traditional sense; they are Operating Design Domain (ODD) boundary crossings
triggered by external events rather than routine interaction patterns.

This scenario family serves as an ODD-stress proxy for testing automated mobile
vehicle (AMMV) / mobile robot behavior under public-event conditions, using
existing 2D simulation primitives.

## Claim Boundary

This is an **experimental ODD-stress proxy**, not a real-world event
reconstruction or robotaxi deployment model. It does not claim or prove
realistic crowd dynamics, traffic-signal realism, remote-assistance modeling,
or planner-ranking changes. The scenarios are architecture-level stress fixtures
for identifying how planners, simulators, and evaluation pipelines behave under
unusual conditions.

## Scenario Dimensions

The `event_disruption` family covers the following disruption dimensions:

| Dimension | Represented | Notes |
|---|---|---|
| Dense pedestrian exit flow | Yes | Bidirectional, high-density, multiple archetypes |
| Blocked path / temporary closure | Yes | `platform_semantics` hazard regions |
| Non-cooperative pedestrians | Yes | Zero-speed blockers, proxemic-hold semantics |
| Unexpected hazard objects | Yes | Rigid obstacle proxies (firework debris) |
| Communication-loss flag | Planned | Metadata-only for first slice |
| Low-battery / limited-fallback horizon | Planned | Metadata-only for first slice |
| Remote assistance unavailable | Yes | Metadata flag |
| Emergency stop / fallback-brake trigger | Metadata | Expected failure-mode contract only |

## Scenario Files

- **Archetype:** `configs/scenarios/single/event_disruption_public_exit.yaml`
- **Manifest:** `configs/scenarios/event_disruption.yaml`

## Scenario Design: `event_disruption_public_exit_blocked_path_v1`

A deterministic template with:

- **Map:** `classic_station_platform` (plaza-like open space)
- **Robot:** Differential-drive, must transit through the disruption zone
- **Pedestrian archetypes:**
  - `exiting_crowd`: higher-speed outbound flow (60% of population)
  - `counter_flow`: lower-speed inbound flow (25%)
  - `non_cooperative`: zero-speed path blockers (15%)
- **Hazard regions** (metadata-only):
  - `temporary_blockage_01`: bbox barricade simulating event closure
  - `hazard_object_01`: polygon debris proxy (rigid obstacle)
- **Expected failure modes:** `fallback_brake`, `stuck`, `near_miss`,
  `excessive_pedestrian_disruption`, `collision_with_hazard`

## Running

```bash
# Load and validate the scenario through the existing loader:
uv run python -m robot_sf.benchmark.cli \
  --scenario configs/scenarios/event_disruption.yaml \
  --max-episodes 1

# Run smoke tests:
uv run pytest tests/scenarios/test_event_disruption_scenarios.py -v
```

## Known Limitations

1. **No real crowd dynamics:** Pedestrian trajectories are authored, not
   generated from a crowd-simulation model.
2. **No remote-assistance model:** The `remote_assistance_available: false`
   flag is metadata-only; there is no remote-assistance protocol in the
   simulator.
3. **Hazard regions are metadata:** The `platform_semantics` regions are
   currently metadata-only and may not be enforced by all planner backends.
   A follow-up issue should track planner-backend enforcement.
4. **No battery or communication model:** Low-battery and communication-loss
   dimensions are not yet represented.

## Follow-Up Work

- Planner-backend enforcement of `platform_semantics` hazard regions
- Continuous-spawn crowd density variants (issue #3813 sustained-flow
  infrastructure may be reusable)
- Communication-loss and low-battery simulation layers
- Benchmark campaign over seeds and planner configurations
- Scenario variants: `crowd_navigation` map (Francis 2023), urban crossing,
  emergency-vehicle interaction

## Sources

- Waymo fleet clogs Presidio after July 4 fireworks:
  <https://abc7news.com/post/waymo-fleet-clogs-presidio-july-4-fireworks-leaving-vehicles-stranded-towed/19455862/>
- Waymo looking into vehicles driving over fireworks in San Francisco:
  <https://www.cbsnews.com/sanfrancisco/news/waymo-fireworks-san-francisco-july-4th/>