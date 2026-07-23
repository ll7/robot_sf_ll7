# Temporary Construction-Corridor Scenarios

[Back to Documentation Index](../README.md) | [Scenario Zoo](scenario_zoo/index.md)

## Purpose

This document describes the **temporary construction-corridor scenario family**, a set of
ODD/context stress proxies for AMMV (Autonomous Mobility-as-a-Service Vehicle) planning under
temporary infrastructure constraints. These scenarios model corridor narrowing, detour, and
zigzag barrier layouts using existing SVG obstacle infrastructure — no runtime code changes are
required.

## ODD/Context Stressor Distinction

These scenarios are **ODD/context stressors**, not real-world event reconstructions:

- **ODD stressor**: The scenario perturbs the robot's operational design domain by introducing
  temporary infrastructure (construction barriers) that narrows or redirects the navigable
  corridor. The robot must adapt its path planning and clearance behavior to the constrained
  geometry.
- **Context stressor**: The scenario tests the robot's ability to distinguish temporary
  construction infrastructure from permanent walls. The `platform_semantics` metadata annotates
  which obstacles represent temporary/construction barriers (vs. permanent boundary walls),
  enabling downstream systems to reason about infrastructure permanence.
- **Not a real-world event**: These are synthetic 2D fixtures designed for architectural stress
  testing. They do not prove or claim realistic construction-site dynamics or robotaxi
  deployment equivalence.

## Source

Source: WorkDrive AMMV ODD stress scenarios. See
[Issue #6055](https://github.com/ll7/robot_sf_ll7/issues/6055).

## Scenario Family

The family contains three atomic scenarios, each modeling a distinct temporary construction
corridor layout:

| Scenario | Map | Layout | Temporary barriers |
| --- | --- | --- | --- |
| `construction_corridor_narrowing` | `construction_corridor_narrowing.svg` | Corridor narrows from width 10 to width 4 in the central section | 2 (top and bottom narrowing walls) |
| `construction_corridor_detour` | `construction_corridor_detour.svg` | Central barrier blocks the direct path, forcing a detour through a narrow gap | 1 (central blockage) |
| `construction_corridor_zigzag` | `construction_corridor_zigzag.svg` | Two zigzag barriers force a serpentine path through alternating corridor gaps | 2 (alternating deflection barriers) |

### Corridor geometries

All maps use a 20x20 viewBox with a corridor running left-to-right. The outer boundary walls
are permanent; the internal barriers are temporary construction infrastructure.

- **Narrowing**: The wide corridor (y=5..15, width 10) narrows to width 4 (y=8..12) in the
  central section (x=8..12). The robot must center itself and traverse the constriction.
- **Detour**: A central barrier (x=8..12, y=7..13) blocks the direct path. The robot must
  detour through the narrow gap above the barrier (y=4..7, width 3).
- **Zigzag**: Two barriers create a serpentine path. Barrier 1 (x=6..10, y=4..10) extends from
  the top wall; barrier 2 (x=10..14, y=10..16) extends from the bottom wall. The robot must
  navigate alternating upper and lower corridor gaps.

## Platform Semantics

Each scenario includes `platform_semantics` metadata that annotates which obstacles represent
temporary construction infrastructure. This distinguishes temporary barriers from permanent
walls, enabling downstream systems to reason about infrastructure permanence.

Each temporary barrier region is annotated with:

- `kind: hazard` — marks the region as a hazard/obstacle.
- `shape: bbox` — bounding box shape.
- `bounds: [min_x, min_y, max_x, max_y]` — bounding box coordinates.
- `metadata.infrastructure_type: temporary_construction_barrier` — identifies the obstacle as
  temporary construction infrastructure.
- `metadata.permanent: false` — explicitly marks the obstacle as non-permanent.
- `metadata.barrier_role` — describes the barrier's function (e.g., `corridor_narrowing`,
  `path_blockage`, `zigzag_deflection`).
- `metadata.hazard_class: rigid_obstacle` — classifies the hazard type.

## Metrics

All required metrics are reported by the existing metric pipeline without code changes:

- **collision** — robot contact with temporary barriers or permanent walls.
- **near_miss** — close interactions without contact, visible in proximity metrics.
- **clearance** — minimum distance to obstacles (critical for narrow passages).
- **stuck** — robot unable to make progress in constrained geometry.
- **path_efficiency** — path elongation from detours and zigzag routing.
- **fallback** — robot invokes fallback behavior when navigation fails.

## Configuration

The scenario archetype is defined in
[configs/scenarios/archetypes/temporary_construction_corridor.yaml](../configs/scenarios/archetypes/temporary_construction_corridor.yaml).

A manifest that includes all three scenarios is available at
[configs/scenarios/sets/temporary_construction_corridor_v1.yaml](../configs/scenarios/sets/temporary_construction_corridor_v1.yaml).

Maps are registered in
[maps/registry.yaml](../maps/registry.yaml) under the map IDs
`construction_corridor_narrowing`, `construction_corridor_detour`, and
`construction_corridor_zigzag`.

## Validation

```bash
# Validate SVG maps parse correctly
uv run python -c 'from robot_sf.nav.svg_map_parser import convert_map; convert_map("maps/svg_maps/construction_corridor_narrowing.svg")'
uv run python -c 'from robot_sf.nav.svg_map_parser import convert_map; convert_map("maps/svg_maps/construction_corridor_detour.svg")'
uv run python -c 'from robot_sf.nav.svg_map_parser import convert_map; convert_map("maps/svg_maps/construction_corridor_zigzag.svg")'

# Validate scenario archetype loads and passes schema validation
uv run python -c 'from robot_sf.training.scenario_loader import load_scenarios; from robot_sf.benchmark.scenario_schema import validate_scenario_list; scenarios = load_scenarios("configs/scenarios/archetypes/temporary_construction_corridor.yaml"); assert len(scenarios) > 0; errors = validate_scenario_list(scenarios); assert not errors'
```

## Claim Boundary

These scenarios are **draft ODD/context stress proxies**, not certified benchmark evidence.
They are intended for planner exploration and stress testing. Before use in planner ranking or
benchmark interpretation, they require runner integration and executed benchmark evidence.
Fallback or degraded execution of these scenarios does not count as successful evidence.
