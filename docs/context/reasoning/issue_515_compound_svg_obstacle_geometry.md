# Issue 515 Compound SVG Obstacle Geometry Reasoning

Issue: [#515](https://github.com/ll7/robot_sf_ll7/issues/515)

## Goal

Preserve the full geometry of compound SVG obstacle paths instead of collapsing them to a single
ring. The target semantics are:

- detached obstacle members remain separate,
- hole/exterior structure is preserved,
- legacy single-ring maps still work without schema churn.

## Decision

Use a compound-geometry model as the source of truth and keep `Obstacle` as a compatibility
adapter.

Concrete shape of the change:

- `robot_sf/nav/obstacle.py` now carries Shapely geometry in addition to the legacy `vertices`
  surface.
- `robot_sf/nav/svg_map_parser.py` preserves all polygon members recovered by `make_valid(...)`
  instead of selecting only the largest polygon.
- `robot_sf/nav/occupancy_grid.py`, `robot_sf/gym_env/robot_env.py`, and
  `robot_sf/nav/navigation.py` now consume polygon geometry directly.

## Why This Option

An adapter-only fix would still force the repo to choose between hole support, detached members,
and legacy vertex lists on a call-site-by-call-site basis. That keeps the semantics split and makes
future regressions likely.

Preserving the compound geometry once, then adapting where needed, keeps the contract consistent:

- parser output stays faithful,
- occupancy rasterization can respect holes,
- planners see the same obstacle shape the map author described,
- old code can still inspect `Obstacle.vertices` for a representative ring.

## Validation Path

Planned/targeted checks for this work:

```bash
uv run pytest tests/test_svg_obstacle_self_intersection.py tests/test_occupancy_polygon_fill.py tests/test_svg_classic_maps_format.py
```

Broader readiness gate:

```bash
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

## Current Boundary

This note covers the issue-515 geometry decision and its immediate consumers only.

Deferred follow-up scope, if needed:

- broader plot/inspection polish for holes in rendered maps,
- serialization updates for any downstream artifact that should persist compound geometry explicitly.
