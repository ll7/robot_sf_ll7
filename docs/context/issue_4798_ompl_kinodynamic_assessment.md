# Issue #4798 — OMPL Kinodynamic Route Diagnostics Assessment

**Date:** 2026-07-07
**Status:** Assessment complete; smoke diagnostic implemented.
**Claim status:** Diagnostic-only. No benchmark or paper-facing claims.

## Summary

OMPL (Open Motion Planning Library) control-based planners can be used as an
optional kinodynamic-route diagnostic in robot_sf_ll7. The pip package `ompl`
(version 2.0.x) installs cleanly and provides Python bindings for control-based
sampling planners (RRT, SST, KPIECE).

## How this differs from existing planners

| Planner class | Planning type | Motion model | Pedestrian awareness |
|---|---|---|---|
| A\*/Theta\* (`classic_global_planner`) | Grid-based, holonomic | None (free movement in 2D) | No |
| ORCA / Social Force (local) | Reactive sampling | Differential-drive at control layer | Yes |
| OMPL diagnostic (this module) | Sampling-based, kinodynamic | Differential-drive with bounds | No |

A*/Theta\* routes are not constrained by the robot's turning radius or speed
limits. The OMPL diagnostic checks whether a proposed route segment is
feasible under differential-drive dynamics. It is not a replacement for local
planners — it is a diagnostic for route quality assessment.

## Dependency assessment

- **Package:** `ompl==2.0.1` (PyPI)
- **Install:** `pip install ompl` or `uv pip install ompl`
- **Import:** `import ompl.base; import ompl.control`
- **Size:** ~4.4 MiB download
- **System deps:** None beyond standard Python (bindings are pure-Python wrappers)
- **Fail-closed:** Module uses lazy imports; graceful degradation when absent.

## Implementation

- **Module:** `robot_sf/planner/ompl_smoke.py`
- **Public API:**
  - `check_ompl_available()` → `(bool, str | None)`
  - `smoke_plan(start, goal, config, obstacle_polygons)` → `OmplSmokeResult`
  - `compare_with_classic_route(ompl_result, classic_path)` → diagnostics dict
- **State space:** SE(2) — `(x, y, theta)` with differential-drive propagation
- **Default planner:** RRT (Rapidly-exploring Random Trees)
- **Collision checking:** Shapely Polygon buffering (optional)

## Out of scope

- Dynamic pedestrian prediction
- Integration as the primary global planner
- Benchmark campaigns or paper-facing claims
- Mandatory OMPL dependency

## Validation

```bash
uv run pytest tests/planner/test_ompl_smoke.py -v
uv run ruff check robot_sf/planner/ompl_smoke.py
```
