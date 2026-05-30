# Issue #1675 Learned Risk-Surface Interface

Date: 2026-05-30

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1675>

## Scope

This note defines the first Robot SF-native local risk/potential surface contract inspired by the
Dyn-NPField pattern identified in issue #1617. It does not import upstream code, load a learned
checkpoint, or create benchmark-strength evidence. The implementation is a deterministic fixture
producer plus an adapter that projects the surface into the existing occupancy-grid contract.

## Contract

The code surface is `robot_sf.planner.learned_risk_surface`.

* `LocalRiskSurfaceSpec` defines the frame, resolution, size, origin, threshold, and producer ID.
* `LocalRiskSurface` stores a normalized `float32` grid with shape `(height_cells, width_cells)`.
* Frame convention is ego-frame only: `+x` forward, `+y` left, robot at `(0, 0, heading=0)`.
* Resolution is metres per cell; the default surface is 4 m by 4 m at 0.25 m/cell.
* Values are normalized to `[0, 1]`; invalid shape, non-finite values, unsupported frames, or
  invalid geometry raise `RiskSurfaceUnavailable`.
* `occupancy_grid()` exposes the risk surface as obstacle and combined channels so existing
  occupancy-aware planners can consume it without hidden fallback behavior.

## Smoke Path

`RiskSurfacePlannerAdapter` is the deterministic produce-and-consume smoke path:

1. `deterministic_pedestrian_risk_surface(...)` builds a Gaussian local risk field from structured
   Robot SF pedestrian positions.
2. `attach_risk_surface_to_observation(...)` adds `occupancy_grid`, `occupancy_grid_meta`, and
   `local_risk_surface_diagnostics`.
3. The wrapped `RiskDWAPlannerAdapter` consumes the surface through the normal
   `OccupancyAwarePlannerMixin` helpers.
4. Missing robot state, invalid pedestrian data, or malformed surface metadata fails closed to a
   `(0.0, 0.0)` command with `availability_status=not_available`.

This proves the interface can be produced and consumed locally. It is not a claim that a learned
surface improves navigation.

## Benchmark Boundary

Before this interface can support benchmark-strength evidence, a follow-up must add:

* model provenance and license review for any learned producer,
* training/evaluation config lineage and normalization metadata,
* deterministic inference and device handling,
* a scenario matrix that actually exercises pedestrian and obstacle risk,
* per-step diagnostics showing raw surface values, planner consumption, and guard/fallback status,
* comparison against existing local planners without counting fallback or degraded execution as
  success.

The adapter diagnostics keep `benchmark_strength=false` until those requirements are met.

## Validation

Targeted validation for this implementation:

```bash
uv run pytest -q tests/planner/test_learned_risk_surface.py
git diff --check origin/main...HEAD
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```
