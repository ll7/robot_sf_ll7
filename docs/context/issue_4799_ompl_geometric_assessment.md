# Issue #4799: OMPL Geometric Route Oracle Assessment

**Date:** 2026-07-07
**Status:** Assessment complete — recommend go for optional diagnostic adapter.
**Claim status:** Diagnostic-only; not benchmark evidence.

## Shortlist and Recommendation

Shortlist from issue #4799: RRTConnect, BITstar, RRTstar, InformedRRTstar, PRMstar, SPARS/SPARS2.

**Recommendation: Go** — but only as an optional diagnostic tool, not as a core planner.
BITstar is the most useful single candidate for Robot SF.

## Dependency Feasibility

| Aspect | Finding |
| --- | --- |
| Package | `ompl` PyPI wheel (v2.0.1), nanobind-based Python bindings |
| Install | `uv pip install ompl` (not in pyproject.toml; optional only) |
| Availability | Works on Linux x86_64; Windows/Mac ARM status unknown |
| Caveats | Nanobind resource-leak warnings at interpreter exit (benign for offline diagnostics) |
| Missing | `STRRTstar` (Space-Time RRT*) is not exposed in the PyPI wheel |

## Comparison Against ClassicGrid Planner (Theta*V2)

Smoke test on `classic_bottleneck.svg`, start=(20.0, 31.0), goal=(20.0, 8.0):

| Planner | Path length (m) | Planning time (s) | Waypoints | Optimal? |
| --- | --- | --- | --- | --- |
| Classic (Theta*V2) | 23.00 | ~0.9 (includes import overhead) | 2 | Yes |
| OMPL BITstar | 23.00 | ~0.001 | 50 | Yes |
| OMPL RRTConnect | 31.09 | ~0.001 | 50 | No (+35%) |
| OMPL RRTstar | 23.00 | ~5.0 (full budget) | 50 | Yes (slow convergence) |

### Key observations

1. **BITstar** reaches optimality orders of magnitude faster than RRTstar and is
   faster than the grid planner once imports are amortized — making it the most
   useful single candidate for route-quality diagnostics.

2. **RRTConnect** finds a feasible path quickly but the route is suboptimal
   (+35% longer), limiting its standalone diagnostic value.

3. **RRTstar** converges to the optimal solution asymptotically but is slow on
   typical Robot SF maps — less useful than BITstar for the same purpose.

4. **ClassicTheta*V2** already produces optimal paths with minimal waypoints,
   so OMPL does not replace it; the value is in *comparison*.

## Utility Assessment

### Where OMPL adds value

- **Route-quality annotations**: BITstar produces continuous-space routes whose
  length and clearance can be compared against grid-resolved paths, surfacing
  grid-resolution artifacts the A*/Theta* planner might miss at coarse resolution.

- **Clearance diagnostics**: The obstacle union approach (shapely-based validity
  checker) makes it straightforward to evaluate minimum clearance from obstacles
  without needing to rasterize to a grid.

### Where OMPL does NOT add value

- Dynamic obstacle avoidance (issue explicitly out of scope).
- Replacing the grid planner — the grid planner is fast, optimal for its
  resolution, and already integrated.
- Kinodynamic planning — OMPL geometric planners ignore dynamics.

## Deliverables

- `robot_sf/planner/ompl_geometric_adapter.py`: Thin adapter exposing OMPL planners
  against `MapDefinition`, with `shapely`-based obstacle checking.
- `tests/test_planner/test_ompl_geometric_adapter.py`: Unit and integration tests,
  with graceful skipping when OMPL is not installed.
- This context note.

## Decision

**Adopt as optional diagnostic.** The adapter is useful for:
1. Quantifying route-length differences between continuous-space and grid routes
2. Detecting grid-resolution artifacts at low resolution settings
3. Providing route-quality annotations for benchmark scenarios

Do NOT integrate into the runtime planner pipeline. Keep as an offline diagnostic
tool importable only when the user installs `ompl` manually.

## Validation Commands

```bash
# Unit tests (no OMPL required)
uv run pytest tests/test_planner/test_ompl_geometric_adapter.py -v -k "not slow"

# Integration tests (OMPL required)
uv run pytest tests/test_planner/test_ompl_geometric_adapter.py -v

# Lint
uv run ruff check robot_sf/planner/ompl_geometric_adapter.py tests/test_planner/test_ompl_geometric_adapter.py
```