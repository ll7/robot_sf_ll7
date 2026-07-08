# Issue #4797 OMPL ST-RRT* Feasibility Assessment 2026-07-07

Date: 2026-07-07

Related issue:

- [robot_sf_ll7#4797](https://github.com/ll7/robot_sf_ll7/issues/4797): Assess OMPL ST-RRT* as a space-time route-feasibility oracle

Related context:

- [external_planner_reuse_checklist.md](external_planner_reuse_checklist.md)
- [pyproject.toml](../../pyproject.toml)

## Goal

Assess whether OMPL's Space-Time RRT* (`STRRTstar`) should be integrated into `robot_sf_ll7` as an offline route-feasibility diagnostic for dynamic-obstacle scenarios.

This note is a fail-closed dependency and feasibility spike assessment. It does not implement a new planner and does not add OMPL to core dependencies.

## Feasibility Spike Results (Phase 1)

A feasibility spike was conducted in the project environment to verify package installation and `STRRTstar` availability:

1. **Installation:** We attempted installing the official Python bindings using `uv pip install ompl` inside the virtual environment (Python 3.13.13). The package resolved and installed successfully (`ompl==2.0.1` from PyPI).
2. **Import Verification:** Importing `ompl` and its submodules (e.g., `ompl.base`, `ompl.geometric`, `ompl.control`) works without throwing runtime link errors.
3. **STRRTstar Binding Check:** We inspected the attributes of `ompl.geometric` and `ompl._ompl.geometric` to check if `STRRTstar` was exposed:
   ```python
   from ompl import geometric
   print(dir(geometric))
   ```
   *Result list:* `['AORRTC', 'BFMT', 'BITstar', 'BKPIECE1', 'Discretization', 'DiscretizationCellData', 'FMT', 'InformedRRTstar', 'KPIECE1', 'LBKPIECE1', 'PRM', 'PRMstar', 'PathGeometric', 'PathSimplifier', 'RRT', 'RRTConnect', 'RRTstar', 'SORRTstar', 'SimpleSetup', ...]`

   The class `STRRTstar` is **missing** from the pre-built `ompl` wheel.

### Root Cause and Dependency Constraints

While the OMPL C++ library implements `STRRTstar` (Space-Time RRT*), the standard Python bindings distributed on PyPI do not bind or expose it. Exposing `STRRTstar` would require:
- Checking out OMPL source code;
- Modifying the Nanobind/Py++ binding configuration files in the `py-bindings` directory to expose `ompl::geometric::STRRTstar` and `ompl::base::SpaceTimeStateSpace`;
- Recompiling the OMPL library from source with native C++ compiler tools, Boost, Eigen, and CMake.

Compiling OMPL from source and maintaining custom-built wheels goes against the project's requirement to keep core dependencies lightweight and standard. Building it dynamically on developer/CI environments would destabilize the `uv` package-installation workflow. Therefore, using OMPL `ST-RRT*` via Python bindings is **not feasible** under the current repository constraints.

## Orthogonality

### Orthogonality to Static Global Planners (A* / Theta*)
- Existing global planners in `robot_sf_ll7` (such as A* and Theta* from `python-motion-planning`) operate on a purely spatial static-map grid. They compute a geometric path in 2D space `(x, y)` and do not account for time `t` or moving obstacles.
- A space-time route-feasibility planner plans directly in state space `(x, y, t)` under velocity bounds. It can determine if a collision-free path exists that navigates *around* time-indexed pedestrian occupancy tubes.

### Orthogonality to Local Planners / Controllers
- Local planners (e.g., TEB, MPPI, DWA, MPC) react to local obstacle predictions in real time. They frequently fail or get trapped in local minima (deadlocks) in highly constrained scenarios (like narrow corridor crossings).
- A space-time oracle operates offline with full future scenario knowledge. It computes a global solution over the space-time volume to distinguish between:
  - **Local policy failure:** The scenario is solvable, but the local controller failed to find the solution.
  - **Scenario infeasibility:** No collision-free space-time route exists under the scenario's boundaries and agent dynamics.

## Verdict

Current tier: **Rejected/Deferred (not feasible due to missing bindings).**

1. **No Dependency Changes:** Do not add `ompl` to core or optional dependencies in `pyproject.toml` since the target planner class `STRRTstar` is unavailable in the standard distribution.
2. **Defer implementation:** Reject building a custom OMPL source binding compiler inside the Robot SF pipeline.
3. **No Claims:** No benchmark, metric, or dissertation claim is made from this assessment.

## Alternative Paths

If a space-time route-feasibility oracle is required in the future, the following options are recommended:
- **Python-native Space-Time A*/Theta*:** Implement a simple space-time A* grid planner directly in Python/Numba inside `robot_sf/benchmark/diagnostics/` or extend `python-motion-planning`. This avoids heavy C++ compiled bindings.
- **Standalone C++ Binary Wrapper:** Write a minimal C++ program using native OMPL `STRRTstar` that reads scenario JSON files, solves the path, and outputs JSON solution files. Run this binary as a subprocess diagnostic tool instead of importing OMPL into the Python virtual environment.

## Validation

Validation was conducted using source/provenance inspection of PyPI `ompl==2.0.1` and namespace attribute reflection:
- `import ompl` succeeds, but `ompl.geometric` lacks `STRRTstar`.
- `from ompl import _ompl` succeeds, and `_ompl.geometric` contains only `['AORRTC', 'InformedRRTstar', 'RRT', 'RRTConnect', 'RRTstar', 'SORRTstar']`.
