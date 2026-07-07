# Issue #4795: Centralized MAPF Oracle Assessment

Issue: [#4795](https://github.com/ll7/robot_sf_ll7/issues/4795)
Date: 2026-07-07
Author: codex-harness/SAIA-Qwen3.6-27B cheap-lane worker

## Summary

Assessed whether the centralized MAPF algorithms from
[atb033/multi_agent_path_planning](https://github.com/atb033/multi_agent_path_planning)
(SIPP, CBS, TPG) are useful for `robot_sf_ll7` as offline diagnostic oracles.
**Recommendation: implement a small diagnostic script.**

## Prior Investigation

- [#725](https://github.com/ll7/robot_sf_ll7/issues/725) /
  [PR #731](https://github.com/ll7/robot_sf_ll7/pull/731): concept-inspired
  `nmpc_social` local planner (decentralized, from the same upstream repo).
- [#726](https://github.com/ll7/robot_sf_ll7/issues/726) /
  [PR #727](https://github.com/ll7/robot_sf_ll7/pull/727): HRVO local planner
  baseline with secondary provenance in
  `configs/algos/hrvo_camera_ready.yaml`.
- `robot_sf/benchmark/algorithm_metadata.py` references the upstream repo for
  HRVO provenance only.

## Repo Search Results

No existing SIPP, CBS, or MAPF implementation or issue found. The upstream repo
is only referenced as secondary provenance for the decentralized HRVO adapter.
The `diagnostic` baseline category in `algorithm_metadata.py` covers planning
diagnostics but no centralized MAPF entry exists.

## Upstream Inventory

| Algorithm | Path | Size | License | What It Does |
|-----------|------|------|---------|--------------|
| SIPP | `centralized/sipp/sipp.py` | ~4.5 KB | MIT | Safe Interval Path Planning — A* with time-window feasibility under dynamic obstacles |
| CBS | `centralized/cbs/cbs.py` | ~12 KB | MIT | Conflict-Based Search — tree search for multi-agent conflict resolution |
| TPG | `centralized/scheduling/tpg.py` | ~6.5 KB | MIT | Temporal Plan Graph — post-process CBS plans with kinematic scheduling |

All MIT-licensed. Source is small, readable Python with YAML input/output.

## Fit Assessment

### Why These Algorithms Are Orthogonal to Current Planners

Current planners (`hrvo`, `orca`, `social_force`, `nmpc_social`, `dwa`, `teb`,
`ppo`, etc.) are **decentralized local social-navigation policies**:

- Each agent acts on local observations.
- Social compliance is continuous and reactive.
- Pedestrians are not controlled by the robot.

SIPP/CBS/TPG are **centralized multi-agent path-finding algorithms** for
cooperative agents:

- Global map and all agent states are known.
- Discrete grid/world representation.
- All agents are cooperative and follow plans.

They solve a different question: "is there a globally feasible schedule?" vs.
"how does this policy react locally?"

### Diagnostic Value

As **offline oracles**, they can answer:

1. **Route feasibility**: Is a path from start to goal even possible given
   static obstacles? (Basic A* on occupancy grid)
2. **Time-window feasibility**: Does a time-reserved path exist given known
   obstacle trajectories? (SIPP)
3. **Multi-agent conflict diagnosis**: Where and when do agent plans conflict?
   (CBS — deferred, higher complexity)
4. **Oracle path metrics**: Shortest path length, required wait steps,
   schedule slack. (TPG post-processing)

This separates **route infeasibility** from **local interaction failure**,
which is the core gap the issue identifies.

## Recommendation

**Implement a small diagnostic script** (Phase 1 + Phase 3 from the issue).

### What Is Delivered

A standalone script `scripts/tools/mapf_oracle_diagnostic.py` that:

1. Parses an SVG map file into a coarse occupancy grid (squares/rectangles
   in the `obstacles` group become occupied cells).
2. Runs a SIPP-style single-agent path planner on the static occupancy grid
   (no dynamic obstacles in v1 — that is the Phase 4 extension).
3. Emits diagnostic JSON with:
   - `mapf_feasible`: whether a path exists
   - `oracle_path_length`: number of steps in the shortest path
   - `oracle_path`: list of grid coordinates
   - `grid_dimensions`: (rows, cols)
   - `occupancy_ratio`: fraction of cells occupied
   - `start_grid` / `goal_grid`: discretized start/goal positions
   - `diagnostic_status`: "feasible", "infeasible", or "degenerate"

### What Is Deferred

- ~~Dynamic obstacle time-windows~~ → implemented (PR #4809): SIPP search.
- ~~Multi-agent CBS conflict resolution~~ → implemented (this PR): CBS search.
- TPG schedule post-processing (requires CBS output).
- Integration into the benchmark campaign loop (requires follow-up issue).

### Why Not Code Reuse from Upstream

The upstream SIPP implementation is ~100 lines and depends on its own
`SippGraph`/`State` classes and YAML input format. A clean reimplementation
for the `robot_sf_ll7` SVG-map format is simpler than adapting the upstream
code, and avoids dependency surface concerns. The algorithm is well-known
(Safe Interval Path Planning, Li et al. ICRA 2011) and the static-obstacle
variant reduces to standard A*.

## Provenance

- Upstream repo: https://github.com/atb033/multi_agent_path_planning
- License: MIT
- Original authors: Ashwin Bose (@atb033)
- SIPP paper DOI: 10.1109/ICRA.2011.5980306
- CBS paper: Sharon et al. AAAI 2009
- This diagnostic is a clean-room reimplementation, not a copy.

## Validation

```bash
cd /home/luttkule/git/robot_sf_ll7
uv run python scripts/tools/mapf_oracle_diagnostic.py \
  maps/svg_maps/classic_crossing.svg \
  --start 1 1 --goal 38 38 --grid-size 40

uv run pytest tests/scripts/tools/test_mapf_oracle_diagnostic.py -v

uv run ruff check scripts/tools/mapf_oracle_diagnostic.py tests/scripts/tools/test_mapf_oracle_diagnostic.py
```

## Links

- Issue: [#4795](https://github.com/ll7/robot_sf_ll7/issues/4795)
- External planner reuse checklist: `docs/context/external_planner_reuse_checklist.md`
- Planner zoo context: `docs/ai/planner_zoo_context.md`
