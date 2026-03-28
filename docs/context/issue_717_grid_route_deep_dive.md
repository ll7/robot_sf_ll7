# Grid Route Experimental Planner Deep Dive

## Goal

Document `grid_route` as its own experimental planner and record how it behaves across the full
shipped scenario-set surface, not only the original six-scenario static slice.

## Planner Contract

- Planner key: `grid_route`
- Implementation: `robot_sf/planner/grid_route.py`
- Config: `configs/algos/grid_route_camera_ready.yaml`
- Readiness tier: experimental / testing-only opt-in
- Command space: native `unicycle_vw`
- Core idea:
  - build a short 8-connected route over the occupancy grid,
  - pick a local waypoint on that route,
  - track the waypoint with bounded unicycle control,
  - stop or turn in place when immediate frontal clearance collapses.

## Intended Strengths

- static-obstacle detours where pure reactive steering keeps dithering,
- wall-side commitment once topology is visible in the occupancy grid,
- clean failure boundary for obstacle topology problems.

## Known Boundaries

- not paper-facing,
- not a dynamic-social planner,
- not holonomic,
- depends on structured occupancy-grid observations being available,
- currently weakest on tight-gap alignment such as `narrow_passage`.

## Validation Entry Point

- Deep-dive runner:
  - `scripts/validation/run_grid_route_deep_dive.py`
- Output root:
  - `output/validation/grid_route_deep_dive/`

## Interpretation Policy

- Static and atomic scenario sets are the planner's intended proof surface.
- Dynamic and classic crossing sets are still useful as stress tests, but weak results there should
  be interpreted as out-of-scope planner limitations, not as benchmark-ready evidence.
- Validation fixtures that are designed to be invalid should fail closed and be reported as such.

## Result

Assessment: `promising testing-only planner, continue`

The planner is materially stronger than the earlier `safety_barrier` spike on the repository's
atomic scenario surface, but it still has a clean limit: tight-gap navigation and crowd-heavy
crossing scenarios are not solved by the current design.

### Deep-dive artifact

- Runner output:
  - `output/validation/grid_route_deep_dive/iter2/summary.json`
  - `output/validation/grid_route_deep_dive/iter2/summary.md`

### Overall

- sets attempted: `5`
- sets succeeded: `4`
- sets failed: `1`
- executed episodes: `126`
- success rate: `0.6825`
- collision rate: `0.2143`
- termination counts: `86 success`, `27 collision`, `13 max_steps`

### Atomic suite

Manifest: `configs/scenarios/sets/atomic_navigation_minimal_full_v1.yaml`

- scenarios: `23`
- episodes: `69`
- success rate: `0.7536`
- collision rate: `0.1884`

Strong cases:
- all eight empty-map directions: `3/3` success each
- `goal_behind_robot`: `3/3`
- `goal_very_close`: `3/3`
- `small_angle_precision`: `3/3`
- `line_wall_detour`: `3/3`
- `single_obstacle_rectangle`: `3/3`
- `single_ped_crossing_orthogonal`: `3/3`

Mixed cases:
- `single_obstacle_circle`: `2/3`
- `corner_90_turn`: `2/3`
- `corridor_following`: `2/3`
- `symmetry_ambiguous_choice`: `2/3`
- `start_near_obstacle`: `1/3`
- `u_trap_local_minimum`: `1/3`

Failing cases:
- `narrow_passage`: `0/3`, all collision
- `head_on_interaction`: `0/3`, all max-steps
- `overtaking_interaction`: `0/3`, `2 collision`, `1 max_steps`

Interpretation:
- `grid_route` is strong on topology-aware static navigation and simple nominal progress.
- It is still weak when the route must thread a narrow opening or when dynamic social interaction
  dominates the problem.

### Verified-simple subset

Manifest: `configs/scenarios/sets/verified_simple_subset_v1.yaml`

- scenarios: `10`
- episodes: `30`
- success rate: `0.6667`
- collision rate: `0.2000`

Interpretation:
- The planner keeps the same strengths as the atomic suite.
- The drop relative to the static-only slice comes entirely from dynamic interaction scenarios and
  the same unresolved `narrow_passage` case.

### Static six-scenario slice

Manifest: `configs/scenarios/sets/safety_barrier_static_slice_v1.yaml`

- scenarios: `6`
- episodes: `18`
- success rate: `0.7778`
- collision rate: `0.2222`

Per-scenario summary:
- `empty_map_8_directions_east`: `3/3`
- `goal_behind_robot`: `3/3`
- `single_obstacle_rectangle`: `3/3`
- `line_wall_detour`: `3/3`
- `single_obstacle_circle`: `2/3`
- `narrow_passage`: `0/3`, all collision

Interpretation:
- The earlier `14/18` result was stable and not an artifact of a lucky run selection.
- `narrow_passage` remains the clean acceptance gate for the next planner iteration.

### Classic crossing subset

Manifest: `configs/scenarios/sets/classic_crossing_subset.yaml`

- scenarios: `3`
- episodes: `9`
- success rate: `0.0000`
- collision rate: `0.4444`

Interpretation:
- This planner should still be treated as static-obstacle-first.
- The deep dive confirms that it does not generalize to crowd-heavy crossing behavior just because
  it has a stronger occupancy-grid route primitive.

### Validation fixture behavior

Manifest: `configs/scenarios/sets/atomic_navigation_validation_fixtures_v1.yaml`

- scenarios: `1`
- status: fail-closed `partial-failure`
- reason: `Failed to sample 1 points in zone without obstacle overlap after 20 attempts.`

Interpretation:
- This is the correct outcome.
- The invalid fixture is designed to prove that unusable maps fail closed; `grid_route` does not
  mask that contract.

## Recommendation

Keep `grid_route` as the active experimental continuation path.

Why:
- it materially outperforms `safety_barrier` on the intended static/atomic surface,
- it solves wall-side commitment and most simple static detours,
- and its remaining failure boundary is narrow and concrete.

Do not broaden it into paper-facing benchmark claims yet.

Required next acceptance gate:
- improve `narrow_passage` without regressing the current atomic-suite strengths.

Dynamic/crowd limitation:
- treat `classic_crossing_subset`, `head_on_interaction`, and `overtaking_interaction` as evidence
  that the current planner is not a social-navigation planner.

## Validation

### Canonical benchmark matrix result

Config:
- `configs/benchmarks/classic_interactions_francis2023_grid_route_experimental.yaml`

Campaign root:
- `output/benchmarks/camera_ready/classic_interactions_francis2023_grid_route_experimental_issue717_grid_route_matrix_20260328_163532/`

Headline result on `configs/scenarios/classic_interactions_francis2023.yaml`:
- episodes: `141`
- success: `0.0071`
- collisions: `0.0922`
- SNQI: `-0.2610`
- runtime: `62.9 s`
- throughput: `2.2409 eps/s`
- benchmark availability: `available`
- SNQI contract: `fail`

Interpretation:
- `grid_route` is much faster than several other experimental planners on this surface.
- But it is effectively not goal-reaching on the canonical benchmark matrix.
- The low collision rate does not rescue it, because the dominant behavior is timeout / stalled
  non-completion rather than strong socially competent navigation.
- This means the atomic/static wins do not generalize to the main classic/francis2023 benchmark.

Reference context against nearby experimental planners on the same matrix:
- `prediction_planner_v2_full`: success `0.0567` to `0.0780`, collisions `0.1986` to `0.2270`
- `prediction_planner_v2_xl_ego`: success `0.0567` to `0.0709`, collisions `0.2128` to `0.2340`
- `risk_dwa`: success `0.0142`, collisions `0.2908`
- `stream_gap`: success `0.0000`, collisions `0.0213`

Interpretation:
- `grid_route` beats `stream_gap` on success and `risk_dwa` on collision rate.
- It does not beat the prediction-planner baselines on goal-reaching.
- Its current position is: fast, low-collision, but still too incomplete to be considered a
  strong benchmark planner.

Weakness pattern from the canonical matrix artifacts:
- almost every family stays at `0.0` success,
- only `francis2023_entering_elevator` reaches `0.3333`,
- repeated weak families include:
  - `bottleneck`
  - `crossing`
  - `doorway`
  - `head_on_corridor`
  - `overtaking`
  - `robot_overtaking`
  - `robot_crowding`
- `near_miss` exposure remains high in several social families even without formal collisions:
  - `leave_group`
  - `robot_crowding`
  - `group_crossing`
  - `exiting_elevator`

Takeaway:
- `grid_route` currently behaves like a topology-aware static route follower that can avoid some
  collisions, but it does not solve social interaction or time-critical passage negotiation.
- The planner is still worth continuing as an experimental branch, but not as a candidate for
  benchmark promotion.

## Validation

- `uv run python -m py_compile scripts/validation/run_grid_route_deep_dive.py`
- `uv run python scripts/validation/run_grid_route_deep_dive.py --output-dir output/validation/grid_route_deep_dive/iter2`
- `uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/classic_interactions_francis2023_grid_route_experimental.yaml --mode preflight --label issue717_grid_route_matrix`
- `uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/classic_interactions_francis2023_grid_route_experimental.yaml --label issue717_grid_route_matrix`
