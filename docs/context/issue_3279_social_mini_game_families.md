# Issue #3279 — Social Mini-Game scenario families (v0)

**Status:** current · **Freshness:** dated (2026-06-22) · **Evidence tier:** proposal / diagnostic-smoke

## What landed

A first runnable cut of the Social Mini-Game local-navigation scenario families called out in
the research report, implemented as *parameterized generated* scenarios on the existing
`robot_sf.benchmark.scenario_generator.generate_scenario` path (no new generator engine).

- Config: [`configs/scenarios/sets/issue_3279_social_mini_game_families_v0.yaml`](../../configs/scenarios/sets/issue_3279_social_mini_game_families_v0.yaml)
- Test: `tests/benchmark/test_social_mini_game_families_issue_3279.py`

Each family entry is a deterministic fixture (fixed `id` + `seeds: [3279]`) **and** a
parameterized generator path (consumed by `generate_scenario`); the manifest records a
mechanism label, the generator parameters, and the Issue #3423
`metadata.social_mini_game_controls` block in `metadata`.

## Covered mechanisms (this cut)

| Family | mechanism_aware_suite_id | Generator mapping | Issue #3423 controls |
| --- | --- | --- | --- |
| doorway | `doorway_bottleneck_negotiation` | `obstacle=bottleneck`, `flow=bi` | narrow `width_m`, doorway-bottleneck `occlusion_geometry`, fixed-seed `start_timing_s`, medium `yielding_pressure` |
| hallway | `hallway_bidirectional_passing` | `obstacle=open`, `flow=bi` | open-corridor `width_m`, no static `occlusion_geometry`, fixed-seed `start_timing_s`, medium `yielding_pressure` |
| intersection | `intersection_crossing_negotiation` | `obstacle=open`, `flow=cross` | open-crossing `width_m`, no static `occlusion_geometry`, fixed-seed `start_timing_s`, high `yielding_pressure` |
| blind_corner | `blind_corner_occlusion_exposure` | `obstacle=maze`, `flow=uni` | narrow L-corner `width_m`, documented `l_corner_blind_corner` `occlusion_geometry`, fixed-seed `start_timing_s`, medium `yielding_pressure` |
| crowded_traffic | `crowded_traffic_merge_negotiation` | `density=high`, `flow=merge`, `groups=0.2` | open-merge `width_m`, no static `occlusion_geometry`, fixed-seed `start_timing_s`, high `yielding_pressure` |

## Issue #3423 Control Exposure

The manifest now exposes `width_m`, `occlusion_geometry`, `start_timing_s`, and
`yielding_pressure` as first-class Social Mini-Game metadata under
`metadata.social_mini_game_controls`. These controls are documented equivalents
for the existing generated-scenario vocabulary
(`density`/`flow`/`obstacle`/`groups`/`speed_var`/`goal_topology`/`robot_context`),
not a new generator engine or behavioral certification layer.

`blind_corner` no longer relies on an unnamed coarse proxy: its control metadata
declares `occlusion_geometry: l_corner_blind_corner`, and the test suite checks
that the generated `maze` obstacle path contains both vertical and horizontal
occluder segments suitable for the documented L-corner / blind-corner
equivalent.

## Still Not Covered

- **Parameterized generator semantics.** The Issue #3423 controls are manifest-level documented
  equivalents. The lower-level generator still consumes the existing vocabulary, so changing
  `width_m`, delayed actor starts, or continuous yielding-pressure dynamics remains future work.
- **Planner smoke scope.** The v0 family set now has an executable diagnostic smoke against the
  baseline-safe `simple_policy` planner. This proves the benchmark runner can emit one episode
  record per family; it is still not planner-ranking or benchmark-strength mechanism evidence.
  Broader planner comparisons remain out of scope for this note.
- **Geometry fidelity.** The blind-corner row is an explicit generated L-corner equivalent, not a
  calibrated reproduction of a real built environment or the Francis 2023 map.

## Claim boundary

These families are scenario *inputs* only. A generated layout proves nothing about planner
behavior; nothing here is planner-ranking, transfer, or benchmark-strength mechanism evidence.

## Validation runs (2026-06-22)

- `uv run pytest tests/benchmark/test_social_mini_game_families_issue_3279.py -q` → 8 passed
- `uv run python -c "import robot_sf.benchmark.scenario_generator"` → import OK
- `python scripts/demo/run_robot_sf_smoke.py --matrix configs/scenarios/sets/issue_3279_social_mini_game_families_v0.yaml --planners simple_policy --horizon 30 --workers 1 --output-root output/benchmarks/issue_3279_social_mini_game_smoke` → passed, 5 episode records

## Validation Runs (2026-06-22, Issue #3423 Update)

- `uv run pytest tests/benchmark/test_social_mini_game_families_issue_3279.py -q`
  validates the control metadata, blind-corner L-corner equivalent, deterministic generation, and
  one-episode-per-family `simple_policy` smoke.
