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
mechanism label and the parameter values in `metadata`.

## Covered mechanisms (this cut)

| Family | mechanism_aware_suite_id | Generator mapping |
| --- | --- | --- |
| doorway | `doorway_bottleneck_negotiation` | `obstacle=bottleneck`, `flow=bi` |
| hallway | `hallway_bidirectional_passing` | `obstacle=open`, `flow=bi` |
| intersection | `intersection_crossing_negotiation` | `obstacle=open`, `flow=cross` |
| blind_corner | `blind_corner_occlusion_exposure` | `obstacle=maze`, `flow=uni` |
| crowded_traffic | `crowded_traffic_merge_negotiation` | `density=high`, `flow=merge`, `groups=0.2` |

## Not yet covered (follow-up under #3279)

- **First-class parameter exposure** for `width_m`, `occlusion_geometry`, `start_timing_s`,
  and `yielding_pressure`. The current generator vocabulary
  (`density`/`flow`/`obstacle`/`groups`/`speed_var`/`goal_topology`/`robot_context`) only
  *approximates* these triggers; extending the generator schema is deferred.
- **Full planner smoke run.** This cut proves the families generate deterministic, schema-valid
  scenario inputs. It does **not** run a baseline planner across the families. The canonical
  smoke command shape is documented in the config header; producing episode outputs against a
  baseline is the next step and remains diagnostic-smoke, not benchmark-strength evidence.
- **L-corner geometry fidelity.** `blind_corner` uses the coarse `maze` obstacle layout as an
  occlusion proxy; a dedicated L-corner geometry is a follow-up.

## Claim boundary

These families are scenario *inputs* only. A generated layout proves nothing about planner
behavior; nothing here is planner-ranking, transfer, or benchmark-strength mechanism evidence.

## Validation run (2026-06-22)

- `uv run pytest tests/benchmark/test_social_mini_game_families_issue_3279.py -q` → 5 passed
- `uv run python -c "import robot_sf.benchmark.scenario_generator"` → import OK
