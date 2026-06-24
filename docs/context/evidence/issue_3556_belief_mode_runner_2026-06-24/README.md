# Issue #3556 ScenarioBelief Drop-vs-Retain Benchmark Runner Evidence (2026-06-24)

**Status:** integration + mechanism landed and tested. The evidence-producing **campaign run** is the
next step (run-gated only by compute, not by missing capability).

## Claim boundary (read first)

- **What this delivers:** the *capability* for the real benchmark runner
  (`robot_sf.benchmark.map_runner`) to run the `oracle` / `uncertain_retained` / `uncertain_dropped`
  belief-mode contrast with the production `stream_gap` planner + ScenarioBelief uncertainty gate.
  Before this, `map_runner` fed `stream_gap` no uncertainty sidecar, so the gate could not be
  exercised in the real runner at all.
- **What this does NOT yet deliver:** the campaign result. No paper-grade / nominal-benchmark claim is
  made here — that requires running the launch packet on a predeclared scenario family + seed matrix.
- **Uncertainty source caveat:** out-of-field-of-view agents are marked uncertain by a configurable
  FOV/range rule on the benchmark observation — **not** a calibrated perception/occlusion model.

## What landed

- `robot_sf/benchmark/scenario_belief_policy_hook.py` — builds a ScenarioBelief from each benchmark
  observation, degrades the existence confidence of out-of-FOV/range agents (keeping rows 1:1 with the
  observation so the gate can act), projects through the production
  `project_scenario_belief_for_planner`, and merges the uncertainty sidecar into the observation.
  `BeliefModeStreamGapAdapter` wraps the real `StreamGapPlannerAdapter` and delegates everything else.
  Fail-closed: any projection failure omits the sidecar (planner keeps every agent — conservative).
- `robot_sf/benchmark/map_runner.py` — the `stream_gap` branch reads a `belief_mode` algo-config knob;
  when set it wraps the adapter and sets the planner uncertainty gate (ON only for `uncertain_dropped`).
- `configs/benchmarks/scenario_belief_drop_vs_retain_issue_3556.yaml` — launch packet for the campaign.

## Mechanism proof (unit-level, this PR)

On a benchmark-style observation with one agent ahead (in FOV) and one behind (out of 120° FOV):

| Mode | uncertainty rows | gate | agents dropped | stream_gap command |
| --- | --- | --- | --- | --- |
| `oracle` | `[0.98, 0.98]` | off | 0 | wait `(0.0, 0.0)` |
| `uncertain_retained` | `[0.2, 0.98]` | off | 0 | wait `(0.0, 0.0)` — **same as oracle** |
| `uncertain_dropped` | `[0.2, 0.98]` | on | **1** | commit `(0.95, 0.0)` |

So retaining the uncertain agent matches oracle, and only *dropping* it changes the planner's
decision — the #3471 mechanism, now through the real adapter + gate. Verified by
`tests/benchmark/test_scenario_belief_policy_hook_issue_3556.py` (6 tests; the `map_runner` wiring
test runs under `torch`).

## Next step — the campaign run (evidence-producing)

```bash
# Run each belief_mode variant through the canonical map-runner campaign on the predeclared
# scenario family + seed matrix, then compute the episode-level safety contrast (#3471 metrics).
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/scenario_belief_drop_vs_retain_issue_3556.yaml ...
```

Then classify on the evidence ladder (`revise_default_gating_unsafe` / `retention_dominates` /
`not_distinguishable_at_matrix` / `inconclusive_oracle_baseline_unsafe`), justifying the seed count via
`scripts/tools/analyze_seed_sufficiency.py`. Aim for a near-safe oracle baseline (the #3471 caveat
where the controlled oracle collided ~42%).

## Known dependencies / caveats

- The `to_socnav_struct` non-negative position clipping and the `stream_gap` absent-next-goal bug
  (both found in #3471, the latter tracked as #3555) can affect real-runner scenarios — account for
  them when authoring the scenario family.
- A calibrated perception/occlusion uncertainty source (vs the FOV rule here) is the natural successor.
