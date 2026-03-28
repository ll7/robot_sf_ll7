# Issue 717: Safety Barrier Native Spike

## Goal

Evaluate whether a clean-room, testing-only `safety_barrier` planner is promising enough to keep
as a native planner path after `#695` ruled out provenance-safe upstream wrapping of
`safe_control`.

## Scope

- In scope:
  - one native `safety_barrier` planner entrypoint
  - explicit experimental/testing-only readiness gating
  - one six-scenario static / verified-simple slice
- Out of scope:
  - upstream wrapping or vendoring
  - paper-facing benchmark claims
  - dynamic pedestrian reasoning
  - solver-backed CBF-QP equivalence claims

## Evidence

- Planner implementation:
  - `robot_sf/planner/safety_barrier.py`
- Readiness and metadata wiring:
  - `robot_sf/benchmark/algorithm_readiness.py`
  - `robot_sf/benchmark/algorithm_metadata.py`
  - `robot_sf/benchmark/map_runner.py`
- Validation assets:
  - `configs/algos/safety_barrier_camera_ready.yaml`
  - `configs/scenarios/sets/safety_barrier_static_slice_v1.yaml`
  - `scripts/validation/run_safety_barrier_static_slice.py`
- Reference comparison:
  - `configs/algos/risk_dwa_camera_ready.yaml`

## Result

Assessment: `limited but interesting, needs hypothesis-driven revision`

The clean-room spike is benchmark-contract compliant and runnable, but the current heuristic is
too weak to continue broadening as-is.

### Observed static-slice outcomes

Initial spike (`issue717_iter3`):
- episodes: `18`
- success rate: `0.0000`
- collision rate: `0.6667`
- terminations: `6 terminated`, `12 collision`

Per scenario:
- `empty_map_8_directions_east`: `0/3` success, all `terminated`
- `goal_behind_robot`: `0/3` success, all `terminated`
- `single_obstacle_circle`: `0/3` success, all `collision`
- `single_obstacle_rectangle`: `0/3` success, all `collision`
- `line_wall_detour`: `0/3` success, all `collision`
- `narrow_passage`: `0/3` success, all `collision`

Issue `#718` redesign iteration (`issue718_iter1` / `issue718_iter2`):
- episodes: `18`
- success rate: `0.5000`
- collision rate: `0.5000`
- terminations: `9 success`, `9 collision`

Per scenario:
- `empty_map_8_directions_east`: `3/3` success
- `goal_behind_robot`: `3/3` success
- `single_obstacle_rectangle`: `3/3` success
- `single_obstacle_circle`: `0/3` success, all `collision`
- `line_wall_detour`: `0/3` success, all `collision`
- `narrow_passage`: `0/3` success, all `collision`

Rejected redesign branch (`issue718_iter4`):
- episodes: `18`
- success rate: `0.3333`
- collision rate: `0.5000`
- terminations: `6 success`, `9 collision`, `3 terminated`

Interpretation:
- Remembering the previous waypoint and slowing by cross-track error did not solve the remaining
  obstacle-geometry failures.
- It regressed `single_obstacle_rectangle` from `3/3` success to `0/3` terminated.
- That route-memory heuristic is not a good continuation path for this planner.

Rejected follow-up redesigns:
- sampled trajectory rollouts over a short `(v, omega)` lattice
- route-corridor tracking with obstacle-side commitment
- repulsive obstacle-field steering

Interpretation:
- none of those variants improved the best known `9/18` static-slice result,
- the sampler only matched `9/18` after fixing its activation bug,
- and the other variants either regressed unit behavior or reduced the slice to `6/18` or
  other worse outcomes.
- The negative result is now clean: further local-reactive tweaks on `safety_barrier` are not the
  highest-value continuation path.

### Different planner design contrast

Occupancy-grid route planner (`grid_route_iter4` retained state):
- episodes: `18`
- success rate: `0.7778`
- collision rate: `0.2222`
- terminations: `14 success`, `4 collision`

Per scenario:
- `empty_map_8_directions_east`: `3/3` success
- `goal_behind_robot`: `3/3` success
- `single_obstacle_rectangle`: `3/3` success
- `line_wall_detour`: `3/3` success
- `single_obstacle_circle`: `2/3` success, `1/3` collision
- `narrow_passage`: `0/3` success, all `collision`

Earlier route-planner snapshot (`grid_route_iter2`):
- reached the same `14/18` success ceiling,
- but failed `narrow_passage` by timeout instead of collision (`1 collision`, `3 terminated` total).

Rejected route-planner refinement (`grid_route_iter3`):
- line-of-sight waypoint skipping reduced the slice to `11/18`,
- and turned `line_wall_detour` into `0/3` collision.

Quick config sweep:
- `obstacle_inflation_cells = 0` produced the same `14/18` result as the default config.
- shortening the waypoint lookahead to `3` cells regressed the slice to `6/18`.

Interpretation:
- The route-planning design outperformed the best `safety_barrier` result on the same proof slice.
- That is strong evidence that the remaining hard cases are topology-sensitive, not just reactive
  steering failures.
- `narrow_passage` is still unresolved, and the exact failure mode there is not yet stable, so the
  route planner is not ready for broader benchmark claims either.
- Even with that caveat, it is a substantially stronger next direction than continuing to patch the
  barrier controller.

Reference `risk_dwa` run on the same slice:
- episodes: `18`
- success rate: `0.1667`
- collision rate: `0.6667`
- terminations: `3 success`, `3 terminated`, `12 collision`

Interpretation:
- The spike does run correctly through the repository’s benchmark path.
- The original `0/18` result was dominated by a route-targeting bug: the planner switched to the
  `[0, 0]` `goal_next` sentinel before finishing the active waypoint.
- Fixing that route-targeting bug cleared the open-space east, turn-around, and rectangular-obstacle
  cases without changing the benchmark plumbing.
- The remaining failures are all obstacle-geometry failures, not nominal-progress failures.
- Several reactive redesigns were tested and rejected because they did not beat the `9/18`
  baseline or actively regressed it.
- Because of that, broadening `safety_barrier` to larger suites or paper-facing configs is still
  misleading, even though the planner family is more plausible than the initial `0/18` snapshot
  suggested.
- A different planner design, `grid_route`, already shows a materially better static-slice result,
  which narrows the search: topology-aware planning is more promising here than continued
  barrier-style steering tweaks.

## Recommendation

Do not broaden `safety_barrier` beyond testing-only status in its current form.

The planner family remains conceptually interesting because:
- the provenance boundary is clean,
- the integration burden is low,
- and the repository can run it end to end.

But any continuation should start from a redesign hypothesis, not incremental benchmark expansion.

Recommended next focus:
- keep `safety_barrier` as a documented negative-result spike,
- shift experimentation to topology-aware planners such as `grid_route`,
- use `narrow_passage` as the next acceptance gate for any continuation,
- do not revisit local-reactive `safety_barrier` redesigns without a new, explicit structural
  hypothesis.

## Validation

- `uv run pytest tests/planner/test_safety_barrier.py tests/benchmark/test_algorithm_metadata_contract.py tests/benchmark/test_map_runner_preflight_profiles.py tests/benchmark/test_map_runner_utils.py -q`
- `uv run python scripts/validation/run_safety_barrier_static_slice.py --output-dir output/validation/safety_barrier_static_slice/issue717_iter3`
- `uv run pytest tests/planner/test_safety_barrier.py -q`
- `uv run python scripts/validation/run_safety_barrier_static_slice.py --output-dir output/validation/safety_barrier_static_slice/issue718_iter1`
- `uv run python scripts/validation/run_safety_barrier_static_slice.py --output-dir output/validation/safety_barrier_static_slice/issue718_iter2`
- `uv run python scripts/validation/run_safety_barrier_static_slice.py --output-dir output/validation/safety_barrier_static_slice/issue718_iter4`
- `uv run python scripts/validation/run_safety_barrier_static_slice.py --output-dir output/validation/safety_barrier_static_slice/issue718_iter5`
- `uv run python scripts/validation/run_safety_barrier_static_slice.py --output-dir output/validation/safety_barrier_static_slice/issue718_iter17`
- `uv run python scripts/validation/run_safety_barrier_static_slice.py --algo risk_dwa --algo-config configs/algos/risk_dwa_camera_ready.yaml --output-dir output/validation/safety_barrier_static_slice/risk_dwa_reference`
- `uv run python scripts/validation/run_safety_barrier_static_slice.py --algo grid_route --algo-config configs/algos/grid_route_camera_ready.yaml --output-dir output/validation/safety_barrier_static_slice/grid_route_iter2`
- `uv run python scripts/validation/run_safety_barrier_static_slice.py --algo grid_route --algo-config configs/algos/grid_route_camera_ready.yaml --output-dir output/validation/safety_barrier_static_slice/grid_route_iter3`
- `uv run python scripts/validation/run_safety_barrier_static_slice.py --algo grid_route --algo-config configs/algos/grid_route_camera_ready.yaml --output-dir output/validation/safety_barrier_static_slice/grid_route_iter4`

Artifacts:
- `output/validation/safety_barrier_static_slice/issue717_iter3/summary.json`
- `output/validation/safety_barrier_static_slice/issue717_iter3/summary.md`
- `output/validation/safety_barrier_static_slice/issue718_iter1/summary.json`
- `output/validation/safety_barrier_static_slice/issue718_iter1/summary.md`
- `output/validation/safety_barrier_static_slice/issue718_iter2/summary.json`
- `output/validation/safety_barrier_static_slice/issue718_iter2/summary.md`
- `output/validation/safety_barrier_static_slice/issue718_iter4/summary.json`
- `output/validation/safety_barrier_static_slice/issue718_iter4/summary.md`
- `output/validation/safety_barrier_static_slice/issue718_iter5/summary.json`
- `output/validation/safety_barrier_static_slice/issue718_iter5/summary.md`
- `output/validation/safety_barrier_static_slice/issue718_iter17/summary.json`
- `output/validation/safety_barrier_static_slice/issue718_iter17/summary.md`
- `output/validation/safety_barrier_static_slice/risk_dwa_reference/summary.json`
- `output/validation/safety_barrier_static_slice/risk_dwa_reference/summary.md`
- `output/validation/safety_barrier_static_slice/grid_route_iter2/summary.json`
- `output/validation/safety_barrier_static_slice/grid_route_iter2/summary.md`
- `output/validation/safety_barrier_static_slice/grid_route_iter3/summary.json`
- `output/validation/safety_barrier_static_slice/grid_route_iter3/summary.md`
- `output/validation/safety_barrier_static_slice/grid_route_iter4/summary.json`
- `output/validation/safety_barrier_static_slice/grid_route_iter4/summary.md`

## Risks / Follow-ups

- The current planner is intentionally simple, so a negative result here does not invalidate the
  broader barrier-style controller idea.
- Follow-up should be a redesign issue, not a benchmark-surface expansion issue.
- Follow-up tracker: `#718` `Redesign safety_barrier nominal controller before broader evaluation`.
- The stronger next candidate is the topology-aware `grid_route` design, but it still needs a
  narrow-passage solution before broader benchmark use is justified.
