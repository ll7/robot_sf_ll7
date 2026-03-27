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

Safety barrier (`issue717_iter3`):
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

Reference `risk_dwa` run on the same slice:
- episodes: `18`
- success rate: `0.1667`
- collision rate: `0.6667`
- terminations: `3 success`, `3 terminated`, `12 collision`

Interpretation:
- The spike does run correctly through the repository’s benchmark path.
- The heuristic does not even clear the easiest open-space sanity cases yet.
- That means the present controller logic is not just weak on obstacle avoidance; it is also weak on
  basic heading/progress behavior.
- Because of that, broadening to larger suites or paper-facing configs would be misleading.

## Recommendation

Do not broaden `safety_barrier` beyond testing-only status in its current form.

The planner family remains conceptually interesting because:
- the provenance boundary is clean,
- the integration burden is low,
- and the repository can run it end to end.

But any continuation should start from a redesign hypothesis, not incremental benchmark expansion.

Recommended next focus:
- diagnose the open-space timeout / curved-progress behavior first,
- then revisit obstacle filtering once the nominal unicycle controller is reliable.

## Validation

- `uv run pytest tests/planner/test_safety_barrier.py tests/benchmark/test_algorithm_metadata_contract.py tests/benchmark/test_map_runner_preflight_profiles.py tests/benchmark/test_map_runner_utils.py -q`
- `uv run python scripts/validation/run_safety_barrier_static_slice.py --output-dir output/validation/safety_barrier_static_slice/issue717_iter3`
- `uv run python scripts/validation/run_safety_barrier_static_slice.py --algo risk_dwa --algo-config configs/algos/risk_dwa_camera_ready.yaml --output-dir output/validation/safety_barrier_static_slice/risk_dwa_reference`

Artifacts:
- `output/validation/safety_barrier_static_slice/issue717_iter3/summary.json`
- `output/validation/safety_barrier_static_slice/issue717_iter3/summary.md`
- `output/validation/safety_barrier_static_slice/risk_dwa_reference/summary.json`
- `output/validation/safety_barrier_static_slice/risk_dwa_reference/summary.md`

## Risks / Follow-ups

- The current planner is intentionally simple, so a negative result here does not invalidate the
  broader barrier-style controller idea.
- Follow-up should be a redesign issue, not a benchmark-surface expansion issue.
- Follow-up tracker: `#718` `Redesign safety_barrier nominal controller before broader evaluation`.
