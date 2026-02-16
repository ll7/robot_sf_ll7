# Issues #500, #501, #504 Execution Notes (2026-02-16)

## Scope
- #500: Document stochastic reference baseline semantics and category metadata.
- #501: Clarify and extend time-to-goal normalization contract.
- #504: Expose planner-kinematics compatibility and adapter-impact metadata.

## Implemented
- Added shared metadata helper: `robot_sf/benchmark/algorithm_metadata.py`.
  - Canonical algorithm mapping.
  - `baseline_category` contract: `diagnostic | classical | learning`.
  - Policy semantics labels (including stochastic reference semantics for `random`).
  - Planner-kinematics contract (`execution_mode`, adapter markers, command-space metadata).
- Wired metadata enrichment into:
  - `robot_sf/benchmark/runner.py`
  - `robot_sf/benchmark/map_runner.py`
  - `robot_sf/benchmark/full_classic/orchestrator.py`
  - `scripts/tools/policy_analysis_run.py`
- Added adapter-impact probing mode for map runs:
  - New flag plumbing in benchmark run path.
  - Runtime counters for PPO native vs adapted command conversion.
  - Summary-level metadata contract emitted from map-runner preflight.
- Extended metric contract in `robot_sf/benchmark/metrics.py`:
  - Preserved `time_to_goal_norm` behavior (failure clamp = `1.0`).
  - Added `time_to_goal_norm_success_only`.
  - Added `time_to_goal_ideal_ratio` (`time_to_goal / (shortest_path_len / robot_max_speed)`).
  - Added validity flags:
    - `time_to_goal_success_only_valid`
    - `time_to_goal_ideal_ratio_valid`
- Updated episode schema metadata properties:
  - `baseline_category`, `policy_semantics`, `planner_kinematics`, `adapter_impact`.
- Updated public benchmark spec docs:
  - Baseline categories now use `diagnostic/classical/learning`.
  - Explicitly documents stochastic `random` baseline vs deterministic `goal`.
  - Added time-to-goal contract caveats and success-only guidance.

## Validation Plan
- Run targeted unit/integration tests for:
  - metrics contract behavior
  - metadata enrichment in runner/map-runner/orchestrator
  - map-runner preflight summary payloads
- Run focused benchmark CLI tests for new `--adapter-impact-eval` option.
