# Issue #2306 Static-Recenter Activation Trace

Issue: [#2306](https://github.com/ll7/robot_sf_ll7/issues/2306)
Parent issue: [#2261](https://github.com/ll7/robot_sf_ll7/issues/2261)
Predecessor: [issue_2266_static_recenter_activation.md](issue_2266_static_recenter_activation.md)
Related transfer smoke: [issue_2221_static_recenter_transfer.md](issue_2221_static_recenter_transfer.md)
Status: diagnostic-only instrumented rerun; static recentering was inactive on the unsolved
held-out row.

## Goal

Rerun the #2221 static-recenter held-out smoke with activation-level instrumentation so the
repository can distinguish missing evidence from mechanism behavior.

The rerun uses the same matched rows:

- Baseline: `hybrid_rule_v3_fast_progress`
- Mechanism candidate: `issue_2170_static_recenter_only`
- Scenario matrix: `configs/scenarios/sets/issue_2128_heldout_family_transfer_pilot_eval.yaml`
- Seed: `111`
- Horizon: `500`

This result is not benchmark-strength, transfer, or planner-improvement evidence.

## Instrumentation

The rerun uses:

```bash
scripts/validation/run_static_recenter_activation_trace.py
```

The script runs each matched candidate with opt-in `planner_decision_trace` metadata from
`robot_sf/benchmark/map_runner.py`. Normal benchmark records do not include this trace unless the
private episode runner is explicitly called with `record_planner_decision_trace=True`.

## Required Fields

| Scenario | Activation Count | First Activation | Selected Command Source | Command Source Changed | Progress Delta After Activation | Trajectory Delta | Terminal Outcome Changed | Classification |
| --- | ---: | --- | --- | --- | --- | ---: | --- | --- |
| `classic_station_platform_medium` | `0` | `null` | `[]` | `false` | `null` | `0.0 m` | `false` | `mechanism_inactive` |
| `francis2023_intersection_wait` | `0` | `null` | `[]` | `false` | `null` | `0.0 m` | `false` | `comparator_already_solved_case` |

## Interpretation

The prior #2266 result was `activation_evidence_missing_terminal_outcomes_identical`. This rerun
fills that gap: static recentering did not activate on either held-out row.

For `classic_station_platform_medium`, the baseline and mechanism candidate both timed out at
500 steps with 60 near misses and no collisions. Because activation count is zero and the terminal
outcome is unchanged, the appropriate classification is `mechanism_inactive`, not
`mechanism_active_but_irrelevant`.

For `francis2023_intersection_wait`, both rows succeeded in 120 steps with no near misses or
collisions. This is a `comparator_already_solved_case`; it cannot demonstrate static-recenter
benefit even if a future trace caused activation.

## Recommendation

Do not tune or promote static recentering from this held-out slice. Treat the #2221 non-transfer
result as an inactive-mechanism negative for the unsolved row, and prefer route-progress mechanisms
with clearer activation hypotheses unless a new slice predeclares states where static recenter
should activate.

## Evidence

- Compact summary:
  [evidence/issue_2306_static_recenter_activation_trace_2026-06-05/summary.json](evidence/issue_2306_static_recenter_activation_trace_2026-06-05/summary.json)
- Evidence README:
  [evidence/issue_2306_static_recenter_activation_trace_2026-06-05/README.md](evidence/issue_2306_static_recenter_activation_trace_2026-06-05/README.md)

## Validation

Validation commands:

```bash
scripts/dev/run_worktree_shared_venv.sh -- ruff check robot_sf/benchmark/map_runner.py tests/benchmark/test_map_runner_utils.py scripts/validation/run_static_recenter_activation_trace.py
scripts/dev/run_worktree_shared_venv.sh -- ruff format --check robot_sf/benchmark/map_runner.py tests/benchmark/test_map_runner_utils.py scripts/validation/run_static_recenter_activation_trace.py
scripts/dev/run_worktree_shared_venv.sh -- pytest tests/benchmark/test_map_runner_utils.py::test_run_map_episode_merges_planner_runtime_stats -q
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 scripts/dev/run_worktree_shared_venv.sh -- python scripts/validation/run_static_recenter_activation_trace.py --output-json <worktree-local diagnostics summary>
python -m json.tool docs/context/evidence/issue_2306_static_recenter_activation_trace_2026-06-05/summary.json
```
