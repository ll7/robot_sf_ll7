# Issue #2586 Static-Deadlock Trace Fields

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2586>

Related:
- Issue #2452
- Issue #2544
- Issue #2588

## Scope

Issue #2586 added the reportability fields required by the #2452 `static_deadlock_recovery` suite
to episode rows tagged with `metadata.mechanism_aware_suite_id: static_deadlock_recovery`.

The emitted top-level row fields are:

- `low_progress_window`;
- `recenter_activation_count`;
- `distance_to_goal_delta`;
- `local_minimum_indicator`;
- `row_status`.

The existing nested planner-kinematics `execution_mode` field remains the execution-mode source.

## Result

The #2544 smoke matrix was rerun for `social_force` and `orca` after the field binding:

- `social_force`: 3 rows written, 0 successes, 3 `max_steps` terminations;
- `orca`: 3 rows written, 2 successes, 1 `max_steps` termination;
- all six rows now have the required static-deadlock trace/reportability fields;
- all six rows have `row_status: completed`;
- `social_force` on `narrow_passage` is the only row with `local_minimum_indicator.is_local_minimum`
  set to `true`;
- `recenter_activation_count` remains `0` for every row because this smoke only reran baseline
  planners.

The result is classified as `diagnostic_revise`, not `benchmark_candidate`: this PR fixes the
reportability gap, but the evidence is still one seed and baseline planners only. It does not run a
matched static-recenter intervention or the declared #2452 seed set.

## Evidence

Tracked compact evidence:

- [evidence/issue_2586_static_deadlock_trace_fields/summary.json](evidence/issue_2586_static_deadlock_trace_fields/summary.json)
- [evidence/issue_2586_static_deadlock_trace_fields/field_presence.json](evidence/issue_2586_static_deadlock_trace_fields/field_presence.json)
- [evidence/issue_2586_static_deadlock_trace_fields/suite_smoke_table.csv](evidence/issue_2586_static_deadlock_trace_fields/suite_smoke_table.csv)
- [evidence/issue_2586_static_deadlock_trace_fields/why_first_input.json](evidence/issue_2586_static_deadlock_trace_fields/why_first_input.json)
- [evidence/issue_2586_static_deadlock_trace_fields/why_first_report.md](evidence/issue_2586_static_deadlock_trace_fields/why_first_report.md)

The raw JSONL files used to build this compact evidence stayed in worktree-local ignored benchmark
output and are disposable. The tracked files above are the durable review surface.

## Validation

```bash
rtk uv run pytest tests/benchmark/test_issue_2452_mechanism_aware_suites.py tests/benchmark/test_map_runner_utils.py::test_run_map_episode_does_not_stop_on_waypoint_only_success -q
rtk uv run ruff check robot_sf/benchmark/map_runner.py tests/benchmark/test_map_runner_utils.py
rtk uv run robot_sf_bench --quiet validate-config --matrix configs/scenarios/sets/issue_2544_static_deadlock_smoke.yaml
rtk env LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run robot_sf_bench --quiet run --matrix configs/scenarios/sets/issue_2544_static_deadlock_smoke.yaml --out <worktree-local-output>/social_force/episodes.jsonl --schema robot_sf/benchmark/schemas/episode.schema.v1.json --algo social_force --workers 1 --horizon 120 --dt 0.1 --no-video --video-renderer none --record-simulation-step-trace --no-resume --benchmark-profile baseline-safe --socnav-missing-prereq-policy fail-fast --external-log-noise suppress --structured-output json
rtk env LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run robot_sf_bench --quiet run --matrix configs/scenarios/sets/issue_2544_static_deadlock_smoke.yaml --out <worktree-local-output>/orca/episodes.jsonl --schema robot_sf/benchmark/schemas/episode.schema.v1.json --algo orca --workers 1 --horizon 120 --dt 0.1 --no-video --video-renderer none --record-simulation-step-trace --no-resume --benchmark-profile baseline-safe --socnav-missing-prereq-policy fail-fast --external-log-noise suppress --structured-output json
```

## Claim Boundary

The change proves static-deadlock trace-field reportability for the bounded #2544 smoke matrix. It
does not prove planner ranking, mechanism benefit, transfer, or benchmark-candidate status.
Successor issue #2588 ran the next controlled baseline/intervention trace slice and classified it
as `controlled_trace_negative_mixed`; see
[issue_2588_static_deadlock_controlled_trace.md](issue_2588_static_deadlock_controlled_trace.md).
