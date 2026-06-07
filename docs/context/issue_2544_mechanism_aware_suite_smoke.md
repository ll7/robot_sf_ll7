# Issue #2544 Mechanism-Aware Suite Smoke

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2544>

## Scope

Issue #2544 ran the first bounded smoke of the proposal-only #2452 mechanism-aware
local-navigation suite registry. The smoke binds the `static_deadlock_recovery` suite to
`configs/scenarios/sets/issue_2544_static_deadlock_smoke.yaml`, using:

- scenarios: `classic_bottleneck_low`, `classic_head_on_corridor_low`, `narrow_passage`;
- seed budget: `111`;
- planners: `social_force`, `orca`;
- horizon: `120`;
- videos: disabled;
- simulation step trace: enabled.

This is diagnostic smoke evidence for suite operability and reportability only. It is not planner
ranking, transfer evidence, publication evidence, or benchmark-strength mechanism proof.

## Result

The suite executed end-to-end for both planners:

- `social_force`: 3 rows written, 0 successes, 3 `max_steps` terminations;
- `orca`: 3 rows written, 2 successes, 1 `max_steps` termination.

The result is classified as `diagnostic_revise`, not `benchmark_candidate`, because required
static-deadlock mechanism trace fields from the #2452 suite contract were absent from the episode
rows:

- `low_progress_window`;
- `recenter_activation_count`;
- `distance_to_goal_delta`;
- `local_minimum_indicator`;
- `row_status`.

The only required field found anywhere in the rows was `execution_mode`. The runner does expose
top-level `status`, but not the suite contract's explicit `row_status` trace field.

## Evidence

Tracked compact evidence:

- [evidence/issue_2544_mechanism_aware_suite_smoke/summary.json](evidence/issue_2544_mechanism_aware_suite_smoke/summary.json)
- [evidence/issue_2544_mechanism_aware_suite_smoke/suite_smoke_table.csv](evidence/issue_2544_mechanism_aware_suite_smoke/suite_smoke_table.csv)
- [evidence/issue_2544_mechanism_aware_suite_smoke/why_first_input.json](evidence/issue_2544_mechanism_aware_suite_smoke/why_first_input.json)
- [evidence/issue_2544_mechanism_aware_suite_smoke/why_first_report.md](evidence/issue_2544_mechanism_aware_suite_smoke/why_first_report.md)

The raw JSONL files used to build this compact evidence stayed in worktree-local ignored benchmark
output and are disposable. The tracked files above are the durable review surface.

## Validation

```bash
rtk uv run pytest tests/benchmark/test_issue_2452_mechanism_aware_suites.py -q
rtk uv run robot_sf_bench --quiet validate-config --matrix configs/scenarios/sets/issue_2544_static_deadlock_smoke.yaml
rtk uv run robot_sf_bench --quiet run --matrix configs/scenarios/sets/issue_2544_static_deadlock_smoke.yaml --out <worktree-local-output>/social_force/episodes.jsonl --schema robot_sf/benchmark/schemas/episode.schema.v1.json --algo social_force --workers 1 --horizon 120 --dt 0.1 --no-video --video-renderer none --record-simulation-step-trace --no-resume --benchmark-profile baseline-safe --socnav-missing-prereq-policy fail-fast --external-log-noise suppress --structured-output json
rtk env LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run robot_sf_bench --quiet run --matrix configs/scenarios/sets/issue_2544_static_deadlock_smoke.yaml --out <worktree-local-output>/orca/episodes.jsonl --schema robot_sf/benchmark/schemas/episode.schema.v1.json --algo orca --workers 1 --horizon 120 --dt 0.1 --no-video --video-renderer none --record-simulation-step-trace --no-resume --benchmark-profile baseline-safe --socnav-missing-prereq-policy fail-fast --external-log-noise suppress --structured-output json
rtk uv run python scripts/tools/generate_why_first_report.py --input docs/context/evidence/issue_2544_mechanism_aware_suite_smoke/why_first_input.json --output docs/context/evidence/issue_2544_mechanism_aware_suite_smoke/why_first_report.md
```

## Claim Boundary

The smoke shows the proposed suite can be bound to executable rows for two native/core planners and
reported in a why-first format. It also shows the current runner output is missing the trace fields
needed to interpret metric deltas as static-deadlock mechanism evidence. The next proof step is to
add or bind runner instrumentation for the missing static-deadlock trace fields before promoting
the suite beyond diagnostic smoke; follow-up issue #2586 tracks that instrumentation/reportability
work.
