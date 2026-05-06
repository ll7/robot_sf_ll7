# Issue #1002 Complexity and Test Runtime Baseline

## Scope

Issue #1002 adds `scripts/dev/complexity_runtime_baseline.py`, a lightweight stdlib-only command
for repeatable refactor triage. The command reports:

- largest Python modules by simple non-comment source lines,
- longest functions or methods by AST line span,
- optional pytest `--durations` rows when a captured pytest log is supplied with `--pytest-log`.

The command is intentionally not a dashboard. It prints text by default, supports deterministic
JSON with `--json`, and does not create durable generated outputs.

## Baseline Command

```bash
rtk uv run python scripts/dev/complexity_runtime_baseline.py --top 10 robot_sf scripts tests
```

To include slow-test rows, pass a captured pytest output log:

```bash
rtk uv run python scripts/dev/complexity_runtime_baseline.py \
  --top 10 \
  --pytest-log output/tmp/pr_ready_pytest.log \
  robot_sf scripts tests
```

The log path is an input only. Do not promote pytest logs as durable artifacts unless a later issue
explicitly needs them.

## First Snapshot

Top module-size hotspots from the initial run:

| path | code_lines | total_lines |
| --- | --- | --- |
| `robot_sf/planner/socnav.py` | 4379 | 4896 |
| `robot_sf/benchmark/camera_ready_campaign.py` | 3357 | 3645 |
| `robot_sf/benchmark/map_runner.py` | 2858 | 3131 |
| `scripts/training/train_ppo.py` | 2740 | 3072 |
| `tests/benchmark/test_map_runner_utils.py` | 2503 | 2959 |

Top function-size hotspots:

| path | function | lines |
| --- | --- | --- |
| `robot_sf/benchmark/camera_ready_campaign.py` | `run_campaign` | 1003 |
| `robot_sf/benchmark/map_runner.py` | `_build_policy` | 979 |
| `scripts/training/run_predictive_training_pipeline.py` | `main` | 501 |
| `robot_sf/benchmark/scenario_difficulty.py` | `build_scenario_difficulty_analysis` | 453 |
| `scripts/training/train_predictive_planner.py` | `main` | 433 |

Runtime indicators were sampled from the committed PR readiness pytest duration output:

| duration_seconds | phase | nodeid |
| --- | --- | --- |
| 27.40 | call | `tests/examples/test_examples_run.py::test_example_runs_without_error[advanced/03_image_observations.py]` |
| 24.01 | call | `tests/examples/test_examples_run.py::test_example_runs_without_error[occupancy_reward_shaping.py]` |
| 23.77 | call | `tests/examples/test_examples_run.py::test_example_runs_without_error[quickstart/01_basic_robot.py]` |
| 23.05 | call | `tests/test_load_states_and_record_video.py::test_load_states_and_record_video` |
| 22.70 | call | `tests/test_image_system_integration.py::test_image_system_integration` |

## Interpretation

- Benchmark orchestration is the dominant refactor target: `camera_ready_campaign.py` and
  `map_runner.py` appear in both module-size and function-size lists.
- `robot_sf/planner/socnav.py` is the largest file, but this baseline alone does not prove it is
  the first risk-reduction target; pair it with planner-specific behavior tests before splitting.
- The slowest test pressure is concentrated in example/image integration paths, matching the need
  for CI runtime diagnosis rather than broad test removal.
- Training entry points remain large but are less urgent for benchmark correctness than the
  benchmark runner/campaign hotspots.

## Ticket Recommendations

The 2026-05-05 scan already opened enough follow-up containers for this baseline:

- #1001 should consume the benchmark/planner/training complexity evidence before broad refactors.
- #1006 should consume the slow-test duration evidence before changing CI sharding or summaries.
- #1005 remains a bounded maintenance ticket, but this baseline does not rank pedestrian NPC code
  as a top-size hotspot; keep it behind benchmark orchestration and CI-runtime work.

No additional follow-up issue is needed from this slice.

## Validation

RED proof:

```bash
rtk uv run pytest tests/dev/test_complexity_runtime_baseline.py -q
```

Failed before implementation with `ModuleNotFoundError` because
`scripts.dev.complexity_runtime_baseline` did not exist.

GREEN proof:

```bash
rtk uv run pytest tests/dev/test_complexity_runtime_baseline.py -q
rtk uv run ruff check scripts/dev/complexity_runtime_baseline.py tests/dev/test_complexity_runtime_baseline.py
rtk uv run python scripts/dev/complexity_runtime_baseline.py --top 10 robot_sf scripts tests
```

The targeted test passed with `3 passed in 9.75s`; Ruff passed on the touched Python files.

## Artifact Decision

The baseline command writes no files. Targeted pytest refreshed ignored coverage output under
`output/coverage/`; those files are disposable validation artifacts and are not promoted.
