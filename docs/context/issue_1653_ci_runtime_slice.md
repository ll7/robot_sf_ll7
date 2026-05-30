# Issue #1653 CI Runtime Slice

Status: PR-ready implementation note

## Scope

This note records a conservative first slice for #1653. It does not change test selection,
benchmark semantics, branch-protection job names, or required validation phases.

## Baseline Timing

Recent successful `CI` workflow runs were summarized with:

```bash
uv run python scripts/dev/ci_timing_summary.py --run-id <run-id> --top 8 --json
```

| run id | title | event | total | job span | queue |
| --- | --- | --- | ---: | ---: | ---: |
| 26648705023 | feat: add optional Three.js recording viewer | pull_request | 997s | 992s | 4s |
| 26638333993 | feat: add lidar tracked social-force adapter (#1669) | push | 926s | 901s | 24s |
| 26637972460 | docs: define learned policy adapter interface | pull_request | 904s | 900s | 3s |
| 26637835994 | docs: audit SiT Dataset terms | pull_request | 898s | 895s | 2s |
| 26637057677 | docs: define learned policy adapter interface | pull_request | 928s | 924s | 3s |
| 26637057815 | feat: add lidar tracked social-force adapter | pull_request | 926s | 921s | 4s |
| 26636789637 | fix: align lidar compatibility safety barrier gate | pull_request | 921s | 918s | 2s |
| 26636095246 | docs: survey local planner repositories | pull_request | 1351s | 1348s | 2s |

The observed successful-run median is about 927s total, with one longer documentation PR run at
1351s. Queue time is small in this sample, so most elapsed time is inside CI jobs.

## Tooling Gap

`gh run view --json jobs` returns job `startedAt` and `completedAt`, but the step objects observed
for run `26648705023` do not include step timestamps. The timing helper therefore produced empty
`slowest_steps` output even when jobs and step names were present.

This slice updates `scripts/dev/ci_timing_summary.py` to report slowest jobs as a fallback and to
make missing step timestamps explicit in Markdown output.

## Local Slowest Tests

The branch readiness run used:

```bash
BASE_REF=origin/main PYTEST_NUM_WORKERS=8 scripts/dev/pr_ready_check.sh
```

It completed in 292.71s with 4432 passed and 11 skipped. The slowest reported calls were:

| test | duration |
| --- | ---: |
| `tests/examples/test_examples_run.py::test_example_runs_without_error[advanced/01_backend_selection.py]` | 15.79s |
| `tests/planner/test_policy_stack_v1.py::test_policy_stack_runs_atomic_topology_smoke_through_map_runner` | 14.42s |
| `tests/examples/test_examples_run.py::test_example_runs_without_error[quickstart/01_basic_robot.py]` | 14.42s |
| `tests/test_osm_backward_compat.py::TestOSMBackwardCompat::test_osm_map_with_robot_environment` | 13.61s |
| `tests/examples/test_examples_run.py::test_example_runs_without_error[advanced/08_multi_pedestrian.py]` | 13.08s |
| `tests/examples/test_examples_run.py::test_example_runs_without_error[benchmarks/demo_social_nav_scenarios.py]` | 12.89s |
| `tests/adversarial/test_adversarial_search.py::test_multi_ped_adversarial_runtime_config_resets_and_steps` | 12.76s |
| `tests/examples/test_examples_run.py::test_example_runs_without_error[advanced/21_occupancy_grid_workflow.py]` | 12.72s |
| `tests/examples/test_examples_run.py::test_example_runs_without_error[advanced/07_single_pedestrian.py]` | 12.71s |
| `tests/examples/test_examples_run.py::test_example_runs_without_error[quickstart/03_custom_map.py]` | 12.69s |

The local slowest-test evidence points at example smoke coverage and one policy-stack smoke as the
first places to inspect before changing test selection or sharding.

## Low-Risk Runtime Reduction

The `fast-feedback` job still computes coverage JSON and compares against the cached baseline.
Routine pull-request runs no longer upload the full coverage HTML/database artifact on success.
Coverage artifacts remain uploaded when `fast-feedback` fails and on `main`, where baseline
maintenance happens.

This preserves the proof bar while avoiding a success-path artifact upload for ordinary PRs.

## Follow-Up

Future #1653 work should use the improved job-level timing output to decide whether setup, tests,
smoke scripts, or artifact uploads dominate current runs. Avoid moving required checks until a
before/after CI run confirms the impact.
