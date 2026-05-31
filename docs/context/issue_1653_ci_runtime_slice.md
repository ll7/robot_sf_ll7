# Issue #1653 CI Runtime Slice

Status: xdist scheduler slice

## Scope

This note records the conservative #1653 timing slice after the initial artifact-upload trim landed
on `main` in PR [#1681](https://github.com/ll7/robot_sf_ll7/pull/1681). Issue #1766 adds a new
`examples-smoke` CI job to make example smoke tests separately visible/timed while keeping them
required by the aggregate `ci` job.

## Baseline Timing

Recent successful `CI` workflow runs were summarized with:

```bash
uv run python scripts/dev/ci_timing_summary.py --run-id <run-id> --top 8 --json
```

The post-#1681 sample below covers ten successful runs from 2026-05-30 after
`ci: report job timings and trim PR artifacts (#1681)` reached `main`.

| run id | title | event | total | job span | queue | fast-feedback | smoke-artifacts | examples-smoke |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 26676909707 | docs: define learned policy artifact manifests (#1686) | pull_request | 832s | 829s | 2s | 813s | 263s | n/a |
| 26676725258 | docs: add queue exhaustion audit example (#1688) | pull_request | 931s | 928s | 2s | 920s | 260s | n/a |
| 26676578865 | docs: add root layout inventory (#1690) | pull_request | 935s | 931s | 3s | 927s | 228s | n/a |
| 26676471223 | feat: add learned risk surface interface (#1675) | pull_request | 940s | 904s | 35s | 899s | 264s | n/a |
| 26676440525 | docs: add proxemic comfort profile slice (#1676) | pull_request | 894s | 891s | 2s | 886s | 253s | n/a |
| 26676440608 | docs: record open issue execution audit | pull_request | 925s | 901s | 24s | 897s | 249s | n/a |
| 26676322034 | docs: add topology hypothesis diagnostic audit (#1674) | pull_request | 793s | 790s | 2s | 785s | 251s | n/a |
| 26676034847 | feat: add learned risk surface interface (#1675) | pull_request | 882s | 881s | 1s | 875s | 244s | n/a |
| 26675734525 | docs: add proxemic comfort profile slice (#1676) | pull_request | 916s | 913s | 2s | 906s | 253s | n/a |
| 26675563950 | ci: report job timings and trim PR artifacts (#1681) | push | 974s | 897s | 76s | 891s | 279s | n/a |

Observed post-trim sample medians:

- Total wall time: 920s, with a 793s minimum and 974s maximum.
- Job span: 899s median.
- Queue time: usually low, except two runs at 24s and 76s.
- `fast-feedback`: 894s median and 879.9s mean.
- `smoke-artifacts`: 253s median and 254.4s mean.

The current dominant critical-path cost is still `fast-feedback`. `smoke-artifacts` finishes in
parallel and is not the wall-time limiter unless `fast-feedback` is shortened substantially.
The sample is not enough to claim a statistically meaningful speedup from PR #1681; it is a
post-change baseline for the next slice.

## Tooling Gap

`gh run view --json jobs` returns job `startedAt` and `completedAt`, but the step objects observed
for sampled runs do not include step timestamps. The timing helper therefore produced empty
`slowest_steps` output even when jobs and step names were present.

PR #1681 updated `scripts/dev/ci_timing_summary.py` to report slowest jobs as a fallback and to
make missing step timestamps explicit in Markdown output. The post-trim sample still has no
step-level durations, so PR #1700 added explicit phase timing emitted by
`scripts/dev/ci_driver.sh` instead of relying on GitHub's step payload.

`fast-feedback` and `smoke-artifacts` both repeat checkout, uv setup, Python setup, system package
installation, dependency sync, and artifact migration. Because GitHub did not expose step durations
for the sampled runs, the current evidence cannot yet quantify repeated setup or artifact-generation
cost. The new phase timing covers the repository-owned `ci_driver.sh` phases first. This branch also
adds workflow-level timing around dependency sync and artifact migration in both `fast-feedback` and
`smoke-artifacts` using `scripts/dev/ci_step_timer.sh` so repeated setup costs surface directly in
CI logs. If repository phases remain dominant, the next probe should inspect required test grouping
or sharding without weakening coverage.

## Local Slowest Tests

Recent local PR-readiness runs used:

```bash
BASE_REF=origin/main PYTEST_NUM_WORKERS=8 scripts/dev/pr_ready_check.sh
```

They completed in about 293-299s with the full repository suite. These
local timings come from `PYTEST_NUM_WORKERS=8` on `auxme-imech036`, not from GitHub's
`ubuntu-latest` runner, so they are directional rather than CI-equivalent. The slowest reported
calls remain concentrated in example smoke coverage and one policy-stack smoke:

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
first places to inspect before changing test selection or sharding. This does not by itself justify
skipping or weakening those tests; it identifies where a narrower timing probe should go next.

## Low-Risk Runtime Reduction

The `fast-feedback` job still computes coverage JSON and compares against the cached baseline.
Routine pull-request runs no longer upload the full coverage HTML/database artifact on success.
Coverage artifacts remain uploaded when `fast-feedback` fails and on `main`, where baseline
maintenance happens.

The validation contract is preserved because the coverage comparison still runs on every pull
request; PR #1681 only removed the ordinary success-path upload of bulky coverage artifacts.

The aggregate `ci` job remains unchanged in structure and still gates on `fast-feedback`,
`smoke-artifacts`, and the new `examples-smoke` job, preserving the branch-protection-facing job
name and split-job semantics. The `test` CI phase now excludes `tests/examples`; the required
`examples-smoke` job runs the same manifest-driven example smoke harness separately so example
coverage remains required while its timing is no longer hidden inside the monolithic
`fast-feedback` test phase.

## Candidate Quick Wins

| candidate | risk | evidence needed before implementation |
| --- | --- | --- |
| Add phase timing around `lint`, `typecheck`, `test`, `smoke`, and `artifact-policy` in `scripts/dev/ci_driver.sh` | low, implemented in this branch | CI log shows per-phase durations without changing pass/fail semantics |
| Add timing around dependency sync and artifact migration in both CI jobs | low, **done** | CI log identifies whether repeated setup is material compared with test time |
| Split slow example smoke tests into a separately timed subgroup while keeping them required | low, **done** | CI run shows earlier failure signal or lower p90 without reducing coverage |
| Evaluate xdist `worksteal` scheduling for hosted `fast-feedback` tests | low as an experiment, rejected as a CI default in #1768 | PR CI showed no timing win and exposed test-order dependencies |
| Investigate whether docs-only PRs can use a reduced gate | medium-high | Maintainer decision plus path filter proof that benchmark, planner, workflow, config, and code changes still run full gates |

The recommended starting point is instrumentation, not deselection: use the new phase timestamps,
then decide whether the next target is test grouping, setup/cache behavior, or smoke-artifact
policy.

## Xdist Scheduler Slice

Issue [#1768](https://github.com/ll7/robot_sf_ll7/issues/1768) evaluated a narrow scheduler-only
runtime slice for the `fast-feedback` test phase. The latest hosted #1767 run still showed
`fast-feedback` as the wall-time limiter after example smoke separation:

- `ci_driver phase_end phase=test status=0 duration_seconds=563`
- `4486 passed, 9 skipped, 9 warnings in 541.51s`
- slowest non-example calls included `tests/adversarial/test_adversarial_search.py` at 24.30s,
  `tests/classic_interactions/test_headless_safety.py` at 19.68s, and
  `tests/test_load_states_and_record_video.py` at 18.84s.

The #1768 change keeps the same shared pytest wrapper and collected test paths, but lets the
wrapper accept `PYTEST_XDIST_DIST=<mode>` for targeted scheduler experiments. Hosted CI keeps the
historical `load` scheduler because hosted `worksteal` evidence was not good enough to use as the
default.

Local compatibility proof before implementation showed that the current xdist stack accepts
`worksteal` together with the repository's ordering flags:

```bash
PYTEST_NUM_WORKERS=4 uv run pytest -n 4 --dist worksteal --failed-first \
  tests/test_ci_script_contract.py tests/dev/test_resolve_pytest_workers.py -q
```

Result: `11 passed in 21.02s`.

The hosted experiment rejected `worksteal` as a CI default for this suite. Two hosted runs failed
after alternate scheduling exposed order dependencies, and the third run already exceeded the
previous #1767 `fast-feedback` timing window before completion. This is useful negative evidence:
scheduler mode alone is not the next reliable #1653 speed lever.

The first hosted #1768 run failed after `worksteal` changed scheduling enough to expose a shared
temporary path in `tests/sb3_test.py::test_can_load_model_snapshot`. That test wrote to
`./temp/ppo_model.zip`, which is unsafe under alternate xdist scheduling. The branch now uses
pytest's per-test `tmp_path` fixture for the saved model snapshot before rerunning hosted timing.

The second hosted run exposed a second scheduler-order dependency: the SoNIC probe import-state
test left `sys.modules["gym"]` set to a sentinel object after the test completed. The branch now
uses `monkeypatch.setitem()` so pytest restores the caller's original import state after the test.

The accepted #1768 outcome is therefore:

- keep `PYTEST_XDIST_DIST=<mode>` as an explicit wrapper knob for local or future hosted
  experiments,
- do not set `PYTEST_XDIST_DIST=worksteal` in hosted CI by default,
- keep the two test-isolation fixes because they remove real xdist-order coupling.

## Issue #1766 Local Proof

The example-smoke split was locally checked before PR handoff with:

```bash
scripts/dev/ci_driver.sh examples-smoke
```

Local result on 2026-05-30:

- `26 passed in 214.96s`
- `ci_driver phase_end phase=examples-smoke status=0 duration_seconds=226`

This proves the new CI phase exercises the manifest-driven examples harness locally. The required
before/after CI timing still needs to come from the PR run because the new `examples-smoke` job only
exists on this branch.

## Follow-Up

Future #1653 work should inspect CI logs from this branch, then decide whether setup, tests, smoke
scripts, or artifact handling dominate current runs. Avoid moving required checks until a
before/after CI run confirms the impact and a maintainer accepts the coverage boundary.
