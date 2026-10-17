# Test value and differential evidence

All commands used `scripts/dev/run_worktree_shared_venv.sh -- uv run pytest ... -n 2 -q`
(or its explicit `--venv` equivalent in the base/integration checkout). No test
steps a protected seed; negative controls call only the admission guard.

## Four questions per test

No test adds a production test-only seam. `build_probe` is the real manipulation
runner's substrate builder, not a test stub. The table states behavior, credible
regression and the closest coverage gap; parametrizations share these answers.

| Test | Behavior / defect caught | Credible regression | Existing gap |
| --- | --- | --- | --- |
| `test_measured_profile_reaches_production_substrate` | Radius and desired distribution reach live PedState and survive refresh | Loader ignores a field or kernel retains 0.35 m | `test_pedestrian_desired_speed.py` constructs settings directly, bypassing scenario overrides |
| `test_exact_speed_profile_overrides` | Exact 1.29/0.65 means, 0.19 SD, dev seed, true truncation | Binding a rounded speed tier or dropping SD | Existing tier tests check 1.3/0.2, not these explicit overrides |
| `test_dev_only_seed_guard` | Refuse every non-dev seed, None, bool, float | Using the held-out blacklist as a study whitelist | Existing `test_runtime_seed_guard.py` tests held-out boundaries, not arbitrary non-dev integers |
| `test_dev_boundary_seeds_accepted` | Accept 1001 and 1030 | Off-by-one interval | Same guard gap |
| `test_truncation_has_no_clipped_boundary_mass` | True conditional normal, not clipped mass at zero | Removing resampling or ignoring `truncate` | Existing desired-speed tests cover legacy clipping |
| `test_summary_hand_fixture` | Tau-b=-0.5; A success=-1, contact=+1; C paired seconds=+2; empty time pairs stay null | Tie-skipping tau-a, reversed delta sign, time imputation, unpaired CI | Existing rank/fidelity summaries do not cover this paired factor/seed estimand |
| `test_summary_refuses_unpaired_grid` | Incomplete grids cannot produce apparently complete CIs | Dropping missing slots | Existing ranking accepts arbitrary keyed score sets |
| `test_combined_profile_dependency_is_not_silently_ignored` | Combined needs #10073; when present its real law/gain reach substrate | Silently deleting missing wall selection | #10073 selector tests cover its profile alone, not combined measured size/speed |
| `test_legacy_profile_retains_default_kernel_and_geometry` | Legacy geometry 0.40/0.35 and nominal speed 0.65 persist | Opt-in binding applied globally | Existing speed tests do not jointly check force/placement radius |
| `test_native_goal_seconds_reconstructed_without_failure_imputation` | Native successful normalized goal time becomes 30.8 s; failure stays null | Treating omitted raw seconds as absent success data | Existing metrics cover native normalization, not this study extractor |
| `test_plan_inventory_and_pre_dispatch_seed_guard` | Plan resolves 48 × 14 × 30; non-dev sentinel refused before dispatch | Accidentally inheriting release seed policy or shrinking roster | Existing release tests intentionally use a different seed policy |
| `test_default_settings_bytes_and_profile_roundtrip` | Default serialized hash unchanged; explicit controls survive replace | New fields changing default identity, or InitVar lost on copy | `sim_config_test.py` does not freeze this base-to-branch digest |

## Before / after

Base: `61cc91877a159e1d08b321543bd860042520b9ed` (fresh origin/main).

Copied tests for the real loader were run on that base:

```bash
uv run pytest tests/benchmark/test_pedestrian_sensitivity.py::test_exact_speed_profile_overrides \
  tests/benchmark/test_pedestrian_sensitivity.py::test_measured_profile_reaches_production_substrate -n 2 -q
```

**3 failed** for the intended defect: `simulation_config contains unknown keys:
desired_speed_mean, desired_speed_seed, desired_speed_std, desired_speed_truncated`
and, for the combined geometry test, `ped_force_radius`. This is a loader failure,
not a missing-module proof.

The base has no `dev_only` argument. An unadapted copied-test run fails with
`TypeError`; that alone is not behavior proof. A **test-only signature adapter**
accepting and discarding that new flag then delegates to the exact base guard:

```python
original = runtime_seed_guard.check_simulation_seed


def legacy_seed_policy(seed, *, boundary, authorization=None, dev_only=False):
    return original(seed, boundary=boundary, authorization=authorization)
```

Running `test_dev_only_seed_guard` through that adapter: **8 failed, 6 passed**.
The failures are `DID NOT RAISE ValueError` for 0, 1, 110, 141, 311, 1000, 1031
and None. This proves the base's blacklist cannot implement a dev-only study.
The held-out and malformed-input cases already refuse on base; their passing is
not counted as new behavior proof.

There is **no pre-existing study summary on base**. Missing-module failures in
an initial copied-test run are feature absence, **not arithmetic evidence**.
For mathematical proof the new harness was copied into the base proof checkout
and exercised with two explicit numerical controls, using the same fixture:

1. Substitute the base's `rank_metrics.kendall_tau_by_value` (skips tied pairs)
   for scipy tau-b: **1 failed**, `Obtained: -1.0; Expected: -0.5`.
2. Reverse `variant - base` to `base - variant`: **1 failed**, CI `[1, 1]`
   versus expected `[-1, -1]`.

The controls are not represented as an untouched-base numerical regression run.
They demonstrate that the tiny fixture detects plausible wrong summary math;
it also checks the independent `[2, 2]` seconds CI and null missing-time behavior.

Branch validation:

```bash
uv run pytest tests/benchmark/test_pedestrian_sensitivity.py \
  tests/test_scenario_loader_overrides.py tests/sim/test_pedestrian_desired_speed.py \
  tests/test_runtime_seed_guard.py -n 2 -q
uv run pytest tests/sim_config_test.py -n 2 -q
```

**79 passed** and **15 passed**. Integration checkout with the real #10073:
`test_pedestrian_sensitivity.py` plus `test_obstacle_force_profile.py`: **41 passed**.
The dependency test verifies its live surface-distance law, .001 gain and .25 radius.

The default settings digest was independently measured on base and matched on
branch: `0d054279f4a6e090d40e1fa54accaf58e955e624a3309472b44107f8790bb78d`.

CLI negative check rerun with non-dev sentinel 9001: `--mode smoke --seeds 9001 1001` refused before output
creation: `non-development seed 9001 at pedsens; use 1001..1030`.
Plan-only resolution wrote 120,960 intended episodes without dispatch.

Full test suite, Slurm behavior gate and complete 14-arm study were not run.
The scope is preparation and bounded smoke; these checks do not establish model
adoption, realistic pedestrians, or benchmark ranking conclusions.


## Sentinel rejection proof after independent review

The current rejection cases use non-development sentinels `9001` and `9002`,
not sealed seed-shaped literals. The CLI check above was rerun with `9001`
and refused before creating its output directory.

A process-local negative control delegates to the production admission function
with `dev_only=False`, disabling only the study whitelist while preserving
ordinary type and held-out validation. No production source is changed and no
episode is dispatched. Running the two sentinel guard cases and the planning
rejection case gives **3 failed**, each with `DID NOT RAISE ValueError`. The
ordinary implementation passes these same cases in the targeted suite.

These checks protect direct admission and the real `prepare()` boundary. A
removed whitelist or ignored `dev_only` flag admits the sentinels and fails the
tests. Prior sealed/retired blacklist cases could reject for a different reason;
these non-held-out sentinels isolate the study whitelist. The control is entirely
test-process-local and adds no production test-only seam.
