# Issue 646 Social-Navigation-PyEnvs SocialForce and SFM Integration Note

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#642` Prototype fail-fast Social-Navigation-PyEnvs source-harness reproduction
- `robot_sf_ll7#644` Prototype benchmark-facing Social-Navigation-PyEnvs ORCA integration
- `robot_sf_ll7#646` Prototype benchmark-facing Social-Navigation-PyEnvs SocialForce/SFM integration

## Goal

Prototype benchmark-facing upstream wrappers for non-trainable Social-Navigation-PyEnvs
force-model policies beyond ORCA, starting with:

- `crowd_nav.policy_no_train.socialforce.SocialForce`
- `crowd_nav.policy_no_train.sfm_helbing.SFMHelbing`

The issue stays wrapper-thin and provenance-first:

- preserve the upstream `JointState -> ActionXY` policy contract,
- keep the `ActionXY -> unicycle_vw` projection explicit,
- prove real Robot SF execution before making any quality claim,
- and record blockers instead of silently patching upstream behavior.

## Implementation shape

Robot SF now exposes two benchmark-facing prototype planner entries:

- `social_navigation_pyenvs_socialforce`
- `social_navigation_pyenvs_sfm_helbing`

Both use the same integration boundary:

1. map Robot SF SocNav observations into the upstream Social-Navigation-PyEnvs `JointState`
   contract,
2. call the upstream non-trainable policy `predict()` method,
3. project the resulting `ActionXY` command into benchmark-visible `unicycle_vw`.

Canonical local adapter entrypoint:

- `robot_sf/planner/social_navigation_pyenvs_force_model.py`

Canonical benchmark configs:

- `configs/algos/social_navigation_pyenvs_socialforce_probe.yaml`
- `configs/algos/social_navigation_pyenvs_sfm_helbing_probe.yaml`
- `configs/benchmarks/social_navigation_pyenvs_force_models_probe.yaml`

## Canonical validation commands

Targeted adapter and metadata tests:

```bash
uv run pytest -q \
  tests/planner/test_social_navigation_pyenvs_force_model.py \
  tests/benchmark/test_algorithm_metadata_contract.py \
  tests/benchmark/test_map_runner_utils.py \
  -k 'social_navigation_pyenvs_force_model or social_navigation_pyenvs_force_models or force_model_metadata'
```

Sanity-surface benchmark proof:

```bash
uv run --with socialforce python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/social_navigation_pyenvs_force_models_probe.yaml \
  --label issue646_social_navigation_pyenvs_force_models_retry \
  --log-level WARNING
```

## Observed benchmark result

Canonical campaign:

- `output/benchmarks/camera_ready/social_navigation_pyenvs_force_models_probe_issue646_social_navigation_pyenvs_force_models_retry_20260320_143839`

Observed split:

- `social_navigation_pyenvs_sfm_helbing`
  - status: `ok`
  - episodes: `3`
  - success: `1.0000`
  - collisions: `0.0000`
  - time_to_goal_norm: `0.9244`
  - path_efficiency: `1.0000`
  - jerk: `0.0164`
  - projection_rate: `0.0000`
  - infeasible_rate: `0.0000`
- `social_navigation_pyenvs_socialforce`
  - status: `partial-failure`
  - episodes: `0`
  - failed jobs: `3`
  - blocker: `TypeError("Simulator.__init__() got an unexpected keyword argument 'initial_speed'")`

Interpretation:

- `sfm_helbing` is a real benchmark-facing integration proof on the sanity surface.
- `socialforce` is not yet reproducible in the current runtime because the installed
  `socialforce==0.2.3` API is incompatible with the upstream Social-Navigation-PyEnvs wrapper.
- this issue does not justify paper-surface quality claims for either planner.

## Why SocialForce is blocked

The upstream Social-Navigation-PyEnvs `SocialForce` wrapper constructs the external simulator as:

- `socialforce.Simulator(..., initial_speed=..., v0=..., sigma=...)`

But the currently resolved package API is:

- `socialforce==0.2.3`
- `Simulator.__init__(..., ped_space=None, ped_ped=None, field_of_view=None, delta_t=0.4, tau=0.5, oversampling=10, dtype=None, integrator=None)`

So the benchmark failure is not a Robot SF observation-mapping failure.
It is a concrete upstream dependency/API mismatch.

## Recommendation

Recommendation: `prototype only`

What is justified now:

1. keep `social_navigation_pyenvs_sfm_helbing` as an experimental upstream-backed planner entry,
2. keep `social_navigation_pyenvs_socialforce` documented as blocked,
3. avoid paper-facing performance claims until a larger comparison surface is run,
4. open a dedicated follow-up if SocialForce runtime compatibility is worth pursuing.

What is not justified now:

- claiming SocialForce works here,
- merging a hidden compatibility shim for the external `socialforce` package,
- treating the 3-episode sanity success for `sfm_helbing` as benchmark-quality evidence.

## Final verdict

- `social_navigation_pyenvs_sfm_helbing`: `integration proof only`
- `social_navigation_pyenvs_socialforce`: `blocked by dependency API mismatch`
