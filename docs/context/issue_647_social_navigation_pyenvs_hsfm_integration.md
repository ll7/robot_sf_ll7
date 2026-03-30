# Issue 647 Social-Navigation-PyEnvs HSFM Integration Note

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#642` Prototype fail-fast Social-Navigation-PyEnvs source-harness reproduction
- `robot_sf_ll7#644` Prototype benchmark-facing Social-Navigation-PyEnvs ORCA integration
- `robot_sf_ll7#646` Prototype benchmark-facing Social-Navigation-PyEnvs SocialForce/SFM integration
- `robot_sf_ll7#647` Prototype benchmark-facing Social-Navigation-PyEnvs HSFM integration

## Goal

Prototype one benchmark-facing headed-force-model planner entry from
`Social-Navigation-PyEnvs`, starting with:

- `crowd_nav.policy_no_train.hsfm_new_guo.HSFMNewGuo`

The integration stays provenance-first:

- preserve the upstream headed `JointState -> ActionXYW/NewHeadedState` logic,
- expose the headed action contract explicitly,
- avoid hidden rewrites of lateral body velocity into native differential-drive control,
- and prove a real Robot SF benchmark run before making quality claims.

## Implementation shape

Robot SF now exposes one benchmark-facing prototype planner entry:

- `social_navigation_pyenvs_hsfm_new_guo`

Canonical local adapter entrypoint:

- `robot_sf/planner/social_navigation_pyenvs_hsfm.py`

Canonical benchmark configs:

- `configs/algos/social_navigation_pyenvs_hsfm_new_guo_probe.yaml`
- `configs/benchmarks/social_navigation_pyenvs_hsfm_probe.yaml`
- `configs/benchmarks/paper_experiment_matrix_v1_social_navigation_pyenvs_hsfm_new_guo_only.yaml`

## Observation and action contract

The upstream HSFM path is meaningfully different from ORCA and plain SFM:

- upstream self state uses `FullStateHeaded`
- upstream action can be:
  - `ActionXYW(bvx, bvy, w)` for Euler-style updates
  - `NewHeadedState(px, py, theta, bvx, bvy, w)` for RK45-style updates

Robot SF now exposes the missing headed self-state explicitly in the SocNav observation:

- `robot.velocity_xy`
- `robot.angular_velocity`

The adapter maps Robot SF observations into upstream headed state as:

1. derive body-frame self velocity from `robot.velocity_xy` and `robot.heading`,
2. pass explicit `robot.angular_velocity`,
3. call upstream `HSFMNewGuo.predict()`,
4. project body-frame lateral intent into benchmark-visible `unicycle_vw`.

Projection policy:

- `body_velocity_heading_safe_to_unicycle_vw`

This projection is explicit because upstream HSFM can command body-frame lateral motion that a
differential-drive Robot SF benchmark entry cannot execute natively.

## Canonical validation commands

Targeted adapter and metadata tests:

```bash
uv run pytest -q \
  tests/planner/test_social_navigation_pyenvs_hsfm.py \
  tests/test_socnav_env_integration.py \
  tests/benchmark/test_algorithm_metadata_contract.py \
  tests/benchmark/test_map_runner_utils.py \
  -k 'social_navigation_pyenvs_hsfm or socnav_structured_observation_exposes_robot_angular_velocity or hsfm_metadata or hsfm_as_socnav or hsfm_preserves_provenance'
```

Sanity-surface benchmark proof:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/social_navigation_pyenvs_hsfm_probe.yaml \
  --label issue647_social_navigation_pyenvs_hsfm \
  --log-level WARNING
```

Paper-surface comparison:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_social_navigation_pyenvs_hsfm_new_guo_only.yaml \
  --label issue647_social_navigation_pyenvs_hsfm_paper_surface \
  --log-level WARNING
```

## Result

Observed evidence now exists on both the sanity surface and the canonical paper surface.

Sanity-surface proof campaign:

- `output/benchmarks/camera_ready/social_navigation_pyenvs_hsfm_probe_issue647_social_navigation_pyenvs_hsfm_20260320_151625`

Sanity-surface result:

- `social_navigation_pyenvs_hsfm_new_guo`
  - status: `ok`
  - episodes: `3`
  - success: `1.0000`
  - collisions: `0.0000`
  - runtime: `4.2434s`
  - projection rate: `0.0000`
  - infeasible rate: `0.0000`

Paper-surface comparison campaign:

- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_social_navigation_pyenvs_hsfm_new_guo_only_issue647_social_navigation_pyenvs_hsfm_paper_surface_20260320_151645`

Paper-surface result:

- `social_navigation_pyenvs_hsfm_new_guo`
  - status: `ok`
  - episodes: `141`
  - success: `0.0142`
  - collisions: `0.0709`
  - runtime: `33.5952s`
  - SNQI: `-0.2795`
  - projection rate: `0.0000`
  - infeasible rate: `0.0000`

Comparison against current reference rows:

| Planner | Success | Collisions | Runtime (s) | SNQI |
|---|---:|---:|---:|---:|
| Frozen `orca` | `0.2340` | `0.0426` | `98.2864` | `-0.2325` |
| `ppo` | `0.2695` | `0.1844` | `77.5932` | `-0.3541` |
| `social_navigation_pyenvs_orca` | `0.0213` | `0.0638` | `37.3879` | `-0.2974` |
| `social_navigation_pyenvs_hsfm_new_guo` | `0.0142` | `0.0709` | `33.5952` | `-0.2795` |

Interpretation:

- the adapter is operational and the headed contract is explicit,
- the planner is faster than frozen canonical `orca`,
- but it is not competitive on benchmark outcomes,
- and it does not outperform the already weak upstream `social_navigation_pyenvs_orca` prototype on
  success or collisions.

Why the result still matters:

- this is the first benchmark-facing upstream-headed planner entry from
  `Social-Navigation-PyEnvs`,
- it proves Robot SF can carry explicit body-frame and angular-rate contracts without hidden
  approximation,
- and it narrows the remaining search space by showing that simply adding another non-trainable
  upstream planner family is not enough to beat the current benchmark leaders.

## Final verdict

- `social_navigation_pyenvs_hsfm_new_guo`: `integration proof only`

It is source-attributable and benchmark-runnable, but not a benchmark-quality upgrade.
