# Issue #1081 Observation Noise

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1081>

## Scope

Issue #1081 adds opt-in benchmark observation-noise injection for planner inputs. The layer is
applied after the environment produces an observation and before the planner receives it, so metrics
still evaluate the simulator's ground-truth trajectory.

The supported robustness profile fields are:

* pose and heading Gaussian noise (`pose_noise_std_m`, `heading_noise_std_rad`)
* lidar dropout (`lidar_dropout_prob`, `lidar_dropout_value`)
* pedestrian false negatives and false positives
  (`pedestrian_false_negative_prob`, `pedestrian_false_positive_prob`)

The shipped reusable smoke preset is:

* `configs/benchmarks/observation_noise/robustness_smoke_v1.yaml`

## Provenance

Each episode records:

* `observation_noise`
* `observation_noise_hash`
* `observation_noise_stats`
* matching `scenario_params.observation_noise_profile` when enabled
* matching `scenario_params.observation_noise_hash` when enabled

Resume identity includes the noise hash only when noise is enabled, so clean default run IDs remain
unchanged while clean and noisy runs cannot skip each other.
Campaign preflight, matrix summary, manifest, run summary, and campaign summary surfaces also carry
the profile/hash.

## Interpretation Limit

The profile is a controlled benchmark robustness perturbation, not a calibrated real sensor model.
Use it to compare sensitivity under the same configured corruption profile, not to claim real-world
sensor performance.

## Validation

Focused checks passed on 2026-05-09:

```bash
uv run pytest \
  tests/benchmark/test_observation_noise.py \
  tests/benchmark/test_map_runner_utils.py::test_run_map_episode_smoke \
  tests/benchmark/test_map_runner_resume_identity.py \
  tests/benchmark/test_camera_ready_campaign.py::test_load_campaign_config_resolves_observation_noise_profile \
  tests/contract/test_episode_schema.py -q
```

Result: `12 passed`.

The smoke campaign also ran successfully:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/observation_noise_goal_smoke.yaml \
  --mode run \
  --output-root output/benchmarks/issue_1081 \
  --campaign-id observation_noise_goal_smoke_verify \
  --skip-publication-bundle
```

Result: `total_runs=1`, `successful_runs=1`, `total_episodes=1`.

The generated episode JSONL recorded `observation_noise.profile=robustness_smoke_v1`,
`observation_noise.enabled=true`, and `observation_noise_hash=1dead4195d60`.
