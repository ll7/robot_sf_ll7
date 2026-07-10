# Vectorized Environment Support

This note records the public vectorized training entry points and the worker modes covered by
targeted tests.

| Entry point | DummyVecEnv | SubprocVecEnv | ThreadedVecEnv | Start method | Status |
| --- | --- | --- | --- | --- | --- |
| `scripts/training/train_ppo.py` | Supported | Supported | Supported with `worker_mode: threaded` or `threaded_lidar_batch` | `spawn` for subprocess mode | Primary PPO path; both threaded modes are opt-in in-process rollout paths. |
| `scripts/training/train_ppo_with_pretrained_policy.py` | Supported | Supported | Not exposed | `spawn` | Supported PPO fine-tuning path. |
| `scripts/training/train_sac_sb3.py` | Supported | Supported when `num_envs > 1` | Not exposed | `spawn` | Supported SAC path. |
| `scripts/multi_extractor_training.py` | Supported for `single-thread` | Supported for `vectorized` | Not exposed | `spawn` | Legacy extractor-sweep path; retained for compatibility. |

`ThreadedVecEnv` batches independent reset and step calls through a shared in-process worker pool while
preserving Stable-Baselines3's `VecEnv` result, auto-reset, and terminal-observation behavior. Use it
only when each environment is safe to step concurrently in one process. It avoids subprocess
serialization, but it does not provide process isolation and does not by itself establish a rollout
throughput improvement. The compiled LiDAR raycast and postprocessing sections release the Python
global interpreter lock during execution; the surrounding environment step remains ordinary Python.
The default `worker_mode: auto` remains unchanged and chooses subprocess workers when more than one
environment is requested.

Select the mode explicitly in a primary PPO config:

```yaml
num_envs: 4
worker_mode: threaded
```

To coordinate homogeneous static-obstacle LiDAR rows into one compiled batch during each threaded
environment step, select the separate opt-in mode:

```yaml
num_envs: 4
worker_mode: threaded_lidar_batch
```

With `num_envs: 1`, both threaded modes resolve to the existing `DummyVecEnv` fallback. This
preserves the single-environment execution path; no performance result is implied by either
configuration.

`spawn` is the portable subprocess contract. It exercises actual child-process imports and catches
closures, globals, and environment factories that only work under `fork`. Tests should cover
`DummyVecEnv` and at least one real `SubprocVecEnv(start_method="spawn")` reset/step/close smoke for
the primary training path, plus unit coverage that secondary entry points request `spawn`.

Supported vector-env smoke tests intentionally use tiny scenario/config slices. They validate
construction and Gymnasium reset/step/close behavior, not training quality or policy performance.

## Opt-in light detection and ranging (LiDAR) kernel batching

`robot_sf.sensor.range_sensor.raycast_obstacles_batch` provides one central processing unit
(CPU)-safe cross-environment kernel path for callers that can collect several LiDAR inputs together.
It accepts one padded static-obstacle tensor, explicit obstacle counts, scanner positions, and ray
angles, then invokes the established obstacle-raycast arithmetic for every environment in one
compiled dispatch.

The `threaded_lidar_batch` worker mode binds one coordinator to its environment-step workers. Each
worker computes dynamic-object raycasts independently, submits its static-obstacle inputs, and waits
while the coordinator pads obstacle rows and invokes the batch kernel once. Participating
rows must share one ray count and array dtypes, and each worker must reach the same number of LiDAR
calls per step. Incompatible or incomplete batches fail closed; reset-time scans remain scalar.

A batch of one calls the scalar `raycast_obstacles` kernel directly, and primary PPO training
resolves the one-environment worker mode to `DummyVecEnv` before constructing a coordinator.
Multi-environment tests require complete LiDAR observations to be bit-identical to scalar threaded
execution. These checks establish compatibility and rollout integration only; the standard-config
throughput comparison required by issue #4981 remains separate.
