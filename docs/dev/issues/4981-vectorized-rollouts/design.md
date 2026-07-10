# Vectorized Multi-Environment Rollouts #4981

## Goal

Issue #4981 needs a Stable-Baselines3-compatible way to collect several Robot SF rollout steps
without requiring one Python process per environment. This design adds opt-in in-process threaded
execution and coordinated cross-environment light detection and ranging (LiDAR) collection while
keeping the established `DummyVecEnv`, `SubprocVecEnv`, and scalar LiDAR paths unchanged.

## Contract delta

- `scripts/training/train_ppo.py` accepts `worker_mode: threaded` when `num_envs > 1`.
- The mode uses `robot_sf.training.threaded_vec_env.ThreadedVecEnv`, which preserves the
  Stable-Baselines3 `VecEnv` observations, rewards, done flags, automatic resets, terminal
  observations, and information dictionaries.
- With one environment, `threaded` resolves to `dummy`, preserving the existing single-environment
  fallback.
- `worker_mode: threaded_lidar_batch` uses the same threaded vector-environment lifecycle and
  coordinates homogeneous static-obstacle LiDAR rows during each environment step. The mode also
  resolves to `dummy` for one environment.
- `robot_sf.sensor.range_sensor.raycast_obstacles_batch` accepts padded static-obstacle arrays for
  several environments and evaluates them in one compiled central processing unit (CPU)
  dispatch. The coordinator selects it only for the explicit `threaded_lidar_batch` mode.
- The compiled LiDAR raycast and postprocessing entry points release the Python global interpreter
  lock while executing, so independent scalar threaded scans can make progress concurrently.
- A one-environment LiDAR batch calls the existing `raycast_obstacles` kernel directly, preserving
  its output bit for bit.

## Assumption and boundary

The reversible threaded implementation assumes independent Robot SF environments can be stepped
concurrently when their numerical kernels release the Python global interpreter lock. Threaded
workers share a process, so callers needing process isolation should continue using
`worker_mode: subproc`.

The LiDAR coordinator batches only the static-obstacle raycast. All participating workers must
reach the same number of LiDAR calls per step with a common ray count and homogeneous array dtypes.
It pads obstacle rows, then uses one Numba dispatch while the other environment workers wait,
avoiding nested worker dispatches. A timeout or incompatible row fails the coordinated mode
closed instead of falling back silently. Reset-time and automatic-reset scans remain scalar.

## Proof and follow-up

The implementation is proven by Stable-Baselines3 lifecycle tests, a real Robot SF reset/step smoke,
and a coordinated rollout test that compares complete LiDAR observations bit for bit with scalar
threaded execution. This is implementation-integrity evidence, not throughput evidence.

Run the bounded four-mode central processing unit (CPU) comparator with:

```bash
uv run python scripts/validation/run_vecenv_worker_mode_throughput.py \
  --config configs/training/lidar/lidar_ppo_mlp_smoke_issue_1662.yaml \
  --output output/vecenv_throughput.json
```

The JSON records repeated throughput samples, median speedup against a separately measured
one-environment dummy fallback, failures, and config/scenario/commit/host provenance. Its explicit
claim boundary is diagnostic-only. The issue's >3x claim may only be promoted after a reviewed,
sufficiently long standard-config measurement supports it; a bounded smoke is not that evidence.
