# Vectorized Multi-Environment Rollouts #4981

## Goal

Issue #4981 needs a Stable-Baselines3-compatible way to collect several Robot SF rollout steps
without requiring one Python process per environment. This design adds opt-in in-process threaded
execution and an opt-in cross-environment light detection and ranging (LiDAR) kernel adapter while
keeping the established `DummyVecEnv`, `SubprocVecEnv`, and scalar LiDAR paths unchanged.

## Contract delta

- `scripts/training/train_ppo.py` accepts `worker_mode: threaded` when `num_envs > 1`.
- The mode uses `robot_sf.training.threaded_vec_env.ThreadedVecEnv`, which preserves the
  Stable-Baselines3 `VecEnv` observations, rewards, done flags, automatic resets, terminal
  observations, and information dictionaries.
- With one environment, `threaded` resolves to `dummy`, preserving the existing single-environment
  fallback.
- `robot_sf.sensor.range_sensor.raycast_obstacles_batch` accepts padded static-obstacle arrays for
  several environments and evaluates them in one compiled central processing unit (CPU) dispatch.
  Callers opt in explicitly; no environment or trainer selects it by default.
- A one-environment LiDAR batch calls the existing `raycast_obstacles` kernel directly, preserving
  its output bit for bit.

## Assumption and boundary

The reversible threaded implementation assumes independent Robot SF environments can be stepped
concurrently when their numerical kernels release the Python global interpreter lock. Threaded
workers share a process, so callers needing process isolation should continue using
`worker_mode: subproc`.

The LiDAR adapter batches only the static-obstacle raycast and requires a common ray count plus
padded obstacle storage with explicit per-environment counts. It deliberately uses one sequential
Numba dispatch rather than nested CPU parallelism, and it is not yet wired into environment step
coordination. This contract establishes numerical equivalence, not a throughput improvement.

## Proof and follow-up

The implementation is proven by Stable-Baselines3 lifecycle tests, a real Robot SF reset/step smoke,
and synthetic multi-environment tests that compare every batched LiDAR row with the scalar kernel.
It is not throughput evidence. The remaining acceptance steps are integration of batch collection
into a rollout path and a reproducible standard-PPO configuration measurement that compares worker
and kernel modes. The issue's >3x claim may only be promoted if the exact configuration, host, and
sample duration support it.
