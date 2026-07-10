# Vectorized multi-environment rollouts

## Goal

Issue #4981 needs a Stable-Baselines3-compatible way to collect several Robot SF rollout steps
without requiring one Python process per environment. This design adds an opt-in in-process threaded
mode to the primary PPO trainer while keeping the established `DummyVecEnv` and `SubprocVecEnv`
paths unchanged.

## Contract delta

- `scripts/training/train_ppo.py` accepts `worker_mode: threaded` when `num_envs > 1`.
- The mode uses `robot_sf.training.threaded_vec_env.ThreadedVecEnv`, which preserves the
  Stable-Baselines3 `VecEnv` observations, rewards, done flags, automatic resets, terminal
  observations, and information dictionaries.
- With one environment, `threaded` resolves to `dummy`, preserving the existing single-environment
  fallback.

## Assumption and boundary

The reversible implementation assumes independent Robot SF environments can be stepped concurrently
when their numerical kernels release the Python global interpreter lock. It deliberately does not
change simulator state layout or claim a batched Numba force/LiDAR kernel. Threaded workers share a
process, so callers needing process isolation should continue using `worker_mode: subproc`.

## Proof and follow-up

The implementation is proven by Stable-Baselines3 lifecycle tests plus a real Robot SF reset/step
smoke. It is not throughput evidence. The remaining acceptance step is a reproducible standard-PPO
configuration measurement that compares `dummy`, `subproc`, and `threaded`, reports environment
steps per second, and only promotes the issue's >3x claim if the exact configuration, host, and
sample duration support it.
