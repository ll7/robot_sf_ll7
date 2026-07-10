# Vectorized Environment Support

This note records the public vectorized training entry points and the worker modes covered by
targeted tests.

| Entry point | DummyVecEnv | SubprocVecEnv | ThreadedVecEnv | Start method | Status |
| --- | --- | --- | --- | --- | --- |
| `scripts/training/train_ppo.py` | Supported | Supported | Supported with `worker_mode: threaded` | `spawn` for subprocess mode | Primary PPO path; threaded mode is an opt-in in-process rollout path. |
| `scripts/training/train_ppo_with_pretrained_policy.py` | Supported | Supported | Not exposed | `spawn` | Supported PPO fine-tuning path. |
| `scripts/training/train_sac_sb3.py` | Supported | Supported when `num_envs > 1` | Not exposed | `spawn` | Supported SAC path. |
| `scripts/multi_extractor_training.py` | Supported for `single-thread` | Supported for `vectorized` | Not exposed | `spawn` | Legacy extractor-sweep path; retained for compatibility. |

`ThreadedVecEnv` batches independent reset and step calls through a shared in-process worker pool while
preserving Stable-Baselines3's `VecEnv` result, auto-reset, and terminal-observation behavior. Use it
only when each environment is safe to step concurrently in one process. It avoids subprocess
serialization, but it does not provide process isolation and does not by itself establish a rollout
throughput improvement. The default `worker_mode: auto` remains unchanged and chooses subprocess
workers when more than one environment is requested.

Select the mode explicitly in a primary PPO config:

```yaml
num_envs: 4
worker_mode: threaded
```

With `num_envs: 1`, `threaded` resolves to the existing `DummyVecEnv` fallback. This preserves the
single-environment execution path; no performance result is implied by this configuration.

`spawn` is the portable subprocess contract. It exercises actual child-process imports and catches
closures, globals, and environment factories that only work under `fork`. Tests should cover
`DummyVecEnv` and at least one real `SubprocVecEnv(start_method="spawn")` reset/step/close smoke for
the primary training path, plus unit coverage that secondary entry points request `spawn`.

Supported vector-env smoke tests intentionally use tiny scenario/config slices. They validate
construction and Gymnasium reset/step/close behavior, not training quality or policy performance.
