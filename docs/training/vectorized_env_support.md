# Vectorized Environment Support

This note records the public vectorized training entry points and the worker modes covered by
targeted tests.

| Entry point | DummyVecEnv | SubprocVecEnv | Start method | Status |
| --- | --- | --- | --- | --- |
| `scripts/training/train_ppo.py` | Supported | Supported | `spawn` | Primary supported PPO path. |
| `scripts/training/train_ppo_with_pretrained_policy.py` | Supported | Supported | `spawn` | Supported PPO fine-tuning path. |
| `scripts/training/train_sac_sb3.py` | Supported | Supported when `num_envs > 1` | `spawn` | Supported SAC path. |
| `scripts/multi_extractor_training.py` | Supported for `single-thread` | Supported for `vectorized` | `spawn` | Legacy extractor-sweep path; retained for compatibility. |

`spawn` is the portable subprocess contract. It exercises actual child-process imports and catches
closures, globals, and environment factories that only work under `fork`. Tests should cover
`DummyVecEnv` and at least one real `SubprocVecEnv(start_method="spawn")` reset/step/close smoke for
the primary training path, plus unit coverage that secondary entry points request `spawn`.

Supported vector-env smoke tests intentionally use tiny scenario/config slices. They validate
construction and Gymnasium reset/step/close behavior, not training quality or policy performance.
