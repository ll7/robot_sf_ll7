# Optional training-family oversampling

Use `python -m scripts.training.train_ppo_with_family_weights --config <config>`
with the canonical trainer's CLI options. This additive entry point expands
`scenario_sampling.family_weights` into the existing per-scenario `weights`
before workers start; it delegates training, selection and manifests to the
canonical trainer. Without `family_weights`, behavior is unchanged. The separate
entry point avoids concurrent edits to the seed-policy trainer/sampler lane;
consolidating it into the canonical CLI is future integration work.

```yaml
scenario_sampling:
  strategy: random
  family_weights:
    bottleneck: 2.0
    doorway: 2.0
    exiting_elevator: 2.0
    exiting_room: 2.0
    frontal_approach: 2.0
```

Multipliers must be finite and at least one. Each family must match a sampleable
training scenario after filters. Nonmatching scenarios retain uniform weight one,
or their prior explicit per-scenario weight when `weights` was already configured.
The multipliers affect training only: checkpoint evaluation still cycles through
its original scenario profile. The example factor two is a declared pilot setting,
not an optimum estimated from released evaluation results. No frozen config changes.
