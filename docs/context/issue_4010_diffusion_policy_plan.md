# Issue #4010 Diffusion Policy First Slice

This first slice adds a Robot SF-native diffusion-policy local planner contract so later work can
train and benchmark it without inventing the runtime interface again.

## Claim Boundary

- Evidence status: `diagnostic-only`.
- Capability added: adapter-backed `diffusion_policy` planner family with aliases
  `diffusion_rl`, `diffusion_local_policy`, and `colson_style_diffusion`.
- Runtime scope: Robot SF state observations, a PyTorch-only graph-style robot/pedestrian encoder,
  one-step `(linear_velocity, angular_velocity)` diffusion action sampling, bounded command
  projection, and inference-time candidate guidance.
- Exclusions: no full benchmark campaign, no Slurm or graphics processing unit submission, no
  training success claim, and no paper or dissertation claim edit.

## Remaining Work

- Add a config-driven diffusion-policy training smoke that writes a checkpoint, normalizer, and
  manifest.
- Load a trained checkpoint in `map_runner` without `allow_untrained_smoke`.
- Run a representative diagnostic scenario and record fallback/degraded exclusions.
- Add a multimodality probe that classifies distinct action modes on a fixed conflict scenario.
