# Issue #4010 Diffusion Policy Integration Report

Plain-language status: issue #4010 now has a Robot SF-native diffusion-policy local planner
surface, CPU-scale smoke training artifacts, and a checkpoint-backed map-runner load contract. The
current evidence is diagnostic-only. It does not prove navigation quality, benchmark performance, or
a COLSON reproduction.

## Delivered Contract

- Runtime adapter: `DiffusionPolicyAdapter` supports Robot SF state observations, a PyTorch-only
  robot/pedestrian graph-style encoder, one-step `(linear_velocity, angular_velocity)` diffusion
  action sampling, bounded command projection, and inference-time candidate guidance.
- Map-runner policy: `diffusion_policy` is registered with aliases `diffusion_rl`,
  `diffusion_local_policy`, and `colson_style_diffusion`.
- Training smoke: `scripts/training/train_diffusion_policy.py --config <path>` writes a tiny CPU
  smoke checkpoint, normalizer, and provenance manifest.
- Checkpoint-backed load: map-runner construction can load the smoke checkpoint and normalizer
  without `allow_untrained_smoke`.
- New consolidation capability: `--write-diagnostic-packet` writes a compact
  `diffusion_policy_diagnostic_packet.v1` JSON packet that combines smoke artifact provenance,
  map-runner load metadata when supplied by callers, claim boundary, and remaining blockers.

## Claim Boundary

- Evidence status: `diagnostic-only`.
- This is a Robot SF-native, COLSON-style implementation lane, not a COLSON reproduction.
- Fallback, degraded, untrained random inference, and synthetic CPU smoke fixtures are not benchmark
  success evidence.
- Exclusions for this slice: no full benchmark campaign, no Slurm or graphics processing unit
  submission, no paper or dissertation claim edit.

## Remaining Blockers

- Representative rollout: run a diagnostic scenario with explicit fallback/degraded exclusions.
- Multimodal action probe: classify sampled action modes on a fixed conflict case, for example
  pass-left, pass-right, and slow-or-wait clusters.
- Benchmark claim: requires benchmark-grade evidence with comparator, seeds, configuration
  provenance, and failure-mode classification.
- Research comparison: PPO and Recurrent PPO comparison remains a later evaluation lane, not covered
  by the smoke artifact contract.

## Validation Entry Points

- `uv run pytest tests/planner/test_diffusion_policy.py -q`
- `uv run pytest tests/benchmark/test_map_runner_diffusion_policy.py -q`
- `uv run pytest tests/training/test_diffusion_policy_actor.py -q`
- `uv run ruff check robot_sf/planner/diffusion_policy.py robot_sf/benchmark/map_runner_policies/diffusion_policy.py robot_sf/training/diffusion_policy.py tests/planner/test_diffusion_policy.py tests/benchmark/test_map_runner_diffusion_policy.py tests/training/test_diffusion_policy_actor.py`
