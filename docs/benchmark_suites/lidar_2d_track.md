# LiDAR 2D Track

```yaml
suite_id: lidar_2d_track
benchmark_track: lidar_2d_v1
status: contract_and_training_smoke_surface
```

## Purpose

Document the current LiDAR observation-track smoke and compatibility surface. The first useful
checks prove observation metadata, adapter compatibility, and small learned-policy training
plumbing; they do not prove benchmark performance.

## Scenarios And Seeds

- Observation-track smoke config: `configs/benchmarks/lidar/observation_track_smoke_issue_1613.yaml`
- Scenario matrix: `configs/scenarios/sanity_v1.yaml`
- Seed policy: fixed seed `1613`
- Observation level: `lidar_2d`
- Runtime inputs: robot state, goal, and LiDAR rays
- Compatibility audit: `configs/benchmarks/lidar/planner_compatibility_issue_1614.yaml`
- Training smoke config: `configs/training/lidar/lidar_ppo_mlp_smoke_issue_1662.yaml`

## Eligible Planners

The compatibility audit classifies planners. Current first-track learned-policy work centers on
`ppo_lidar_mlp_gate_v1`; other planners must not enter benchmark rows until their LiDAR adapter
contracts and observation metadata pass.

## Metrics

Observation-level metadata, observation mode, runtime input keys, fallback policy, training smoke
completion, wall-clock time, training environment steps/sec, evaluation success/collision/SNQI, and
checkpoint artifact status.

## Canonical Commands

Contract tests:

```bash
uv run pytest -q tests/benchmark/test_lidar_observation_track.py \
  tests/benchmark/test_lidar_planner_compatibility.py
```

Training smoke:

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/training/train_ppo.py \
  --config configs/training/lidar/lidar_ppo_mlp_smoke_issue_1662.yaml \
  --log-level INFO \
  --log-file output/training/lidar_learned_policy/issue_1662/ppo_lidar_mlp_gate_v1_smoke.log
```

## Expected Runtime

Contract tests should be quick. The Issue #1662 training smoke recorded about one minute of wall
clock time on its local run, but runtime depends on hardware and worker setup.

## Claim Boundary

This track currently supports contract and training-smoke evidence only. A benchmark row needs a
durable checkpoint or adapter evidence bundle plus fail-closed result handling.

## Caveats

Do not treat a LiDAR training smoke as benchmark readiness. Output checkpoints under `output/` are
ignored local artifacts unless promoted to a durable store and represented by a tracked manifest.
