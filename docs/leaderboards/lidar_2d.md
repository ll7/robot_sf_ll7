# LiDAR 2D Leaderboard

This page tracks LiDAR learned-policy smoke evidence. The first row is a training-smoke plumbing
result, not a benchmark row or model promotion.

| planner | suite | success | collision | near_miss | low_progress | min_distance | runtime | benchmark_track | evidence_uri | status | claim_boundary |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| `ppo_lidar_mlp_gate_v1` | `lidar_2d_training_smoke` | `0.0000` | `0.0000` | `not_recorded` | `not_recorded` | `not_recorded` | `59.7432s` | `lidar_2d_training_smoke` | [`docs/context/evidence/issue_1662_lidar_ppo_smoke_summary.json`](../context/evidence/issue_1662_lidar_ppo_smoke_summary.json) | `completed_smoke_not_benchmark_evidence` | Documents fixed LiDAR PPO MLP training/eval plumbing only; no durable checkpoint or promotion claim. |

Longer LiDAR training, deployment-adapter, or benchmark rows should not be added here until a
durable checkpoint or tracked benchmark evidence bundle exists.
