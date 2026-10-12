# Prediction Planner Baseline (`prediction_planner`)

This document describes the predictive local planner integrated in `robot_sf`.

## Summary

- Planner key: `prediction_planner` (aliases: `predictive`, `prediction`)
- Category: learning-based local planner adapter
- Adapter class: `PredictionPlannerAdapter` in `robot_sf/planner/socnav.py`
- Model implementation: `robot_sf/planner/predictive_model.py`

The planner predicts short-horizon pedestrian motion and scores sampled robot actions with a multi-term objective (goal progress, collision/near risk, TTC, occupancy, and progress-risk coupling).

## Intended Benchmark Role

- Current readiness tier: `experimental`
- Designed for local planner benchmarking under SocNav structured observations.
- Should be reported as experimental unless explicit benchmark-promotion criteria are met.

## Key Configuration

Use `configs/algos/prediction_planner_camera_ready.yaml` for the current benchmark-ready config profile.

Main parameter groups:
- Model and rollout: `predictive_model_id`, `predictive_horizon_steps`, `predictive_rollout_dt`
- Objective weights: `predictive_goal_weight`, `predictive_collision_weight`, `predictive_ttc_weight`, `occupancy_weight`
- Risk-progress controls: `predictive_progress_risk_*`, `predictive_hard_clearance_*`
- Adaptive search: `predictive_adaptive_horizon_enabled`, `predictive_horizon_boost_steps`, `predictive_near_field_*`

## Physical robot radius

After `bind_env`, prediction clearance and contact penalties use the live robot's
disc radius, matching static footprint checks and trajectory metrics. The
configured `predictive_robot_radius` remains the fallback for unbound calls and is
not mutated by binding. Rebinding an environment without a physical radius clears
the previous override. The `center_v1` and `surface_v2` distance definitions and
their configured safety gaps retain their meanings.

`tests/planner/test_prediction_physical_radius.py` compares surface clearance with
the metric calculation and checks both candidate scorers. This does not implement
capsule geometry (#10093) or the opt-in metric successor in PR #10231 (#10079).
Pedestrian radius migration remains the existing #10074/#10075 work; this repair
only resolves the robot radius. Planner behaviour changes when the configured
radius disagrees with the bound plant, so a hosted behaviour receipt is required.

## Model Resolution

The planner resolves checkpoints via `model/registry.yaml` using `predictive_model_id`.

Recommended default model id:
- `predictive_proxy_selected_v2_full`

If the checkpoint cannot be resolved/loaded and fallback is disabled, the planner fails fast.

## Citations and Provenance

This implementation is inspired by:

1. Chen et al. (2020), *Robot Navigation in Crowds by Graph Convolutional Networks With Attention Learned From Human Gaze*.
2. Alyassi et al. (2025), *Social robot navigation: a review and benchmarking of learning-based methods*.

Important: this is an in-repo reimplementation and benchmark adaptation, not a byte-equivalent reproduction of upstream code.

## Known Limitations

- Hard-seed success improvements are still limited; current improvements mostly raise safety margin/clearance.
- Model availability is local-artifact dependent unless checkpoint distribution is formalized for CI/users.

## Minimal Usage

```bash
uv run python scripts/classic_benchmark_full.py \
  --algo prediction_planner \
  --algo-config configs/algos/prediction_planner_camera_ready.yaml
```
