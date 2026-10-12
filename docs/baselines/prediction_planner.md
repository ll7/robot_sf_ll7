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

## Model Resolution

The planner resolves checkpoints via `model/registry.yaml` using `predictive_model_id`.

Recommended default model id:
- `predictive_proxy_selected_v2_full`

If the checkpoint cannot be resolved/loaded and fallback is disabled, the planner fails fast.

## Bound bicycle candidate motion

After `bind_env`, single-action and sequence searches simulate bicycle candidates
through `PlannerActionAdapter` and `BicycleDriveRobot` at the observation's control
timestep. Acceleration, asymmetric braking, steering curvature and limited reverse
therefore use the execution plant's settings. Static footprint scoring includes
motion while a stop command brakes. Unbound calls retain the unicycle fallback.

Bound forecasts require the rollout timestep to be an integer multiple of the
observed control timestep within a `1e-6` relative ratio tolerance for float32
observation rounding. Each sample advances only whole, fixed control ticks;
nondivisible ratios (for example control 0.15 s / forecast 0.2 s) raise a clear
`ValueError` in both searches. No shortened plant tick or interpolated plant state
is scored. Control 0.15 s / forecast 0.3 s is supported and tested against execution.
`tests/planner/test_prediction_clock_precision.py` checks production float32
nested/flat clocks through both scorers and drivetrains for the shipped 0.1/0.1
and 0.2/0.1 s pairs, and retains rejection of genuinely nondivisible pairs.

`tests/planner/test_prediction_bicycle_rollout.py` compares candidate positions and
headings with executed commands and checks the braking-contact veto. This is
implementation agreement evidence; the matched T-intersection diagnostic in
issue #10109 and the hosted behaviour receipt remain required before behavioural
or benchmark conclusions. Capsule geometry remains separate work in #10093.

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
