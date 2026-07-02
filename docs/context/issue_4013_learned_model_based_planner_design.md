# Issue 4013 Learned Prediction MPC Design Slice

Plain-language summary: this slice adds a small state-based pedestrian predictor that can feed the
existing prediction-aware model predictive control (MPC) local planner, so future work can train the
predictor and compare closed-loop navigation without reopening the retired large world-model track.

## Claim Boundary

- Evidence status: diagnostic-only implementation scaffold.
- Implemented capability: `learned_prediction_mpc` and aliases construct a `PredictionMPCPlannerAdapter`
  with a learned short-horizon pedestrian predictor backend.
- Not claimed: no trained predictor, no full benchmark campaign, no paper-facing navigation claim, no
  DreamerV3/PlaNet/TD-MPC2-style latent world model.
- Fallback/degraded rows: `fallback_to_constant_velocity=true` is explicitly labeled
  `diagnostic_constant_velocity_fallback` and is not benchmark evidence.

## Runtime Contract

- Predictor input: current Robot SF social-navigation observation fields for robot state, goal, and
  pedestrian positions/velocities.
- Predictor output: pedestrian future positions in world coordinates as `PredictedPedestrianFutures`.
- Horizon: short-horizon only; the smoke config uses four steps at `0.2` seconds.
- Default artifact behavior: missing `checkpoint_path`/`model_id` fails closed.
- Diagnostic smoke behavior: `allow_untrained_smoke=true` builds a deterministic tiny MLP with zeroed
  weights and labels predictions `diagnostic_untrained_smoke`.
- World-model boundary: metadata reports `not_full_world_model=true`; the method is a state predictor
  plus MPC constraints, not a generative navigation world model.

## Remaining Work

- Train a real short-horizon predictor on a smoke dataset and publish checkpoint, normalizer,
  manifest, and metrics.
- Load the trained checkpoint without fallback and run a smoke scenario.
- Compare against `cv_prediction_mpc` and a model-free baseline with paired seeds.
- Exclude or separately count fallback/degraded rows before making benchmark or paper-facing claims.
