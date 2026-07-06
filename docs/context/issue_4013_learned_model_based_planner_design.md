# Issue #4013 — Learned Prediction MPC Design Slice

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

- [x] Train a real short-horizon predictor on a smoke dataset and publish checkpoint, manifest, and
  metrics. Delivered by `robot_sf/planner/learned_short_horizon_trainer.py` plus
  `scripts/training/train_learned_short_horizon_predictor_issue_4013.py`
  (`configs/training/learned_short_horizon_predictor_issue_4013_smoke.yaml`). The trainer fits the
  predictor's own architecture on a seeded synthetic robot-repulsion residual task and writes a
  checkpoint that loads as `evidence_tier=checkpoint_loaded` (not `diagnostic_untrained_smoke`).
  A normalizer is intentionally not emitted: the inference predictor consumes raw features, so the
  manifest records `feature_stats` as provenance metadata only, not an applied normalizer.
- [x] Load the trained checkpoint without fallback. `LearnedShortHorizonPedestrianPredictor` loads the
  trained `.pt` and `predict()` returns `source="checkpoint_loaded"` on a smoke observation.
- [ ] Run the paired smoke *scenario* (benchmark runner) for all three arms and produce episode JSONL.
- [ ] Compare against `cv_prediction_mpc` and a model-free baseline with paired seeds using the
  existing `scripts/analysis/compare_model_based_planning_issue_4013.py` report contract.
- [ ] Exclude or separately count fallback/degraded rows before making benchmark or paper-facing claims.

Claim boundary for the trainer: the synthetic robot-repulsion task is a reproducible *learnability
probe*, not real ETH/UCY pedestrian data. A checkpoint is `smoke evidence` that the predictor trains
and loads without fallback; it is not benchmark, navigation-quality, or paper-facing evidence.
