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
- [x] Model-based action selection runs on a smoke scenario. The `learned_prediction_mpc` adapter,
  built on the trained checkpoint via `configs/algos/learned_prediction_mpc_issue_4013_checkpoint.yaml`
  (fail-closed: `allow_untrained_smoke=false`, `fallback_to_constant_velocity=false`), emits a finite,
  bounded, goal-directed unicycle command from `plan()` — including with a pedestrian in the path —
  with the predictor reporting `evidence_tier=checkpoint_loaded` (no fallback). Proven by
  `tests/planner/test_learned_prediction_mpc_checkpoint.py`. This exercises the end-to-end
  model-based selection *path*; it is not the paired multi-arm benchmark comparison below.
- [x] Run the paired smoke *scenario* (benchmark runner) for all three arms and produce episode JSONL.
  Delivered by `scripts/benchmark/run_issue_4013_model_based_comparison.py`, which trains the
  checkpoint when missing, then runs three arms via `map_runner.run_map_batch` on
  `configs/scenarios/single/francis2023_blind_corner.yaml` (seed 4013, horizon 30): the checkpoint
  model-based arm (`configs/benchmarks/issue_4013_model_based_checkpoint_smoke.yaml`),
  `cv_prediction_mpc` (`issue_4013_model_free_smoke.yaml`), and a model-free `goal` baseline
  (`issue_4013_model_free_baseline_smoke.yaml`). Each arm wrote one non-fallback episode row.
- [x] Compare against `cv_prediction_mpc` and a model-free baseline with paired seeds using the
  existing `scripts/analysis/compare_model_based_planning_issue_4013.py` report contract. The report
  reached `status=diagnostic_ready`, `paired_seed_count=1`, no blockers, all five closure criteria
  met. Promoted to `docs/context/evidence/issue_4013_learned_model_based_planning/comparison_report.v1.{json,md}`.
- [x] Exclude or separately count fallback/degraded rows before making benchmark or paper-facing claims.
  All three arms reported `algorithm_metadata.status=ok` with zero excluded fallback/degraded rows;
  the report classifies fallback rows via `_fallback_reason` and keeps the claim boundary diagnostic-only.

Claim boundary for the trainer: the synthetic robot-repulsion task is a reproducible *learnability
probe*, not real ETH/UCY pedestrian data. A checkpoint is `smoke evidence` that the predictor trains
and loads without fallback; it is not benchmark, navigation-quality, or paper-facing evidence.

Claim boundary for the comparison run: single scenario / single seed diagnostic smoke. It proves the
model-based selection path runs end-to-end and is comparable (paired by scenario/seed) against a
constant-velocity predictor and a model-free baseline. In this short horizon no arm reached the goal
(success rate 0.0 for all three) — a truthful smoke observation, not a navigation-quality claim.
Scaling seeds/scenarios into nominal benchmark evidence remains a separate campaign (excluded here).
