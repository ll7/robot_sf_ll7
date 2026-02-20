# Prediction Planner Complete Tutorial

This tutorial explains the `prediction_planner` end to end:

- core concepts from the literature-inspired design,
- what was actually implemented in `robot_sf`,
- how data, training, and benchmarking work,
- how to interpret current results and limitations.

It is based on:

- `output/tmp/predictive_planner/` implementation artifacts,
- Obsidian notes:
  - `Prediction-Planner from alyassiSocialRobotNavigation2025 2026-02-19`
  - `alyassiSocialRobotNavigation2025 prediction planner perplexity 2026-02-19`

## 1. Big Picture

The planner is a model-based local navigation adapter:

1. Predict short-horizon pedestrian trajectories in robot frame.
2. Sample candidate robot commands `(v, omega)`.
3. Roll out each candidate over the same horizon.
4. Score each candidate with goal-progress and safety/risk terms.
5. Execute the lowest-cost candidate.

In this repo, the key difference from full MP-RGL-style systems is:

- no full MCTS tree search,
- instead a deterministic sampled rollout search over a risk-adaptive action lattice.

That is an intentional engineering simplification to keep the benchmark loop tractable and reproducible.

## 2. Concept-to-Code Map

| Concept | Repository Implementation |
|---|---|
| Graph-based interaction prediction | `robot_sf/planner/predictive_model.py` (`PredictiveTrajectoryModel`) |
| Input state in robot frame | `PredictionPlannerAdapter._build_model_input` in `robot_sf/planner/socnav.py` |
| Masked fixed-slot agents | `state: (N,4)` + `mask: (N,)` in model and training scripts |
| Fallback when model unavailable | `PredictionPlannerAdapter._constant_velocity_prediction` |
| Candidate rollout scoring | `PredictionPlannerAdapter._score_action` |
| Risk-adaptive horizon | `_effective_rollout_steps` |
| Risk-adaptive lattice and speed cap | `_candidate_set`, `_risk_speed_cap_ratio` |
| Canonical benchmark config | `configs/algos/prediction_planner_camera_ready.yaml` |
| Benchmark wiring | `robot_sf/benchmark/map_runner.py`, readiness/metadata modules |

## 3. Predictor Model Internals

The predictor input per active pedestrian is:

`(x_rel, y_rel, vx_rel, vy_rel)` in robot frame.

### 3.1 Encoder + message passing

Each agent slot is encoded, then refined by message passing with distance-based attention.

Attention weights are computed from pairwise distances:

Code-style form:

`a_ij = softmax_j(-||p_i - p_j||^2 / tau)`

LaTeX form:

$$
a_{ij} = \operatorname{softmax}_{j}\left(-\frac{\lVert p_i - p_j \rVert^2}{\tau}\right)
$$

with masking to ignore inactive slots.

Update rule per block is residual MLP:

Code-style form:

`h_i <- h_i + MLP([h_i, sum_j a_ij * h_j])`

LaTeX form:

$$
h_i \leftarrow h_i + \operatorname{MLP}\left([h_i,\ \sum_j a_{ij} h_j]\right)
$$

See:

- `PredictiveTrajectoryModel._attention_weights`
- `_MessageBlock.forward`

in `robot_sf/planner/predictive_model.py`.

### 3.2 Trajectory decoding

Decoder predicts per-step deltas and integrates with cumulative sum:

Code-style form:

`p_i,t = p_i,0 + cumsum_t(delta_i,t)`

LaTeX form:

$$
p_{i,t} = p_{i,0} + \sum_{\tau=1}^{t}\Delta p_{i,\tau}
$$

Output shape:

- `future_positions: (B, N, T, 2)`

where `T = horizon_steps`.

## 4. Training Objective and Metrics

Training uses masked SmoothL1 over predicted trajectories:

- function: `masked_trajectory_loss` in `robot_sf/planner/predictive_model.py`.

Evaluation metrics:

- ADE: average displacement error across valid slots and timesteps,
- FDE: final-step displacement error across valid slots.

Code-style form:

`ADE = (1 / |V|) * sum_(i,t in V) ||p_hat_i,t - p_i,t||_2`

`FDE = (1 / |V_T|) * sum_(i in V_T) ||p_hat_i,T - p_i,T||_2`

LaTeX form:

$$
\operatorname{ADE}
= \frac{1}{|V|}
\sum_{(i,t)\in V}
\left\lVert \hat{p}_{i,t} - p_{i,t} \right\rVert_2
$$

$$
\operatorname{FDE}
= \frac{1}{|V_T|}
\sum_{i\in V_T}
\left\lVert \hat{p}_{i,T} - p_{i,T} \right\rVert_2
$$

function:

- `compute_ade_fde` in `robot_sf/planner/predictive_model.py`.

## 5. Planner Adapter Mechanics

The adapter is `PredictionPlannerAdapter` in `robot_sf/planner/socnav.py`.

### 5.1 Data path

1. Build model input from observation:
   - `state`, `mask`, robot pose/heading.
2. Predict future trajectories:
   - learned model if available,
   - constant-velocity fallback if `allow_fallback=True` and model load fails.

### 5.2 Candidate generation

Baseline candidates come from configured speed ratios and heading deltas.

Near predicted crowd interaction:

- increase horizon (`predictive_horizon_boost_steps`),
- cap speed (`predictive_near_field_speed_cap`),
- add denser near-field speed and heading samples.

This creates stronger avoidance authority in difficult crossing moments without globally slowing every scenario.

### 5.3 Scoring function

Each candidate gets a scalar cost:

Code-style form:

`J = -w_goal*progress + w_col*collision + w_near*near + w_ttc*ttc + w_occ*occ + w_v*|v| + w_w*|omega| + w_pr*progress_risk + w_hc*hard_clearance`

LaTeX form:

$$
\begin{aligned}
J =\;& -w_{\text{goal}}\cdot \text{progress}
+ w_{\text{col}}\cdot \text{collision}
+ w_{\text{near}}\cdot \text{near} \\
&+ w_{\text{ttc}}\cdot \text{ttc}
+ w_{\text{occ}}\cdot \text{occ}
+ w_v\cdot |v|
+ w_{\omega}\cdot |\omega| \\
&+ w_{\text{pr}}\cdot \text{progress\_risk}
+ w_{\text{hc}}\cdot \text{hard\_clearance}
\end{aligned}
$$

where:

- `progress_risk` penalizes aggressive progress under low predicted clearance,
- `hard_clearance` penalizes violating a strict clearance band.

This is the core implementation of stronger progress-risk control requested for hard cases.

## 6. Data Pipeline

### 6.1 Collection

Script:

- `scripts/training/collect_predictive_planner_data.py`

Generates rollouts and extracts supervised samples:

- `state`: `(samples, max_agents, 4)`
- `target`: `(samples, max_agents, horizon, 2)`
- `mask`: `(samples, max_agents)`

Output example:

- `output/tmp/predictive_planner/datasets/predictive_rollouts_full_v1.npz`

### 6.2 Hard-case enrichment

Scripts:

- `scripts/training/collect_predictive_hardcase_data.py`
- `scripts/training/build_predictive_mixed_dataset.py`

This oversamples difficult seeds where planner failures persist.

### 6.3 Training

Script:

- `scripts/training/train_predictive_planner.py`

Notable features:

- train/validation split,
- ADE/FDE logging per epoch,
- optional proxy hard-set evaluation every `k` epochs,
- optional checkpoint selection by proxy hard-set metrics.

## 7. Benchmarking and Diagnostics

### 7.1 Single eval

- `scripts/validation/evaluate_predictive_planner.py`

Provides:

- success rate,
- mean minimum distance,
- mean average speed,
- per-scenario deltas and failure taxonomy.

### 7.2 Per-seed diagnostics

- `scripts/validation/run_predictive_hard_seed_diagnostics.py`

Outputs trace JSON per hard seed plus a summary report:

- `output/tmp/predictive_planner/diagnostics/hard_seed_diagnostics/`

### 7.3 Campaign sweep

- `scripts/validation/run_predictive_success_campaign.py`

Ranks planner config variants on:

1. hard-set success,
2. hard-set clearance,
3. global success.

Example campaign:

- `output/tmp/predictive_planner/campaigns/risk_aware_adaptive_check_20260220/`

## 8. How Obsidian Notes Mapped to Final Design

The two notes emphasized:

- graph-based interaction prediction,
- short-horizon strategic planning,
- static obstacle awareness,
- complexity tradeoff versus full MCTS systems.

In this repo, those ideas became:

1. Graph message passing predictor:
   - implemented directly.
2. Strategic lookahead:
   - implemented as sampled rollout scoring instead of MCTS.
3. Static obstacle handling:
   - implemented via occupancy-grid path penalty in scoring (not a separate LiDAR branch model).
4. Reimplementation tractability:
   - prioritized deterministic benchmark integration, reproducible scripts, and explicit quality gates.

This is a deliberate benchmark-oriented adaptation, not a claim of exact upstream MP-RGL reproduction.

## 9. Reproducible Commands (Minimal)

Train:

```bash
uv run python scripts/training/train_predictive_planner.py \
  --dataset output/tmp/predictive_planner/datasets/predictive_rollouts_mixed_v1.npz \
  --output-dir output/tmp/predictive_planner/training/predictive_proxy_selected_v1 \
  --model-id predictive_proxy_selected_v1 \
  --select-by-proxy \
  --proxy-scenario-matrix configs/scenarios/classic_interactions.yaml \
  --proxy-seed-manifest configs/benchmarks/predictive_hard_seeds_v1.yaml
```

Evaluate:

```bash
uv run python scripts/validation/evaluate_predictive_planner.py \
  --checkpoint output/tmp/predictive_planner/training/predictive_proxy_selected_v1/predictive_model.pt \
  --scenario-matrix configs/scenarios/classic_interactions.yaml
```

Campaign:

```bash
uv run python scripts/validation/run_predictive_success_campaign.py \
  --checkpoints output/tmp/predictive_planner/training/predictive_proxy_selected_v1/predictive_model.pt \
  --scenario-matrix configs/scenarios/classic_interactions.yaml \
  --hard-seed-manifest configs/benchmarks/predictive_hard_seeds_v1.yaml \
  --planner-grid configs/benchmarks/predictive_sweep_planner_grid_v1.yaml
```

## 10. Current Status and Realistic Interpretation

- The planner is integrated and reproducible in benchmark workflows.
- Safety/clearance has improved under hard-case tuning.
- Hard-seed success still plateaus on the curated difficult subset.

Interpretation:

- this is no longer an infrastructure gap,
- it is now primarily an algorithm/model gap for the hardest interaction tails.

## 11. References

- Chen et al. (2020) local copy:
  - `output/tmp/predictive_planner/Chen et al. - 2020 - Robot Navigation in Crowds by Graph Convolutional Networks With Attention Learned From Human Gaze.pdf`
- Alyassi et al. (2025) local paper copy used in prior analysis context:
  - `amv_benchmark_paper/context/new_paper2025/z-export/Exported Items/Alyassi et al. - 2025 - Social robot navigation a review and benchmarking of learning-based methods.pdf`
- Internal design and run artifacts:
  - `output/tmp/predictive_planner/README.md`
  - `output/tmp/predictive_planner/reports/2026-02-20_algorithmic_hardcase_roadmap.md`
  - `docs/baselines/prediction_planner.md`
  - `docs/training/predictive_planner_training.md`
