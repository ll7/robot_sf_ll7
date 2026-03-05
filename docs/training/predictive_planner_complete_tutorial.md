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

## How to Read This Guide

### Beginner path (recommended first pass)

Read in this order:

1. Section 1 (big picture)
2. Section 5 (what planner does each control step)
3. Section 6 (what data is collected and why)
4. Section 8 (deterministic prediction behavior)
5. Section 12 (FAQ + consequences)

### Expert path (fast skim)

Jump directly to:

- Section 2 (concept-to-code map),
- Sections 3-4 (model and losses),
- Sections 9-11 (benchmark integration + references),
- Section 12 (implementation caveats).

### Legend

- `[Beginner]` plain-language explanation.
- `[Expert Skip]` implementation details for advanced readers.

## Quick Glossary (Beginner)

- **Adapter**: wrapper that turns observations into robot control commands.
- **Candidate control**: one possible command pair `(v, omega)` to test.
- **Lattice**: finite discrete set of candidate controls.
- **Rollout**: simulated short-horizon trajectory under one candidate command.
- **Near-field risk**: predicted close pedestrian proximity in the near horizon.
- **Mask**: tensor marking which entries are valid and should affect loss/metrics.
- **Target mask**: validity map for future points per pedestrian and timestep.

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

[Beginner] Think of this as:

- first predict where people may move,
- then test many steering options quickly,
- choose the safest useful option right now,
- repeat at the next timestep.

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

[Expert Skip] If you only need implementation entry points, this table is enough.

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

Plain-language interpretation:

- `a_ij` is a normalized closeness weight from pedestrian `i` to neighbor `j`.
- The negative sign means closer neighbors receive larger logits (thus larger weights).
- Squared distance is used for stable/cheap computation and preserves ordering
  of closeness without needing a square root.
- `tau` is a temperature:
  - smaller `tau` => sharper attention (focus on nearest neighbors),
  - larger `tau` => flatter attention (spread influence over more neighbors).

[Beginner] Intuition:

- neighbors closer to pedestrian `i` matter more,
- far neighbors matter less,
- but all valid neighbors still contribute a little unless attention is very sharp.

Update rule per block is residual MLP:

Code-style form:

`h_i <- h_i + MLP([h_i, sum_j a_ij * h_j])`

LaTeX form:

$$
h_i \leftarrow h_i + \operatorname{MLP}\left([h_i,\ \sum_j a_{ij} h_j]\right)
$$

[Beginner] Residual update means:

- model computes a correction (`delta`) and adds it to current feature,
- it does not overwrite everything each step,
- this usually trains more stably.

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

[Beginner] What actually happens in code for one pedestrian:

1. Hidden feature `h_i` goes through the decoder MLP.
2. Decoder outputs `T*2` values and reshapes to `delta_i[t] = (dx_t, dy_t)`.
3. `cumsum` accumulates these deltas over time:
   - step 1 offset = `delta_1`
   - step 2 offset = `delta_1 + delta_2`
   - step 3 offset = `delta_1 + delta_2 + delta_3`
4. Current position `p_i,0` is added to each accumulated offset to get absolute
   future positions in robot frame.
5. If the agent slot is inactive (`mask=0`), outputs are zeroed and ignored later.

Small numeric example:

- current position `(1.0, 2.0)`
- predicted deltas for 3 steps:
  - `(+0.2, +0.0)`, `(+0.1, -0.1)`, `(+0.0, -0.2)`
- cumulative offsets:
  - `(0.2, 0.0)`, `(0.3, -0.1)`, `(0.3, -0.3)`
- decoded trajectory:
  - `(1.2, 2.0)`, `(1.3, 1.9)`, `(1.3, 1.7)`

Consequence:
- The model predicts motion increments, not absolute coordinates directly.
- This tends to make short-horizon motion smoother and easier to learn.

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

### 4.1 ADE vs FDE intuition

[Beginner]

- **ADE** answers: "On average over the whole horizon, how far are predicted
  points from ground truth?"
- **FDE** answers: "At the final horizon step, how far is the endpoint error?"

Why both matter:

- Low ADE + high FDE:
  - trajectory is mostly fine early, but endpoint drifts late.
- High ADE + low FDE:
  - model is noisy mid-horizon but eventually lands near final target.
- Low ADE + low FDE:
  - generally strong trajectory quality.

Units:

- Both are in meters (same frame as prediction target).
- Lower is better.

### 4.2 What is "masked" in ADE/FDE and loss?

[Beginner]

Masked means invalid/padded entries do not contribute to optimization or metrics:

- `mask (N)` says which pedestrian slots are active now.
- `target_mask (N, T)` says which future points are valid after matching.

In this project, loss and ADE/FDE use `target_mask` when available.
This is important because pedestrian identities can appear/disappear over horizon;
without masking, zeros from unmatched points can fake very low error.

### 4.3 Practical interpretation checklist

Use this quick read during training:

1. `train_loss` down, `val_loss` down, ADE/FDE down:
   - learning signal is likely healthy.
2. `train_loss` down but val ADE/FDE flat or worse:
   - overfitting or dataset mismatch.
3. ADE improves but FDE remains high:
   - increase horizon-specific emphasis or inspect long-horizon labels.
4. all metrics near zero from early epochs:
   - treat as suspicious and inspect dataset masks/targets immediately.

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

[Beginner] Why this matters:

- in easy space: planner can keep normal speed and fewer turns,
- in crowded close encounters: planner automatically gets more cautious choices.

### 5.3 Scoring function

Each candidate gets a scalar cost:

Code-style form:

`J = -w_goal*progress + w_col*collision + w_near*near + w_ttc*ttc + w_occ*occ + w_v*|v| + w_w*|omega| + w_pr*progress_risk + w_hc*hard_clearance`

LaTeX form:

$$
J =
-w_{\text{goal}}\cdot \text{progress}
+ w_{\text{col}}\cdot \text{collision}
+ w_{\text{near}}\cdot \text{near}
+ w_{\text{ttc}}\cdot \text{ttc}
+ w_{\text{occ}}\cdot \text{occ}
+ w_v\cdot |v|
+ w_{\omega}\cdot |\omega|
+ w_{\text{pr}}\cdot \text{progress\_risk}
+ w_{\text{hc}}\cdot \text{hard\_clearance}
$$

where:

- `progress_risk` penalizes aggressive progress under low predicted clearance,
- `hard_clearance` penalizes violating a strict clearance band.

This is the core implementation of stronger progress-risk control requested for hard cases.

[Beginner] Interpretation of score:

- lower score is better,
- rewards moving toward goal,
- penalizes collision-like situations, tight near-misses, and risky aggressive motion.

## 6. Data Pipeline

### 6.1 Collection

Script:

- `scripts/training/collect_predictive_planner_data.py`

Generates rollouts and extracts supervised samples:

- `state`: `(samples, max_agents, 4)`
- `target`: `(samples, max_agents, horizon, 2)`
- `mask`: `(samples, max_agents)`
- `target_mask`: `(samples, max_agents, horizon)` (valid future points only)

Important semantics:

- The robot is **not stationary** during collection.
- Rollouts use a simple goal-seeking `(v, omega)` controller so robot motion changes
  the relative frame over time.
- Supervision is extracted in the robot frame at time `t`; future pedestrian
  positions are transformed back to that same frame.
- Pedestrian identities across future steps are assigned with nearest-neighbor matching.
- `target_mask` marks only matched/valid future positions; loss/metrics use this mask
  to avoid zero-target collapse.

[Beginner] Practical consequence:

- if `target_mask` is mostly zero, model can appear to “train” while learning nothing,
- always inspect dataset summary ratios before long runs.

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

## 8. Prediction Type: Deterministic vs Stochastic

Current implementation is **deterministic**:

- The model outputs one trajectory tensor `future_positions` for each pedestrian slot.
- No distribution head (for example no Gaussian mean/variance) is used.
- Runtime planner therefore optimizes controls against a single predicted future.

This means uncertainty is handled implicitly by conservative costs and candidate search,
not by stochastic sampling over forecast distributions.

## 9. How It Integrates Into the Benchmark

Benchmark algorithm key:

- `prediction_planner` (canonical name)

Core integration points:

- Planner adapter: `robot_sf/planner/socnav.py` (`PredictionPlannerAdapter`)
- Benchmark config loader: `robot_sf/benchmark/predictive_planner_config.py`
- Episode runner: `robot_sf/benchmark/map_runner.py`

Runtime flow in benchmark episodes:

1. Load predictive checkpoint + algorithm params.
2. Build model input from current observation.
3. Predict pedestrian futures.
4. Sample candidate robot commands.
5. Roll out and score candidates.
6. Apply best `(v, omega)` action.

## 10. Building Blocks of the Final Steering Planner

The final predictive planner used in runs is composed of:

1. **Learned pedestrian forecaster** (`PredictiveTrajectoryModel`).
2. **Observation-to-model adapter** (state + mask construction in planner adapter).
3. **Candidate command lattice** (speed/heading sampling with risk-aware adjustments).
4. **Short-horizon robot rollout model** (unicycle local rollout).
5. **Multi-term scoring function** (progress + safety + clearance + smoothness terms).
6. **Execution loop** integrated through benchmark `map_runner`.

## 11. Pipeline Exit Codes

The config-first pipeline (`run_predictive_training_pipeline.py`) has two terminal outcomes:

- exit `0`: all post-training stage gates pass (`evaluation`, `hard_seed_diagnostics`, `campaign`)
- exit `2`: one or more stage gates fail

Even on exit `2`, the pipeline writes final artifacts:

- `final_performance_summary.json`
- `final_performance_summary.md`

Use those files as the authoritative failure diagnosis.

- `output/tmp/predictive_planner/diagnostics/hard_seed_diagnostics/`

### 7.3 Campaign sweep

- `scripts/validation/run_predictive_success_campaign.py`

Ranks planner config variants on:

1. hard-set success,
2. hard-set clearance,
3. global success.

Example campaign:

- `output/tmp/predictive_planner/campaigns/risk_aware_adaptive_check_20260220/`

[Expert Skip] Campaign artifacts are the primary source for mode-to-mode comparison.

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

[Beginner] This means the plumbing works; remaining work is mostly model quality.

## 11. References

### 11.1 Academic papers (online)

- Chen et al. (2020), IEEE RA-L:
  - "Robot Navigation in Crowds by Graph Convolutional Networks With Attention Learned From Human Gaze"
  - DOI: `10.1109/LRA.2020.2972868`
  - arXiv: https://arxiv.org/abs/1909.10400

- Alyassi et al. (2025), Frontiers in Robotics and AI:
  - "Social robot navigation: a review and benchmarking of learning-based methods"
  - https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1658643/full

- Francis et al. (2023), social navigation evaluation guidelines:
  - "Principles and Guidelines for Evaluating Social Navigation Algorithms"
  - https://storage.googleapis.com/pirk.io/papers/Francis.etal-2023-SocialNavGuidelines.pdf

### 11.2 Code/public benchmark references (online)

- SocNavBench repository + paper entry point:
  - https://github.com/CMU-TBD/SocNavBench
  - arXiv link from repo: https://arxiv.org/abs/2103.00047

### 11.3 Local copies and internal provenance

- Chen et al. local copy:
  - `output/tmp/predictive_planner/Chen et al. - 2020 - Robot Navigation in Crowds by Graph Convolutional Networks With Attention Learned From Human Gaze.pdf`

- Alyassi et al. local copy used in project context:
  - `amv_benchmark_paper/context/new_paper2025/z-export/Exported Items/Alyassi et al. - 2025 - Social robot navigation a review and benchmarking of learning-based methods.pdf`

- Internal artifacts:
  - `output/tmp/predictive_planner/README.md`
  - `output/tmp/predictive_planner/reports/2026-02-20_algorithmic_hardcase_roadmap.md`
  - `docs/baselines/prediction_planner.md`
  - `docs/training/predictive_planner_training.md`
  - Obsidian notes:
    - `Prediction-Planner from alyassiSocialRobotNavigation2025 2026-02-19`
    - `alyassiSocialRobotNavigation2025 prediction planner perplexity 2026-02-19`

### 11.4 Traceability note

This implementation is an engineering adaptation inspired by the references above.
It is not a claim of exact reproduction of any single upstream planner.

## 12. FAQ and Consequences (Maintainer Notes)

### Q1) Is the robot stationary during predictive dataset collection?

No. The robot moves using a simple goal-seeking controller `(v, omega)` while
pedestrian observations are recorded and transformed into robot frame.

Consequence:
- The predictor is trained on ego-relative dynamics under changing robot pose,
  which is required for planner-time consistency.

### Q2) What exactly is in the predictive dataset?

Per sample:
- `state (N, 4)`: pedestrian `(x_rel, y_rel, vx_rel, vy_rel)` at time `t`,
- `target (N, T, 2)`: matched future positions in frame-at-`t`,
- `mask (N)`: active pedestrian slots,
- `target_mask (N, T)`: valid matched future points.

Consequence:
- Loss and ADE/FDE use `target_mask`; invalid future points do not silently bias
  training toward trivial zero predictions.

### Q3) Is the model stochastic? Do we predict a distribution?

No. Current predictor is deterministic: one `future_positions` tensor per step.
There is no distribution head (no variance, no multi-modal sampling).

Consequence:
- Runtime risk handling relies on conservative cost terms and candidate search,
  not probabilistic expectation/CVaR over prediction uncertainty.

### Q4) How are control candidates generated?

`PredictionPlannerAdapter._candidate_set` builds a deterministic lattice:
- base speed ratios (`predictive_candidate_speeds`),
- base heading deltas (`predictive_candidate_heading_deltas`),
- near-field augmentation (extra speeds/headings),
- risk speed cap from predicted minimum distance,
- conversion to `(v, omega)` with clipping by velocity limits and `dt`,
- dedupe + sort for stable deterministic ordering.

Consequence:
- Reproducible outputs and bounded compute, but limited action-space coverage
  versus richer planners unless the lattice is tuned.

### Q5) Full MCTS tree search vs deterministic sampled rollout: what differs?

MCTS:
- branching search across action sequences,
- explicit explore/exploit policy,
- stronger long-horizon contingency handling,
- substantially higher compute/memory cost.

Deterministic sampled rollout (current):
- fixed finite candidate set per control cycle,
- shallow horizon re-evaluated each step,
- no stochastic tree expansion.

Consequence:
- Better reproducibility and speed for benchmarks, but more myopic behavior in
  delayed-conflict situations.

### Q6) Do we consider probability of predicted actions during rollouts?

No. Candidate rollout scoring uses deterministic predictions and deterministic
cost aggregation only.

Consequence:
- Risk under epistemic/aleatoric uncertainty is under-modeled; robustness depends
  on hand-tuned safety margins and cost weights.

### Q7) Why does training “stop”?

Training stage stops when configured `epochs` is reached. After that, pipeline
runs post-training stages (eval, diagnostics, campaign).

Pipeline exit:
- `0`: all stage gates pass,
- `2`: one or more stage gates fail.

Consequence:
- A non-zero pipeline exit is often a quality-gate failure, not a crash.
  Always inspect `final_performance_summary.json`.

### Q8) Why did we see flat zero train/val metrics?

That pattern indicates degenerate supervision (historically from invalid-future
masking behavior). The pipeline now uses per-horizon `target_mask` end-to-end
to prevent zero-loss collapse.

Consequence:
- If metrics are still flat after the fix, treat as a hard data-quality incident
  and audit dataset ratios (`active_agent_ratio`, `active_target_ratio`) before
  running long jobs.

### Q9) Why did a run warn “completed with failing stage gates”?

Because at least one post-training stage returned non-zero (typically eval
quality thresholds). The pipeline intentionally still writes final summaries.

Consequence:
- Runs remain diagnosable and reproducible even on failure; promotion decisions
  must require stage-gate pass.

### Q10) Is TensorFlow AVX2/FMA message an error?

No. It is an informational CPU optimization note from TensorFlow.

Consequence:
- Safe to ignore for correctness; only relevant for CPU performance tuning.

### Q11) Recommended “full training” profile for BR-07

Use:
- `configs/training/predictive/predictive_br07_all_maps_randomized_full.yaml`

Current recommended band:
- `180-240` epochs (default `220` in full profile),
- proxy every `10` epochs for throughput balance,
- all-maps base collection with randomized seeds.

Consequence:
- Provides strong capacity without unbounded runtime; further gains should come
  from structured sweeps, not only longer epochs.

### Q12) Why not train much longer by default?

Longer epochs increase wall time and proxy-eval cost, and can overfit while
providing little hard-case gain after plateau.

Consequence:
- Prefer evidence-driven extension (curves + hard-case metrics) over blindly
  increasing epochs.
