# Issue #592 Hybrid Obstacle-Context Predictor Design

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/592>

## Current Boundary

`prediction_planner` currently predicts pedestrian trajectories from agent kinematics in the robot
frame:

- default model input: `(x_rel, y_rel, vx_rel, vy_rel)`;
- optional ego-conditioned input: the same four features plus robot velocity, goal direction, and
  goal distance;
- graph interaction: distance-weighted message passing over pedestrian slots;
- obstacle handling: downstream candidate scoring and occupancy penalties in
  `PredictionPlannerAdapter`, not obstacle-aware trajectory decoding.

That means the planner can penalize robot actions near static geometry, but the learned pedestrian
forecast itself does not directly condition on walls, corners, narrow passages, or doorway-like
free-space structure.

## Recommendation

Keep #592 as a research parent. The first implementation split should be a small feature-augmented
baseline, not a CNN or obstacle-node graph prototype.

The staged path is:

1. Add deterministic local obstacle features to the predictive dataset and model input.
2. Compare the feature baseline against the current registered predictor on the same hard-seed
   evaluation surface.
3. Escalate to a grid/CNN or obstacle-node graph model only if the feature baseline shows a
   measurable forecasting or downstream safety benefit.

This keeps the proof obligation proportional to the uncertainty. A full hybrid graph model changes
data collection, checkpoint schema, runtime inference cost, and benchmark provenance before there is
evidence that obstacle conditioning is the limiting factor.

## Architecture Options

### Option A: Local Obstacle Feature Baseline

Add per-agent local geometry features in the same robot-frame coordinate system as the current
state. Candidate features:

- nearest obstacle distance from the pedestrian position;
- obstacle normal direction `(nx, ny)` at the nearest obstacle boundary;
- local free-space or corridor direction `(tx, ty)` when it can be computed deterministically;
- optional signed clearance relative to a planner-facing safety radius.

Expected model shape:

- input dimension grows from `4` or `9` to roughly `7` to `14`, depending on whether ego features
  are combined with obstacle features;
- existing `PredictiveTrajectoryModel` can remain an MLP encoder plus message-passing decoder;
- checkpoint loading remains fail-closed through `PredictiveModelConfig.input_dim`;
- inference cost should stay close to the current predictor if features are precomputed cheaply
  from map/occupancy helpers.

Primary risk: obstacle features must be deterministic and cheap enough to compute in both data
collection and runtime inference. If they are derived from SVG geometry, the same feature extractor
must be used by collection, training, evaluation, and planner runtime.

### Option B: Local Occupancy Grid Encoder

Encode a local obstacle crop around each pedestrian or around the robot and fuse the grid embedding
with agent graph features before trajectory decoding.

Expected model shape:

- add a small CNN or MLP-over-grid encoder;
- fuse obstacle embedding with each agent embedding before message passing or before decoding;
- update dataset artifacts to include grid tensors or enough map context to regenerate them.

Primary risk: the dataset and checkpoint become much larger, and runtime inference now depends on
grid extraction plus a second encoder. This is only justified after Option A shows that obstacle
context helps and still misses cases where richer geometry matters.

### Option C: Obstacle-Node Graph

Represent obstacle samples, wall segments, or free-space boundary points as additional graph nodes.
Fuse pedestrian-pedestrian and pedestrian-obstacle messages in the predictor.

Expected model shape:

- heterogeneous nodes or typed message-passing blocks;
- explicit obstacle-node masks and feature schema;
- larger adapter burden between SVG maps, dataset collection, model training, and runtime inference.

Primary risk: this is the highest-complexity option and the easiest to overfit to map geometry. It
should remain research-only unless the simpler feature and grid variants fail for a well-documented
reason.

## Config-First Experiment Path

The implementation should introduce a config flag or model family name before any long run, for
example:

- `predictive_obstacle_features_v1` for Option A;
- `predictive_obstacle_grid_v1` for Option B;
- `predictive_obstacle_graph_v1` only if a later issue proves the need.

Suggested command path after implementation exists:

```bash
uv run python scripts/training/collect_predictive_planner_data.py \
  --episodes <N> \
  --max-steps <steps> \
  --max-agents 16 \
  --horizon-steps 8 \
  --output output/tmp/predictive_planner/datasets/<dataset>.npz
```

```bash
uv run python scripts/training/train_predictive_planner.py \
  --dataset output/tmp/predictive_planner/datasets/<dataset>.npz \
  --output-dir output/tmp/predictive_planner/training/<run_id> \
  --model-id <model_id> \
  --proxy-scenario-matrix configs/scenarios/sets/classic_cross_trap_subset.yaml \
  --proxy-every-epochs <k>
```

```bash
uv run python scripts/validation/evaluate_predictive_planner.py \
  --checkpoint output/tmp/predictive_planner/training/<run_id>/<checkpoint>.pt \
  --scenario-matrix configs/scenarios/sets/classic_cross_trap_subset.yaml \
  --output-dir output/tmp/predictive_planner/eval/<run_id> \
  --tag <tag>
```

Generated datasets, checkpoints, and evaluation rows belong under `output/` unless promoted through
the model registry or another durable artifact path.

## Required Proof Gates

Before any new obstacle-conditioned predictor can be benchmark-facing:

- dataset diagnostics must show non-degenerate targets and document the obstacle-feature schema;
- unit tests must cover feature extraction shape, masking, and model input-dimension compatibility;
- evaluation must compare against the current registered predictor on the same scenario matrix and
  seed manifest;
- the report must separate ADE/FDE changes from downstream planner metrics such as success,
  collision, near miss, minimum distance, and runtime;
- fallback, degraded, or missing-checkpoint behavior must remain non-success under the benchmark
  fallback policy.

Benchmark claims should use the existing mode language:

- `native` only when the obstacle-conditioned predictor is registered and loaded through the normal
  model path;
- `adapter` only for declared compatibility layers;
- `fallback`, `degraded`, or `not_available` for missing model, failed feature extraction, or any
  runtime path that falls back to constant-velocity prediction.

## Compute And Runtime Expectations

Option A should be the default next step because it is likely to add only small per-agent feature
cost and modest training overhead. It can probably reuse the current CPU evaluation path and a small
proxy matrix before any expensive campaign.

Option B should be treated as moderate cost because local grid extraction and CNN inference happen
for every planner decision. It needs a measured latency budget before benchmark promotion.

Option C should be treated as high cost and research-only until there is evidence that graphing
obstacle nodes solves a specific failure not addressed by Option A or B.

## Follow-Up Boundary

This note does not implement the obstacle-conditioned model. It scopes #592 into proof-first
follow-up work:

- first follow-up: implement deterministic obstacle-feature extraction and a feature-augmented
  predictive model config;
- second follow-up: run the small proxy training/evaluation comparison against the current
  predictor;
- optional later follow-up: prototype the grid or obstacle-node variant only if the first comparison
  justifies the added complexity.

