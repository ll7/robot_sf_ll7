# Issue 403: Retrain with Grid Observation

## Summary
We want to retrain a robot policy using a grid-based observation space (SocNav structured
observation + occupancy grid) and make the experiment publication-ready with reproducible
metrics and uncertainty bands. This document is the living plan and shared context for
decisions, ideas, and open questions.

## Goals
- Define a **human-readable observation-space contract** for the benchmark (fields, shapes,
  frame, grid config, ordering).
- Train a new policy that uses **grid observations** and can be compared to existing baselines.
- Track **publication-grade metrics** (success, collisions, comfort/force, SNQI, etc.) with
  **uncertainty bands** via multi-seed evaluation.
- Produce reproducible artifacts (manifests + JSONL + plots) under `output/`.

## Non-Goals
- No implementation yet; this phase is discovery, requirements, and design.
- Not committing to a specific algorithm beyond PPO until requirements are clear.
- No changes to benchmark schema or CLI unless explicitly required later.

## Scope
- Observation-space definition and training/evaluation protocol.
- Decide on grid configuration, frame, and action limits.
- Decide on training pipeline (legacy `scripts/training_ppo.py` vs structured imitation pipeline).
- Define metrics, evaluation cadence, and artifact outputs for publication.

## Assumptions
- Benchmark plan (2026-01-14) **selects SocNav structured + occupancy grid**.
- Benchmark plan **specifies**: ego frame grid, 0.5 m resolution, 32×32 m extent, unicycle actions,
  and pedestrians ordered by distance (closest-first).
- Code defaults **do not** enforce this automatically; we must set config explicitly.
- `peds_have_obstacle_forces` (ped-robot repulsion) is **not explicitly specified** in benchmark docs
  and defaults to `False` unless set.
- PPO baseline adapter currently supports vector/image obs; **grid observation support is missing**.

## Inputs
- Benchmark plan decisions: `docs/dev/benchmark_plan_2026-01-14.md` (observation/action contract).
- Occupancy grid guide: `docs/dev/occupancy/Update_or_extend_occupancy.md`.
- SocNav structured observation doc: `docs/dev/issues/socnav_structured_observation.md`.
- Training pipelines:
  - Legacy: `scripts/training_ppo.py` (fast but less reproducible).
  - Structured: `scripts/training/train_expert_ppo.py` + configs under `configs/training/ppo_imitation/`.
- Example grid usage: `examples/quickstart/04_occupancy_grid.py`, `examples/occupancy_reward_shaping.py`.

## Output Artifacts
- Training run manifests (per seed) and evaluation manifests.
- JSONL episode logs for evaluation.
- Aggregated tables (CSV/Markdown) and plots with uncertainty bands.
- Model checkpoints under `model/` (or `output/` if policy artifacts should stay outside repo).

## Implementation Plan (current)
### Phase 0: Clarify and Lock the Observation Contract
- Define **exact observation fields** (SocNav struct + occupancy grid + metadata?).
- Decide **grid config** (resolution, width/height, channels, ego vs world frame).
- Confirm **action space** (unicycle only) and limits (v_max, omega_max).
- Decide **pedestrian handling**: max count, ordering, velocity frame (ego vs world).
- Confirm **ped-robot repulsion** toggle (`peds_have_obstacle_forces`) for benchmark runs.

### Phase 1: Training Protocol (publication-grade)
- Pick training pipeline: prefer structured imitation pipeline for reproducibility.
- Decide number of **seeds** for uncertainty bands (recommend ≥5).
- Define **evaluation cadence** (e.g., every N episodes or checkpoints).
- Define **hold-out evaluation** set if generalization is a goal.

### Feature Extractor Design (proposal)
**Goal:** Multi-input policy that processes the occupancy grid like an image while
also digesting SocNav scalar/structured fields, including an ego-frame waypoint vector.

Baseline design (simple + robust):
- **Grid branch (CNN)**: small convolutional stack over `occupancy_grid` (C×64×64),
  followed by flatten or global pooling.
- **SocNav branch (MLP)**: flatten all non-grid fields (robot state, goal current/next,
  ped positions/velocities, map size, timestep).
- **Ego-frame goal vector**: compute `goal_next - robot_position`, rotate into ego frame
  using robot heading (derive inside extractor; avoid modifying env for now).
- **Concatenate** grid features + socnav features → policy MLP head.

Optional upgrades (if baseline underperforms):
- **Pedestrian encoder**: small MLP per pedestrian + pooling (mean/max) instead of full
  flattening to reduce sensitivity to padding.
- **Attention block** for pedestrians (K=10) to improve interaction modeling.
- **History stacking** (if needed): wrap env to stack a few recent SocNav frames.

Metadata handling:
- **Exclude grid metadata fields** from the policy input (grid config is fixed).
  Keep metadata in the observation dict for debugging/compat only.

### Phase 2: Evaluation + Uncertainty
- Use benchmark metrics (success, collisions, near misses, comfort/force, SNQI).
- Compute **confidence intervals** across seeds (bootstrap or t-interval).
- Produce plots with mean + CI bands.

### Phase 3: Integration (later)
- If policy must run in benchmark, add/extend PPO adapter to accept grid observation.

## Validation
- Reproducibility: same seed + config → same metrics.
- Evaluation: fixed scenario set + fixed seeds + deterministic inference.
- Uncertainty: multi-seed runs produce consistent mean/CI bands.

## Risks
- Observation mismatch (benchmark plan vs actual config).
- PPO adapter incompatibility with grid observations.
- Training instability due to high-dimensional grid input.
- Results not comparable if action limits or evaluation scenarios drift.

## Notes
### Decisions (current)
- **Observation/action interface target**: SocNav structured + occupancy grid, unicycle actions.
- **Grid frame**: ego frame (per benchmark plan).
- **Grid resolution/extent**: 0.5 m, 32×32 m (per benchmark plan) → 64×64 cells.
- **Next waypoint signal**: required in observation (goal "next" / next_target_angle).
- **Next waypoint representation**: vector in the robot’s local (ego) frame.

### Open Questions (must answer)
- Are we committing to the benchmark grid config (0.5 m / 32×32 / ego frame) for training?
- Do we include grid metadata fields (`occupancy_grid_meta_*`) in the policy input?
- Should pedestrians exert repulsive forces on the robot (`peds_have_obstacle_forces`)?
- How many training seeds are feasible for uncertainty bands?
- Do we evaluate on the same scenarios as training or hold out a test set?
- Do we need checkpoint-level evaluation curves, or just final policy comparisons?
- Which feature-extractor baseline do we commit to for the first full run?
- Do we want a pedestrian-encoder (MLP/attention) or simple flattening first?

### Hyperparameter Strategy (proposal)
We should stay flexible but **structured**:
- Start with a **single baseline config** (reasonable defaults) and verify it learns.
- Run a **small sweep** over 2–4 high-impact knobs (e.g., learning rate, n_steps,
  entropy coeff, CNN width) using a subset of scenarios.
- Once a stable config is chosen, run **multi-seed training** for uncertainty bands.

Suggested sweep candidates:
- PPO: learning rate, n_steps, batch size, clip_range, entropy_coeff, gamma, gae_lambda.
- CNN: channels per layer, kernel sizes, pooling strategy.
- MLP: hidden dims for socnav branch and policy head.

### Baseline Hyperparameter Set (proposal)
*These are starting points to validate learning; not final.*
- **Algorithm**: PPO (`MultiInputPolicy`)
- **Total timesteps**: 5–10M per seed for full run (start with 0.5–1M for smoke)
- **n_envs**: 8–16 (dependent on CPU/GPU and sim speed)
- **n_steps**: 2048 or 4096
- **batch_size**: 64 or 128
- **learning_rate**: 3e-4 (try 1e-4 if unstable)
- **gamma**: 0.99
- **gae_lambda**: 0.95
- **clip_range**: 0.2
- **ent_coef**: 0.0–0.01
- **vf_coef**: 0.5
- **max_grad_norm**: 0.5
- **policy_net**: [256, 256] after concatenated features

Feature extractor baseline:
- **Grid CNN**: 3–4 conv layers, channels [32, 64, 64], kernel sizes [5, 3, 3],
  stride 2 or max-pool, global average pool → 128–256 dims.
- **SocNav MLP**: 2 layers [128, 128].
- **Concat** → policy MLP.

### Sweep Proposal (small, structured)
Goal: 10–20 runs on a reduced scenario set, then choose 1 config for full training.
1) **Learning rate × n_steps** (core stability)
   - LR ∈ {3e-4, 1e-4}
   - n_steps ∈ {2048, 4096}
2) **Entropy coefficient** (exploration)
   - ent_coef ∈ {0.0, 0.005, 0.01}
3) **CNN width** (capacity)
   - channels ∈ {[32,64,64], [64,128,128]}

Compute budgeting idea:
- Run each sweep config for **0.5–1M steps**, 1–2 seeds.
- Select top 1–2 configs for **full run** (5–10M steps, 5+ seeds).

### Observation Space Notes (current understanding)
**Grid observation** (when `include_grid_in_observation=True`):
- `occupancy_grid`: float32 array shaped `[C, H, W]` with values in `[0, 1]`.
- Channels follow `GridConfig.channels`; recommended for RL: `OBSTACLES`, `PEDESTRIANS`, `COMBINED`.

**Grid metadata fields** (flattened into the top-level observation dict for SB3):
- `occupancy_grid_meta_origin` (shape `(2,)`)
- `occupancy_grid_meta_resolution` (shape `(1,)`)
- `occupancy_grid_meta_size` (shape `(2,)`)
- `occupancy_grid_meta_use_ego_frame` (shape `(1,)`)
- `occupancy_grid_meta_center_on_robot` (shape `(1,)`)
- `occupancy_grid_meta_channel_indices` (shape `(4,)`, int32)
- `occupancy_grid_meta_robot_pose` (shape `(3,)`)

**SocNav structured observation** (when `ObservationMode.SOCNAV_STRUCT`):
- `robot`: position (2), heading (1), speed (1), radius (1)
- `goal`: current (2), next (2)  ← **next waypoint info available here**
- `pedestrians`: positions (N×2), velocities (N×2, ego frame), radius (1), count (1)
- `map`: size (2)
- `sim`: timestep (1)

**Default lidar observation** (non-SocNav):
- `rays`: stacked LiDAR history (timesteps × num_rays)
- `drive_state`: stacked history of `[speed_x, speed_rot, target_distance, target_angle, next_target_angle]`
  - `next_target_angle` only if `sim_config.use_next_goal=True` (default True)

**SocNav + SB3 flattening behavior**
- For SB3 compatibility, SocNav nested dicts are flattened to top-level keys
  (e.g., `robot_position`, `goal_current`, `pedestrians_positions`) and then
  grid + metadata fields are appended.
