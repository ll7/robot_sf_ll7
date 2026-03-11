# Predictive Planner Training Workflow

This runbook documents the reproducible training/evaluation flow for the predictive local planner.

For the full concept + architecture + implementation walkthrough, see:

- `docs/training/predictive_planner_complete_tutorial.md`
- See especially:
  - Section 6 for dataset semantics (`state/target/mask/target_mask`),
  - Section 8 for deterministic prediction behavior,
  - Sections 9-10 for benchmark integration and steering building blocks,
  - Section 11 for pipeline exit-code meaning (`0` vs `2`),
  - Section 12 for FAQ + consequences (MCTS vs sampled rollout, probabilities,
    training-stop reasons, zero-loss diagnosis, and recommended full-profile length).

## Scope

- Data collection for trajectory prediction training
- Model training and checkpoint selection
- Hard-seed diagnostics and campaign evaluation

## Prerequisites

- `uv sync --all-extras`
- SocNav scenario matrix available (for example `configs/scenarios/classic_interactions.yaml`)
- Write permissions for `output/tmp/predictive_planner/`

## Canonical config-first run (recommended)

Run the full predictive pipeline (all maps + randomized seeds + W&B logging + final summary):

```bash
uv run python scripts/training/run_predictive_training_pipeline.py \
  --config configs/training/predictive/predictive_br07_all_maps_randomized_full.yaml \
  --log-level INFO
```

This single command performs:
- base dataset collection over all scenarios with randomized seeds,
- hard-case dataset collection,
- mixed dataset creation,
- predictive model training with proxy selection and W&B logging,
- final evaluation, hard-seed diagnostics, and success campaign,
- final performance summary artifact under the run directory.

Runtime note:
- The pipeline exits `0` only when all post-training gates pass.
- It exits `2` when at least one gate fails (for example evaluation success-rate threshold),
  while still writing full diagnostics (`final_performance_summary.json/.md`).

## Training health checks (must verify)

Before trusting a long run, confirm:

1. Dataset ratios are non-zero:
   - `active_agent_ratio > 0`
   - `active_target_ratio > 0`
2. Training metrics are non-zero and changing:
   - `train_loss`, `val_loss`, `val_ade`, `val_fde`
3. Proxy checks are being executed at configured cadence.

Hard stop condition:
- If all train/val metrics are exactly `0.0` for many epochs, treat as data ingestion failure
  and stop the run.

Observation format compatibility:
- Data collectors now support both nested SOCNAV observations (`obs["pedestrians"]`) and
  flattened keys (`pedestrians_positions`, `pedestrians_velocities`, `pedestrians_count`).
- This avoids silent all-zero pedestrian masks in environments returning flattened payloads.

## 1) Collect training data (manual path)

Base rollout data:

```bash
uv run python scripts/training/collect_predictive_planner_data.py \
  --episodes 200 \
  --output output/tmp/predictive_planner/datasets/predictive_rollouts_full_v1.npz
```

Hard-case-focused data:

```bash
uv run python scripts/training/collect_predictive_hardcase_data.py \
  --scenario-matrix configs/scenarios/classic_interactions.yaml \
  --seed-manifest configs/benchmarks/predictive_hard_seeds_v1.yaml \
  --output output/tmp/predictive_planner/datasets/predictive_rollouts_hardcase_v1.npz
```

Mixed dataset:

```bash
uv run python scripts/training/build_predictive_mixed_dataset.py \
  --base-dataset output/tmp/predictive_planner/datasets/predictive_rollouts_full_v1.npz \
  --hardcase-dataset output/tmp/predictive_planner/datasets/predictive_rollouts_hardcase_v1.npz \
  --output output/tmp/predictive_planner/datasets/predictive_rollouts_mixed_v1.npz
```

## 2) Train predictive model

```bash
uv run python scripts/training/train_predictive_planner.py \
  --dataset output/tmp/predictive_planner/datasets/predictive_rollouts_mixed_v1.npz \
  --output-dir output/tmp/predictive_planner/training/predictive_proxy_selected_v2 \
  --select-by-proxy \
  --proxy-scenario-matrix configs/scenarios/classic_interactions.yaml \
  --proxy-seed-manifest configs/benchmarks/predictive_hard_seeds_v1.yaml
```

Outputs:
- `predictive_model.pt`
- `training_summary.json`
- optional `proxy_eval/*.jsonl`

## 3) Register model id

Ensure `model/registry.yaml` has a matching `model_id` entry and `local_path` for the checkpoint.

Current expected id for benchmark config:
- `predictive_proxy_selected_v2` (default profile)
- `predictive_proxy_selected_v2_full` (full profile)

## 4) Evaluate benchmark behavior

Single evaluation:

```bash
uv run python scripts/validation/evaluate_predictive_planner.py \
  --checkpoint output/tmp/predictive_planner/training/predictive_proxy_selected_v2_full/predictive_model.pt \
  --scenario-matrix configs/scenarios/classic_interactions.yaml
```

Hard-seed diagnostics:

```bash
uv run python scripts/validation/run_predictive_hard_seed_diagnostics.py \
  --scenario-matrix configs/scenarios/classic_interactions.yaml \
  --seed-manifest configs/benchmarks/predictive_hard_seeds_v1.yaml \
  --checkpoint output/tmp/predictive_planner/training/predictive_proxy_selected_v2_full/predictive_model.pt
```

Planner/config campaign:

```bash
uv run python scripts/validation/run_predictive_success_campaign.py \
  --checkpoints output/tmp/predictive_planner/training/predictive_proxy_selected_v2_full/predictive_model.pt \
  --scenario-matrix configs/scenarios/classic_interactions.yaml \
  --hard-seed-manifest configs/benchmarks/predictive_hard_seeds_v1.yaml \
  --planner-grid configs/benchmarks/predictive_sweep_planner_grid_v1.yaml
```

## 5) Quality gates

At minimum:
- No regression in global benchmark success on classic scenario matrix.
- Hard-seed success and failure taxonomy are tracked and compared to baseline.
- Planner adapter tests pass.

## Notes on reproducibility

- Keep algorithm configs explicit (full YAML) for A/B comparisons.
- Always archive campaign summary/report artifacts from `output/tmp/predictive_planner/campaigns/`.
- Prefer the config-first pipeline command above for paper-facing comparable runs.
