# Predictive Planner Training Workflow

This runbook documents the reproducible training/evaluation flow for the predictive local planner.

For the full concept + architecture + implementation walkthrough, see:

- `docs/training/predictive_planner_complete_tutorial.md`

## Scope

- Data collection for trajectory prediction training
- Model training and checkpoint selection
- Hard-seed diagnostics and campaign evaluation

## Prerequisites

- `uv sync --all-extras`
- SocNav scenario matrix available (for example `configs/scenarios/classic_interactions.yaml`)
- Write permissions for `output/tmp/predictive_planner/`

## 1) Collect training data

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
  --output-dir output/tmp/predictive_planner/training/predictive_proxy_selected_v1 \
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
- `predictive_proxy_selected_v1`

## 4) Evaluate benchmark behavior

Single evaluation:

```bash
uv run python scripts/validation/evaluate_predictive_planner.py \
  --checkpoint output/tmp/predictive_planner/training/predictive_proxy_selected_v1/predictive_model.pt \
  --scenario-matrix configs/scenarios/classic_interactions.yaml
```

Hard-seed diagnostics:

```bash
uv run python scripts/validation/run_predictive_hard_seed_diagnostics.py \
  --scenario-matrix configs/scenarios/classic_interactions.yaml \
  --seed-manifest configs/benchmarks/predictive_hard_seeds_v1.yaml \
  --checkpoint output/tmp/predictive_planner/training/predictive_proxy_selected_v1/predictive_model.pt
```

Planner/config campaign:

```bash
uv run python scripts/validation/run_predictive_success_campaign.py \
  --checkpoints output/tmp/predictive_planner/training/predictive_proxy_selected_v1/predictive_model.pt \
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
