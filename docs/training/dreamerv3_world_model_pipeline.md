# DreamerV3 World-Model Pretraining Pipeline

[← Back to Documentation Index](../README.md)

This workflow stages DreamerV3 training so the expensive online fine-tune only starts
after the world model has passed explicit data and pretraining quality gates.

## Intent

Use this pipeline when the direct Dreamer full run is too unstable or too sample-inefficient.
The staged flow is:

1. collect teacher episodes on the exact Dreamer wrapped environment contract,
2. pretrain the Dreamer world model offline on those episodes,
3. enforce a world-model quality gate from the recorded run summary,
4. warm-start the online Dreamer RLModule from the pretrained checkpoint.

## Canonical local commands

Collect an offline dataset:

```bash
source .venv/bin/activate
uv run --extra rllib python scripts/training/collect_dreamer_world_model_episodes.py \
  --dreamer-config configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_world_model_pretrain.yaml \
  --dataset-id br08_world_model_ppo_teacher \
  --teacher-model-id <teacher-model-id> \
  --episodes 200
```

Run offline world-model pretraining:

```bash
source .venv/bin/activate
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_world_model_pretrain.yaml
```

Validate the offline pretrain run:

```bash
source .venv/bin/activate
uv run --extra rllib python scripts/training/check_dreamer_world_model_pretrain.py \
  --run-summary output/dreamerv3/<run-id>/run_summary.json
```

Warm-start the balanced online run from the pretrained RLModule:

```bash
source .venv/bin/activate
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_full_balanced_gpu_r8_warmstart.yaml \
  --restore-checkpoint output/dreamerv3/<pretrain-run>/checkpoints/checkpoint_000040 \
  --restore-component rl_module
```

## Pipeline wrapper

The canonical orchestration entry point is:

```bash
scripts/training/run_dreamerv3_world_model_pipeline.sh all
```

Supported modes:

- `collect`
- `pretrain`
- `gate`
- `finetune`
- `all`

Teacher selection is controlled through environment variables:

- `DREAMER_WM_TEACHER_MODEL_ID`
- `DREAMER_WM_TEACHER_CHECKPOINT`
- `DREAMER_WM_TEACHER_MODE`
- `DREAMER_WM_COLLECT_EPISODES`
- `DREAMER_WM_DATASET_ID`
- `DREAMER_WM_DATASET_DIR`

## Quality gates

### Dataset gate

`scripts/training/collect_dreamer_world_model_episodes.py` fails fast when the exported
dataset is clearly degenerate. The current checks require:

- minimum episode count,
- minimum mean episode length,
- finite returns and finite actions,
- minimum action standard deviation.

This is a coverage sanity check, not a claim of dataset optimality.

### World-model pretrain gate

`scripts/training/check_dreamer_world_model_pretrain.py` validates the pretraining run from
`run_summary.json`. The current checks require:

- minimum completed iterations,
- at least two finite `WORLD_MODEL_L_total` points,
- final/initial world-model loss ratio below a threshold,
- minimum absolute world-model loss drop,
- minimum median trained-step throughput.

These checks are intentionally simple. They are meant to reject obviously bad runs before a
long Slurm fine-tune starts.

## Recommended cluster workflow

Use the dedicated batch script:

```bash
scripts/dev/sbatch_use_max_time.sh SLURM/Auxme/auxme_pro6000_dreamer_world_model_pipeline.sl
```

Set either `DREAMER_WM_TEACHER_MODEL_ID` or `DREAMER_WM_TEACHER_CHECKPOINT` before
submission. The batch wrapper runs the staged pipeline directly and preserves artifacts under
the canonical `output/` tree.

## Notes

- RLlib offline episode reading requires `msgpack-numpy` in addition to `ray[rllib]`.
- The current warm-start restores the `rl_module` component only. That preserves the
  pretrained world model while letting online learner state be rebuilt cleanly.
- This does not guarantee successful control learning, but it raises the probability that
  the model is actually learning dynamics before it is asked to solve the task online.
