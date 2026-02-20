# Model Registry

This folder contains a human-readable registry (`model/registry.md`) and a
machine-readable registry (`model/registry.yaml`) for trained policies.

## Auto-population

Registry entries are designed to be auto-populated by tools and training
workflows. Use `robot_sf.models.upsert_registry_entry(...)` in your pipelines or
notebooks to insert/update entries without manual YAML editing.

### Registry fields (summary)

- `model_id` (required): unique identifier used by `resolve_model_path(...)`.
- `local_path`: local checkout path to the model file (if available).
- `config_path`: training config used to produce the model.
- `commit`: git commit hash for reproducibility.
- W&B metadata (optional): either `wandb_run_path` or `wandb_entity` +
  `wandb_project` + `wandb_run_id` so auto-download works.
- `wandb_file`: file to download from the run (defaults to `model.zip`).
- `tags` / `notes`: free-form metadata for discovery.

### Minimal schema (YAML)

```yaml
version: 1
models:
  - model_id: my_model_id
    display_name: Short human-readable title
    local_path: model/my_model_id/model.zip
    config_path: configs/training/...
    commit: abcdef123456
    wandb_run_path: entity/project/run_id
    wandb_entity:
    wandb_project:
    wandb_run_id: run_id
    wandb_file: model.zip
    tags: ["ppo", "socnav"]
    notes: ["Add any extra context here."]
```

## Using models from the registry

Programmatic usage:

```python
from robot_sf.models import resolve_model_path

path = resolve_model_path(
    "ppo_expert_grid_socnav_403_diffdrive_reverse_no_holdout_15m_20260123T153448",
    allow_download=True,
)
```

If the model is not present locally and W&B metadata is configured in
`model/registry.yaml`, the helper will download the artifact into
`output/model_cache/<model_id>/`.

## Notes

- Keep the per-model README in each model folder for deep context (observation,
  training config, known limitations).
- The YAML registry is the source of truth for automation and should contain
  the W&B run metadata when available.

## Predictive Planner Models

Current predictive planner model ids in `model/registry.yaml` include:

- `predictive_rgl_v1`
- `predictive_rgl_full_v1`
- `predictive_rgl_sweep_h256_mp4_s42`
- `predictive_rgl_sweep_h192_mp3_s7_wd5e5`
- `predictive_proxy_selected_v1`

Recommended benchmark default:

- `predictive_proxy_selected_v1`

The camera-ready predictive config uses this id via
`configs/algos/prediction_planner_camera_ready.yaml`.
