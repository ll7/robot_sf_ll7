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
- W&B metadata (optional): prefer `wandb_artifact_path` for durable model artifacts;
  otherwise use either `wandb_run_path` or `wandb_entity` + `wandb_project` +
  `wandb_run_id` so auto-download works.
- `wandb_file`: file to download from the run (defaults to `model.zip`).
- `github_release`: public GitHub release pointer used for credential-free
  hydration. Include `repository`, `tag`, `asset_name`, `file_name`, `sha256`,
  and `size_bytes`.
- `local_only`: mark entries that are only valid when the recorded `local_path`
  exists on the current machine.
- `replacement_model_id`: optional migration target surfaced in local-only
  resolution errors.
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
    wandb_artifact_path: entity/project/artifact_name:version
    github_release:
      repository: ll7/robot_sf_ll7
      tag: artifact/models-2026-05-registry-v1
      asset_name: my_model_id-model.zip
      file_name: model.zip
      sha256: abc123...
      size_bytes: 123456
    local_only: false
    replacement_model_id: my_model_id_v2
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

If the model is not present locally and `github_release` metadata is configured
in `model/registry.yaml`, the helper downloads the public release asset into
`output/model_cache/<model_id>/` and verifies the recorded SHA256 before returning
the path. Entries without a public pointer can still use W&B metadata as a
private/backfill provenance path.

### Durable vs local cache fields

Treat `github_release` as the preferred public checkpoint pointer when it is
present. Treat `wandb_artifact_path` as the preferred lineage/backfill pointer.
Treat `local_path` under `output/model_cache/` as a cache location that may be absent in a fresh checkout.
If a paper-facing registry entry still has `commit: null`, the W&B run or artifact pointer is the
recoverable source, but the missing source commit should remain visible as a provenance caveat until
it is repaired.

## Notes

- Keep the per-model README in each model folder for deep context (observation,
  training config, known limitations).
- The YAML registry is the source of truth for automation and should contain
  the W&B run metadata when available.

## PPO Expert Model

Current PPO expert model id in `model/registry.yaml`:

- `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`

Deprecated legacy entries kept for traceability:

- `ppo_expert_br06_v2_15m_all_maps_20260302T152332` (killed early, deprecated)
- `ppo_expert_br06_v2_15m_all_maps_20260303T074433`

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
