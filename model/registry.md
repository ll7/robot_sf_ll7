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
- `github_release` metadata (optional): preferred public retrieval pointer for promoted or
  preserved artifacts. Include `repo`, `tag`, `asset_name`, `url`, `sha256`, and `size_bytes`.
- W&B metadata (optional): lineage and private/backfill provenance. Prefer `wandb_artifact_path`
  when preserving W&B provenance; otherwise use either `wandb_run_path` or `wandb_entity` +
  `wandb_project` + `wandb_run_id`.
- `wandb_file`: file to download from the run (defaults to `model.zip`).
- `local_only`: mark entries that are only valid when the recorded `local_path`
  exists on the current machine.
- `replacement_model_id`: optional migration target surfaced in local-only
  resolution errors.
- `benchmark_promotion` (required for benchmark-promoted learned checkpoints): observation-track
  claim boundary and policy input contract.
- `tags` / `notes`: free-form metadata for discovery.

### Benchmark promotion metadata

Learned checkpoints promoted for benchmark claims must include `benchmark_promotion` so LiDAR,
grid/SocNav, privileged-state, and adapter-derived policies cannot be confused in reports. The
vocabulary follows `docs/context/issue_1612_observation_track_architecture.md`.

Required for `claim_boundary: benchmark_promoted` or `benchmark_candidate`:

- `benchmark_track`: stable aggregation lane such as `grid_socnav_v1` or `lidar_2d_v1`.
- `track_schema_version`: schema slug such as `observation-track.v1`.
- `observation_level`: benchmark observation-level vocabulary from
  `robot_sf/benchmark/observation_levels.py`.
- `observation_mode`: active environment or policy observation mode, such as `socnav_state`,
  `sensor_fusion_state`, or a documented learned-policy dict contract.
- `allowed_observation_keys`: concrete policy input keys allowed at evaluation time.
- `goal_encoding`: how the current route or goal enters the observation.
- `sensor_geometry`: grid, ray, stack, range, or other sensor-shape details that define the track.
- `privileged_input_status`: explicit statement about evaluation-time privileged inputs.

Research-only, smoke-only, legacy, or otherwise non-benchmark entries may omit those track fields
only when `benchmark_promotion.claim_boundary` is one of `research_only`, `smoke_only`,
`legacy_non_track`, or `not_for_benchmark` and `non_benchmark_reason` explains the claim boundary.

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
      repo: owner/repo
      tag: artifact/models-YYYY-MM-registry-v1
      asset_name: my_model_id-model.zip
      url: https://github.com/owner/repo/releases/download/artifact/models-YYYY-MM-registry-v1/my_model_id-model.zip
      sha256: ...
      size_bytes: 123
    benchmark_promotion:
      claim_boundary: benchmark_promoted
      benchmark_track: grid_socnav_v1
      track_schema_version: observation-track.v1
      observation_level: tracked_agents_no_noise
      observation_mode: socnav_state
      allowed_observation_keys: [robot_state, goal, tracked_agents]
      goal_encoding: current route goal in planner observation
      sensor_geometry: tracked-agent state, no LiDAR ray geometry
      privileged_input_status: no evaluation-time privileged inputs
      reference: docs/context/issue_1612_observation_track_architecture.md
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

If the model is not present locally and `github_release` metadata is configured in
`model/registry.yaml`, the helper downloads and verifies the public release asset into
`output/model_cache/<model_id>/`. If no GitHub release pointer is available, W&B metadata remains a
private/provenance fallback.

### Public artifacts vs provenance fields

Treat `github_release` as the preferred public checkpoint pointer for preserved/canonical models.
Treat W&B fields as experiment-lineage provenance unless the entry is intentionally private. Treat
`local_path` under `output/model_cache/` as a cache location that may be absent in a fresh checkout.
If a paper-facing registry entry still has `commit: null`, keep the missing source commit visible as
a provenance caveat until it is repaired.

Promoted benchmark configs are listed in
`configs/benchmarks/promoted_config_surfaces.yaml`. Those config files must use durable ids or
artifact pointers such as `model_id`, not direct `model_path` or `resume_from` values under
`output/`. Validate the boundary with:

```bash
uv run python scripts/validation/check_local_model_artifacts.py
```

### Publishing preserved models

Use `scripts/tools/publish_model_registry_release.py` to migrate W&B-backed model entries to GitHub
release assets:

```bash
uv run python scripts/tools/publish_model_registry_release.py \
  --tag artifact/models-YYYY-MM-registry-v1 \
  --download-missing \
  --execute-upload \
  --create-release \
  --update-registry
```

The script stages model files, per-model metadata JSON, `manifest.json`, and `SHA256SUMS`; uploads
them to the release; and writes `github_release` pointers back into `model/registry.yaml`.

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

### Legacy PPO Snapshot Compatibility

The following legacy PPO checkpoints remain supported through the durable model registry and are
covered by `scripts/validation/check_legacy_ppo_snapshot_parity.py`:

- `ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200`
- `ppo_expert_br06_v2_15m_all_maps_20260302T152332`
- `ppo_expert_br06_v2_15m_all_maps_20260303T074433`

Each supported row must keep a GitHub release pointer with `asset_name`, `sha256`, and
`size_bytes`. The validation script's default inventory mode is cheap and does not download
checkpoints. When a checkpoint is already hydrated, or when download cost is explicitly acceptable,
run the opt-in one-step smoke with:

```bash
uv run python scripts/validation/check_legacy_ppo_snapshot_parity.py \
  --smoke-model-id ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200 \
  --allow-download
```

Root-local debug checkpoints such as `model/run_023.zip`, `model/run_043.zip`,
`model/ppo_model_retrained_10m_2024-09-17.zip`, and
`model/ppo_model_retrained_10m_2025-02-01.zip` are explicitly unsupported for durable compatibility
because they lack registry provenance and release checksums. They may remain useful for local
debugging, but benchmark or regression workflows should use a registry `model_id`.

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
