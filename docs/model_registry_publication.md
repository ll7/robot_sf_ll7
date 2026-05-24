# Model Registry Publication

The model registry supports a split public/provenance artifact policy.

- `github_release` is the credential-free hydration path for public registry
  entries. `resolve_model_path(..., allow_download=True)` downloads the release
  asset and verifies the recorded SHA256 before returning it.
- W&B metadata remains lineage/backfill metadata. It can identify the original
  run or artifact, but public users should not need W&B credentials for canonical
  registry assets.
- `local_path` values under `output/model_cache/` are cache hints only. A fresh
  checkout may not have those files.
- Non-public entries should either be marked `local_only: true` with a
  replacement hint or retain a clear private W&B/local-only boundary.

## Current Public Release

The current public bundle is:

- Repository: `ll7/robot_sf_ll7`
- Release tag: `artifact/models-2026-05-registry-v1`
- Manifest assets: `manifest.json` and `SHA256SUMS`

The release contains 11 registry-backed model assets: the canonical PPO
benchmark checkpoint, seven historical/runner-up PPO checkpoints, and three
predictive planner checkpoints. `model/registry.yaml` records each asset name,
cache filename, SHA256, size, and metadata asset name under `github_release`.

## Updating Pointers

After staging or refreshing a release manifest, update the registry with:

```bash
uv run python scripts/tools/publish_model_registry_release.py apply-manifest \
  --manifest output/issue_1458_release/manifest.json \
  --sha256s output/issue_1458_release/SHA256SUMS \
  --write
```

Review the diff before committing. The helper inserts or replaces only
`github_release` blocks for matching model IDs.

To audit publication coverage without editing:

```bash
uv run python scripts/tools/publish_model_registry_release.py inventory \
  --manifest output/issue_1458_release/manifest.json
```

## Verification

Use a fresh cache directory when proving public access, for example:

```bash
uv run python - <<'PY'
from robot_sf.models import resolve_model_path

path = resolve_model_path(
    "ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417",
    cache_dir="output/model_registry_public_smoke",
)
print(path)
PY
```

The command should succeed without W&B credentials and should fail if the public
asset is missing or its SHA256 differs from `model/registry.yaml`.
