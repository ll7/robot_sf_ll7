# Model Registry Publication Workflow

This workflow preserves selected trained policies as public GitHub release assets while keeping W&B
as private experiment-lineage provenance.

## Why

Training run data has not been published broadly. Public users should not need W&B or Hugging Face
project credentials to load preserved Robot SF models from `model/registry.yaml`.

For preserved models, the public source of truth is a curated GitHub release asset with checksum
metadata. W&B fields may remain in the registry as provenance/backfill fields, but model hydration
should prefer `github_release`.

## When

Run this workflow when a trained policy becomes one of:

- promoted or canonical,
- paper-facing,
- needed by public examples or benchmark reproduction,
- worth preserving beyond local `output/` and private W&B.

Do not publish scratch runs, failed runs, broad training logs, or exploratory checkpoints unless a
maintainer explicitly promotes them.

## What

For each selected W&B-backed registry entry, publish:

- the model artifact, for example `model.zip` or `predictive_model.pt`,
- a legal bundle containing `LICENSE`, `MODEL_CARD.md`, `PROVENANCE.json`, and any
  declared notices,
- `<model_id>-metadata.json` with registry provenance,
- release-level `manifest.json`,
- release-level `SHA256SUMS`.

Then update the registry entry with:

```yaml
public_artifact_source: github_release
github_release:
  repo: ll7/robot_sf_ll7
  tag: artifact/models-YYYY-MM-registry-v1
  asset_name: <model_id>-model.zip
  url: https://github.com/ll7/robot_sf_ll7/releases/download/<tag>/<asset>
  sha256: <sha256>
  size_bytes: <bytes>
  metadata_asset: <model_id>-metadata.json
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
```

Use a non-benchmark boundary such as `research_only` or `smoke_only` plus
`non_benchmark_reason` when the entry is preserved for analysis or launch checks but is not eligible
for benchmark claims.

Every publishable entry must also include a `licensing` mapping. It must state the
SPDX license expression, copyright holder, redistribution basis, source repository
and revision, source-archive checksum, weights origin, training-code and
training-data licenses, and repository-relative `license_file`, `model_card_file`,
and `included_notices` paths. Publication fails closed when any of this rights
evidence is missing; repository or inference-code licensing is not inferred as a
weights license.

The publisher also audits every existing `github_release` row before staging a new release. A
public row without the licensing mapping or its hashed `legal_bundle` blocks the entire operation;
this prevents a new release from masking older unverified public artifacts. Missing local model
files likewise block the operation rather than producing a partial release; use
`--download-missing` only when the registry's immutable source metadata is available.

## How

Inventory and stage the current W&B-backed entries without uploading:

```bash
uv run python scripts/tools/publish_model_registry_release.py \
  --tag artifact/models-YYYY-MM-registry-v1 \
  --download-missing \
  --output-json output/model_registry_release/plan.json
```

Publish a new release and update `model/registry.yaml`:

```bash
uv run python scripts/tools/publish_model_registry_release.py \
  --tag artifact/models-YYYY-MM-registry-v1 \
  --download-missing \
  --execute-upload \
  --create-release \
  --update-registry \
  --output-json output/model_registry_release/published.json
```

Publish one model while testing the flow:

```bash
uv run python scripts/tools/publish_model_registry_release.py \
  --tag artifact/models-YYYY-MM-registry-v1 \
  --model-id ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417 \
  --download-missing \
  --execute-upload \
  --create-release \
  --update-registry
```

If the release already exists, omit `--create-release`.

## Agent Coding Contract

An implementation agent should:

1. Read `model/registry.yaml` and select W&B-backed entries.
2. Resolve each selected model from existing `local_path` or hydrate it with `--download-missing`.
3. Stage release assets under `output/model_registry_release/<tag>/`.
4. Upload staged assets with `gh release upload`.
5. Update `github_release` pointers in `model/registry.yaml`.
6. Run targeted registry tests.
7. Verify at least one fresh-cache download through `resolve_model_path`.

Agents must not publish unselected scratch runs or training logs.

## Verification

After publication:

```bash
uv run pytest tests/models/test_registry.py tests/tools/test_publish_model_registry_release.py
```

Verify the release asset can be downloaded without using W&B:

```bash
rm -rf output/model_cache/<model_id>
uv run python - <<'PY'
from robot_sf.models import resolve_model_path
path = resolve_model_path("<model_id>", allow_download=True)
print(path)
PY
sha256sum output/model_cache/<model_id>/<asset_name>
```

The checksum must match `model/registry.yaml`.

### CI cache and transient release throttling

GitHub-release model downloads use a bounded three-attempt retry policy for HTTP 429 and 5xx
responses. A numeric or HTTP-date `Retry-After` header is honored, capped at 30 seconds; malformed
or missing headers use the local backoff. Every downloaded artifact is still checksum-verified
before atomic publication. A cache entry is reused only when its registry-pinned SHA-256 matches;
an invalid entry is replaced, never trusted. This verification also applies when a GitHub-release
entry's `local_path` points into `output/model_cache/<model_id>/`. Entries whose local path
intentionally names an unpacked in-tree checkpoint while the release asset is an archive retain that
local checkpoint contract.

CI may use the repository's pre-populated, SHA-keyed model cache (`actions/cache`) as an optimization,
but this path does not add credentials, a mirror, or a redistribution authorization. If the public
release endpoint remains unavailable after bounded retries, preflight exits with a blocking
`external_unavailable` result and must not admit a fallback or nominal readiness result.

Also inspect:

```bash
gh release view artifact/models-YYYY-MM-registry-v1 --repo ll7/robot_sf_ll7
gh release download artifact/models-YYYY-MM-registry-v1 \
  --repo ll7/robot_sf_ll7 \
  --pattern manifest.json \
  --pattern SHA256SUMS \
  --dir output/model_registry_release/verify
```
