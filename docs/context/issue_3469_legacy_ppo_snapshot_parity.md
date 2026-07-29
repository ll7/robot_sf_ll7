# Issue #3469 Legacy PPO Snapshot Parity

Issue: [#3469](https://github.com/ll7/robot_sf_ll7/issues/3469)

## Summary

This change turns legacy PPO snapshot compatibility into an executable inventory and smoke contract.
The default check is cheap and deterministic: it verifies that legacy BR-06 PPO checkpoints that
should remain supported are represented in `model/registry.yaml` with durable GitHub release
metadata. It also byte-matches the Phase-A legacy sources against their recorded checksums and
retains an explicit unsupported-local classification for any root-local snapshots that have not
been promoted.

## Phase-A durable legacy entries

PR [#6325](https://github.com/ll7/robot_sf_ll7/pull/6325), implementing issue
[#6321](https://github.com/ll7/robot_sf_ll7/issues/6321) as Phase A of parent issue
[#6268](https://github.com/ll7/robot_sf_ll7/issues/6268), published the previously unregistered
legacy binaries as durable release-backed entries. The release is
[`artifact/legacy-models-2026-07-registry-v1`](https://github.com/ll7/robot_sf_ll7/releases/tag/artifact/legacy-models-2026-07-registry-v1).
The set is nine single-file PPO zip checkpoints plus the three-file GA3C-CADRL TensorFlow
checkpoint published as one tarball bundle. Every entry declares
`benchmark_promotion.claim_boundary: legacy_non_track`; this is provenance and compatibility
evidence, not benchmark evidence.

The canonical registry and manifest are the source of truth for the complete model-id and checksum
list: [`model/registry.yaml`](../../model/registry.yaml),
[`model/registry.md`](../../model/registry.md), and the release `manifest.json`.

## Supported Legacy Registry IDs

- `ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200`
- `ppo_expert_br06_v2_15m_all_maps_20260302T152332`
- `ppo_expert_br06_v2_15m_all_maps_20260303T074433`

Each supported row must retain a `github_release` pointer with `asset_name`, `sha256`, and
`size_bytes`.

## Previously unsupported snapshots now durable

The following files were previously treated as debug-only. Phase A promoted them into the durable
registry while preserving their in-tree bytes and paths; they are now `supported` inventory rows
with `legacy_non_track` claim boundaries:

- `model/run_023.zip`
- `model/run_043.zip`
- `model/ppo_model_retrained_10m_2024-09-17.zip`
- `model/ppo_model_retrained_10m_2025-02-01.zip`

The pedestrian PPO zips and the GA3C-CADRL checkpoint are also covered by the Phase-A durable
inventory. Any future root-local snapshot without a corresponding durable entry remains
`unsupported_local_only`; the validator's guard is retained for that case.

## Commands

Cheap inventory check:

```bash
uv run python scripts/validation/check_legacy_ppo_snapshot_parity.py --json
```

Fresh-cache release hydration and byte-identity check:

```bash
uv run python scripts/validation/check_legacy_ppo_snapshot_parity.py \
  --verify-release-hydration --cache-dir /tmp/legacy-model-cache --json
```

The default inventory checks the in-tree source bytes without downloading. The explicit hydration
mode downloads each single-file release asset into the requested isolated cache, verifies the
GA3C archive and all three component checksums, and confirms that GA3C still resolves to its
existing in-tree checkpoint path. Fresh-cache hydration is provenance proof only; it does not
promote these checkpoints to a benchmark track.

Opt-in hydrated-checkpoint smoke:

```bash
uv run python scripts/validation/check_legacy_ppo_snapshot_parity.py \
  --smoke-model-id ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200 \
  --allow-download
```

## Claim Boundary

The default inventory check is compatibility/provenance evidence, not a performance benchmark. The
opt-in smoke proves that a hydrated checkpoint can load, predict one action, and execute one current
Gymnasium `make_robot_env` step with valid reward/termination/info contract shape. It does not claim
current benchmark performance for old snapshots.
