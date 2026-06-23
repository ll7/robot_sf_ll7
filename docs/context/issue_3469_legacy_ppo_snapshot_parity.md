# Issue #3469 Legacy PPO Snapshot Parity

Issue: [#3469](https://github.com/ll7/robot_sf_ll7/issues/3469)

## Summary

This change turns legacy PPO snapshot compatibility into an executable inventory and smoke contract.
The default check is cheap and deterministic: it verifies that legacy BR-06 PPO checkpoints that
should remain supported are represented in `model/registry.yaml` with durable GitHub release
metadata, and it records root-local debug `.zip` files as explicitly unsupported for durable
compatibility.

## Supported Legacy Registry IDs

- `ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200`
- `ppo_expert_br06_v2_15m_all_maps_20260302T152332`
- `ppo_expert_br06_v2_15m_all_maps_20260303T074433`

Each supported row must retain a `github_release` pointer with `asset_name`, `sha256`, and
`size_bytes`.

## Unsupported Local Snapshots

The following root-local files are treated as debug-only unless promoted into the registry with
durable provenance:

- `model/run_023.zip`
- `model/run_043.zip`
- `model/ppo_model_retrained_10m_2024-09-17.zip`
- `model/ppo_model_retrained_10m_2025-02-01.zip`

## Commands

Cheap inventory check:

```bash
uv run python scripts/validation/check_legacy_ppo_snapshot_parity.py --json
```

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
