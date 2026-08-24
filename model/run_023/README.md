# Legacy PPO debug checkpoint run_023 (durable)

## Artifact
- Model id: `legacy_ppo_run_023`
- Durable asset: `legacy_ppo_run_023.zip`
- Release: `ll7/robot_sf_ll7` tag `artifact/legacy-models-2026-07-registry-v1`
- Release URL: https://github.com/ll7/robot_sf_ll7/releases/download/artifact/legacy-models-2026-07-registry-v1/legacy_ppo_run_023.zip
- Registry SHA-256: `54333166928fcb028a47b8f8d16bf10eaa997ca49cf6fc15224f31d7c63d5dbf`
- Resolves through `robot_sf.models.registry.resolve_model_path("legacy_ppo_run_023")`
  into `output/model_cache/legacy_ppo_run_023/legacy_ppo_run_023.zip`.

## Cutover
Phase B of #6268: the in-tree binary `model/run_023.zip` is removed and replaced by this
stub. Code that needs the checkpoint resolves it through the registry/release pointer above;
the bytes are the same as the released asset (SHA-256 pinned). The legacy observation adapter
lives in `robot_sf/training/observation_wrappers.py` (`run_023` flattened format).
