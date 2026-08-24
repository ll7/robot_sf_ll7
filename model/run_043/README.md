# Legacy PPO debug checkpoint run_043 (durable)

## Artifact
- Model id: `legacy_ppo_run_043`
- Durable asset: `legacy_ppo_run_043.zip`
- Release: `ll7/robot_sf_ll7` tag `artifact/legacy-models-2026-07-registry-v1`
- Release URL: https://github.com/ll7/robot_sf_ll7/releases/download/artifact/legacy-models-2026-07-registry-v1/legacy_ppo_run_043.zip
- Registry SHA-256: `70843ac56cb5f2a0532ce059074927398f4210505e862645bf8e46cec5bfe466`
- Resolves through `robot_sf.models.registry.resolve_model_path("legacy_ppo_run_043")`
  into `output/model_cache/legacy_ppo_run_043/legacy_ppo_run_043.zip`.

## Cutover
Phase B of #6268: the in-tree binary `model/run_043.zip` is removed and replaced by this
stub. Code that needs the checkpoint resolves it through the registry/release pointer above;
the bytes are the same as the released asset (SHA-256 pinned).
