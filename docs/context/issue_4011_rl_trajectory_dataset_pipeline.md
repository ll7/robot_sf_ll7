# Issue #4011 RL Trajectory Dataset Pipeline

This slice adds the first reusable offline reinforcement learning trajectory dataset contract for
simulation-run-derived records. It standardizes episode-major JSONL rows containing observations,
actions, rewards, return-to-go, terminal and truncated flags, pedestrian state, robot state,
scenario provenance, deterministic split assignment, and a manifest with leakage checks.

## Scope

- Schema version: `RLTrajectoryDataset.v1` with `RLTrajectoryEpisode.v1` JSONL rows.
- Manifest version: `rl_trajectory_dataset_manifest.v1`.
- Reward convention: `environment_step_reward`.
- Return convention: `undiscounted_future_return_to_go`.
- Source path: benchmark episode JSONL records whose `algorithm_metadata.simulation_step_trace`
  contains per-step `rl.reward`, `rl.terminated`, and `rl.truncated`.
- Durable boundary: committed evidence is a tiny preview fixture and manifest only. Full datasets
  under `output/` remain worktree-local until promoted to a durable artifact store with an explicit
  URI and checksum.

## Entry Points

- `robot_sf/benchmark/rl_trajectory_dataset.py` loads, validates, flattens, and writes the dataset.
- `robot_sf/benchmark/schemas/rl_trajectory_dataset_manifest.schema.v1.json` defines the manifest
  schema; semantic split leakage checks live in
  `robot_sf/benchmark/schemas/rl_trajectory_dataset_schema.py`.
- `scripts/benchmark/record_rl_trajectory_dataset.py` converts benchmark episode traces into JSONL
  dataset rows plus a manifest.
- `scripts/validation/validate_trajectory_dataset.py --path <dataset.jsonl>` recognizes the new
  JSONL contract through `TrajectoryDatasetValidator`.

## Out Of Scope

- No offline reinforcement learning policy training.
- No Decision Transformer training.
- No offline-to-online fine-tuning.
- No benchmark campaign, Slurm job, graphics processing unit job, or paper/dissertation claim.

## Validation

Focused validation should cover:

```bash
uv run pytest \
  tests/benchmark/test_rl_trajectory_dataset.py \
  tests/contract/test_rl_trajectory_dataset_manifest_schema.py \
  tests/training/test_collect_expert_trajectories_dt_preflight.py \
  tests/validation/test_validate_trajectory_dataset_cli.py \
  -q
```

The preview evidence bundle under
`docs/context/evidence/issue_4011_rl_trajectory_dataset_smoke_2026-07-02/` is intentionally tiny and
is not a durable training dependency.
