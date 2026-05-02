# PPO num_envs benchmark configs

These configs run the issue-576 BR-06 v8 host-scaling benchmark on imech156-u. The numbered YAML
files are the launchable configs; `expert_ppo_issue_576_br06_v8_num_envs_benchmark_base_1m.yaml`
holds shared settings so the per-variant files only carry the explicit `policy_id`, `num_envs`,
and W&B tags.

Canonical command:

```bash
uv run python scripts/training/train_ppo.py --config configs/training/ppo/benchmark_num_envs/expert_ppo_issue_576_br06_v8_num_envs_benchmark_08_1m.yaml
```

Replace `08` with `14`, `16`, `30`, or `32` to run the other host-scaling variants.
