# PPO `num_envs` Benchmark on `imech156-u`

This benchmark measures host utilization and early-training stability for the current PPO training surface on `imech156-u`.

Scope:
- same PPO/reward/foresight setup,
- same warm-start checkpoint,
- same 1M continuation budget,
- same 500k evaluation cadence,
- only `num_envs` changes.

## Configs

Run these configs in this order:

1. `configs/training/ppo/benchmark_num_envs/expert_ppo_issue_576_br06_v8_num_envs_benchmark_32_1m.yaml`
2. `configs/training/ppo/benchmark_num_envs/expert_ppo_issue_576_br06_v8_num_envs_benchmark_30_1m.yaml`
3. `configs/training/ppo/benchmark_num_envs/expert_ppo_issue_576_br06_v8_num_envs_benchmark_16_1m.yaml`
4. `configs/training/ppo/benchmark_num_envs/expert_ppo_issue_576_br06_v8_num_envs_benchmark_14_1m.yaml`
5. `configs/training/ppo/benchmark_num_envs/expert_ppo_issue_576_br06_v8_num_envs_benchmark_08_1m.yaml`

Run them one at a time. Do not run them concurrently on `imech156-u`, otherwise the throughput comparison is invalid.

## Canonical command

```bash
uv run python scripts/training/train_ppo.py \
  --config <config-path> \
  --log-level WARNING
```

Example:

```bash
uv run python scripts/training/train_ppo.py \
  --config configs/training/ppo/benchmark_num_envs/expert_ppo_issue_576_br06_v8_num_envs_benchmark_32_1m.yaml \
  --log-level WARNING
```

## What to compare

Primary metrics:
- `perf/train_env_steps_per_sec`
- `time/fps`
- wall-clock time to 15.5M and 16.0M timesteps

Secondary metrics:
- `eval/success_rate`
- `eval/collision_rate`
- `eval/snqi`
- whether the run crashes or stalls

Interpretation rule:
- pick the highest-throughput setup that does not materially degrade early eval success/collision behavior.
- if 30/32 are only marginally faster than 14/16 but less stable, prefer 14/16.

## Summarizer

After the runs finish:

```bash
uv run python scripts/tools/summarize_ppo_num_envs_benchmark.py \
  --entity ll7 \
  --project robot_sf \
  --group benchmark-num-envs-imech156u-v8 \
  --output-json output/benchmarks/ppo_num_envs_imech156u_summary.json \
  --output-md output/benchmarks/ppo_num_envs_imech156u_summary.md
```

This emits a compact table with throughput and latest eval metrics per run.
