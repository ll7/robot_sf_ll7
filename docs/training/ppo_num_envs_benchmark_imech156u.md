# PPO `num_envs` Benchmark on `imech156-u`

This benchmark measures host utilization and early-training stability for the current PPO training surface on `imech156-u`.

Scope:
- same PPO/reward/foresight setup,
- same warm-start checkpoint,
- same 1M continuation budget,
- same 500k evaluation cadence,
- only `num_envs` changes.

## Results

The benchmark completed for `num_envs` 8, 14, 16, 30, and 32. Two different
recommendations came out of it:

- `num_envs=30` is the best raw-throughput point on `imech156-u`.
- `num_envs=8` produced the strongest full 141-episode evaluation candidate.

Measured host-throughput snapshot from W&B:

| `num_envs` | `time/fps` | `perf/train_env_steps_per_sec` | Peak RSS (MB) | Peak system memory % |
| --- | ---: | ---: | ---: | ---: |
| 8 | 310.8 | 316.3 | 12013 | 31.0 |
| 14 | 397.5 | 410.3 | 18119 | 49.6 |
| 16 | 405.2 | 398.6 | 21515 | 57.4 |
| 30 | 532.7 | 552.4 | 35525 | 96.4 |
| 32 | 518.1 | 566.1 | 39689 | 96.8 |

Interpretation:

- `30` beats `32` on `time/fps` and is close enough on env-steps/sec that it is
  the better raw-throughput choice.
- `32` is already on the wrong side of the memory-pressure cliff.
- `14` and `16` give much more headroom and are safer defaults when long-run
  stability matters more than absolute throughput.

Measured full 141-episode policy-analysis results for the best saved checkpoint
from the benchmark runs:

| candidate | success | collision |
| --- | ---: | ---: |
| promoted PPO v3 baseline | 0.8511 | 0.1489 |
| `num_envs=8` best checkpoint | 0.8652 | 0.1348 |
| `num_envs=14` best checkpoint | 0.8582 | 0.1418 |
| `num_envs=30` best checkpoint | 0.8582 | 0.1418 |

Interpretation:

- The benchmark does not support a single universal "best `num_envs`" answer.
- `30` is the best host-throughput setting on this machine.
- `8` is currently the strongest evaluated model candidate among the tested
  benchmark runs.
- `14` is the best compromise if you want materially higher throughput than `8`
  without running into the memory pressure seen at `30`/`32`.

## Recommendation

Use these defaults on `imech156-u`:

- throughput benchmarking / short exploratory runs: `num_envs=30`
- long production PPO runs with headroom: `num_envs=14`
- quality-sensitive reproduction of the strongest 1M-step candidate from this
  benchmark: `num_envs=8`

Do not default to `num_envs=32` on this host. It is slightly faster than `30`
on env-steps/sec, but worse on `time/fps`, pushes memory usage to the same
ceiling, and produced worse early policy quality.

## Auto modes

PPO now supports two host-aware auto sizing modes:

- `num_envs: auto_throughput`
  - default behavior
  - aims for the highest practical env count while still leaving a small
    learner/OS margin
- `num_envs: auto_stable`
  - conservative mode
  - leaves materially more memory headroom for long runs and background system
    overhead

`num_envs: auto` resolves to `auto_throughput`.

Important:

- use the auto modes for host-adaptive runs,
- keep explicit numeric `num_envs` in benchmark configs when exact
  reproducibility matters.

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
