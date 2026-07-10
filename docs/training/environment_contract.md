# Robot SF Environment Contract And Training Provenance

This note is the reviewer-facing contract for Robot SF training and benchmark work. It summarizes
where environment behavior belongs, who owns rollouts, how training rewards differ from benchmark
outcomes, and which run-record fields must be preserved for PPO/RL provenance.

Design source: [Issue #1037 RL Environment Patterns](../context/issue_1037_rl_environment_patterns.md).

## Public Environment Contract

Create environments through `robot_sf/gym_env/environment_factory.py`:

- `make_robot_env(...)` for standard robot navigation observations,
- `make_image_robot_env(...)` for image observations,
- `make_pedestrian_env(...)` for pedestrian-policy training against a robot model.

Do not instantiate concrete environment classes directly in training, benchmark, examples, or new
workflow code. The factory layer owns:

- config normalization and legacy-kwargs compatibility,
- deterministic seeding when `seed` is provided,
- reward profile/curriculum construction from `reward_name`, `reward_func`, and
  `reward_curriculum`,
- recording and render option precedence,
- public metadata such as suite, scenario, algorithm, and recording seed.

Environment `reset()` and `step()` follow Gymnasium semantics:

- `reset()` initializes one scenario instance and returns the observation plus info metadata.
- `step(action)` advances physics and returns `(observation, reward, terminated, truncated, info)`.
- `terminated` is for environment outcomes such as route completion or collision.
- `truncated` is for horizon or externally imposed cutoff.
- `info` should carry diagnostic metadata needed by trainers, wrappers, and benchmark harnesses
  without changing the observation contract.

Environment code should contain simulation state transitions, observation construction, termination
state, and dense training reward calculation. Trainer code should contain optimization, checkpoint
cadence, vectorization, and trainer-specific logging. Benchmark code should contain evaluation
episode ownership, schema validation, planner runtime metadata, and benchmark metric aggregation.

## Rollout Ownership

Robot SF has two different rollout owners:

| Workflow | Rollout owner | Primary path | Output contract |
|---|---|---|---|
| PPO/SB3 training | Stable-Baselines3 through vectorized envs | `scripts/training/train_ppo.py` | checkpoints, eval timelines, training run manifests |
| RLlib training | RLlib algorithm workers | `scripts/training/train_dreamerv3_rllib.py` | RLlib checkpoints and run diagnostics |
| Benchmark evaluation | Robot SF benchmark harness | `robot_sf/benchmark/runner.py`, `robot_sf/benchmark/cli.py` | schema-checked episode JSONL and aggregate reports |

Training loops may use dense rewards from `robot_sf/gym_env/reward.py`. Benchmark runs must use
benchmark metrics and outcome fields from the benchmark episode schema. A high training reward is
not benchmark success evidence.

## Reward Versus Benchmark Outcome

Training reward is an optimization signal. It may be dense, curriculum-driven, shaped, or tuned for
sample efficiency. In PPO runs it is resolved from `env_factory_kwargs` by
`scripts/training/train_ppo.py`; the default profile is `route_completion_v2` unless a reward name,
custom callable, or curriculum is supplied. For the full per-term weight table (all named profiles
and legacy reward functions), the collision-penalty FAQ, and the checkpoint-profile recipe, see
[Reward Profiles Reference](./reward_profiles.md).

Benchmark success is an evaluation contract. Benchmark claims must rely on:

- schema-checked episode records,
- explicit termination and outcome fields,
- component metrics such as success, collisions, near misses, time, comfort, force, and jerk,
- SNQI only as a versioned composite with fixed weights/baselines,
- execution mode and readiness metadata (`native`, `adapter`, `fallback`, `degraded`, etc.).

Fallback or degraded execution is not successful benchmark evidence. Use the canonical
[benchmark fallback policy](../context/issue_691_benchmark_fallback_policy.md) when deciding whether
a planner row can support a benchmark claim.

Do not use LLM-as-judge, subjective text scoring, or training reward totals as benchmark-success
criteria. Those signals can be useful diagnostics in separate research workflows, but they do not
replace deterministic schema-backed benchmark outcomes.

## PPO Run-Record Checklist

For every reviewer-facing PPO/RL training record, preserve the fields below. Required fields should
be present in the config, startup summary, manifest, PR text, or context note. Optional local notes
help debugging but are not durable provenance unless promoted to a tracked note or manifest.

| Field | Required? | Where to check | Why it matters |
|---|---|---|---|
| Training command and config path | required | `uv run python scripts/training/train_ppo.py --config ...` | Replays the same workflow |
| `policy_id` | required | training YAML, startup summary, expert manifest | Identifies checkpoints and registry entries |
| Training scenario config | required | `scenario_config`, `input_artefacts` | Defines the task source |
| Evaluation scenario config | required | `evaluation.scenario_config` or fallback to `scenario_config` | Separates train and hold-out surfaces |
| Seed policy | required | `seeds`, `randomize_seeds`, `evaluation.randomize_seeds` | Bounds reproducibility and variance |
| Reward profile or curriculum | required | `env_factory_kwargs.reward_name`, `reward_func`, `reward_curriculum` | Explains the optimization signal |
| Environment factory kwargs | required | training YAML, startup summary when summarized | Captures public env behavior toggles |
| Vectorization mode and env count | required | `num_envs`, resolved worker mode startup log | Affects throughput and stochastic exposure |
| Evaluation cadence | required | `evaluation.step_schedule` | Defines checkpoint/eval timing |
| Git/config hashes or copied config manifest | required | training manifest, checkpoint config copy, PR/context note | Makes artifacts auditable |
| Checkpoint path and config manifest | required | expert manifest | Rehydrates the trained policy |
| Episode log, eval timeline, per-scenario eval, perf summary | required when generated | training run manifest | Supports metric review beyond headline reward |
| Artifact destination and durability decision | required for handoff | PR body or context note | Prevents hidden `output/` dependencies |
| Local machine/resource notes | optional | `local.machine.md`, context note | Explains local throughput constraints |
| W&B or external artifact pointer | optional unless relied on | model registry or PR/context note | Makes non-local artifacts recoverable |

If a required field is absent, do not silently infer it from chat history. Add it to the run note,
manifest, or follow-up issue before using the run as evidence.

## Validation Commands

Use these lightweight checks when updating this contract or reviewing referenced paths:

```bash
test -f robot_sf/gym_env/environment_factory.py
test -f robot_sf/gym_env/reward.py
test -f scripts/training/train_ppo.py
test -f robot_sf/benchmark/schemas/episode.schema.v1.json
test -f docs/context/issue_691_benchmark_fallback_policy.md
```

For a runnable PPO provenance smoke path:

```bash
uv run python scripts/training/train_ppo.py \
  --config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml \
  --dry-run \
  --log-level WARNING
```
