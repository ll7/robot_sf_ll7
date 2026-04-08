# Issues 578, 608, and 609: DreamerV3 BR-08 parity note

Date: 2026-04-08

## Consolidated goal

Issues #578, #608, and #609 are best treated as one DreamerV3 BR-08 parity slice:

- #609 provides the training-side scenario-matrix surface.
- #608 provides the evaluation-side scenario-matrix surface.
- #578 remains the umbrella for deciding whether the resulting DreamerV3 challenger path
  is credible enough to spend larger Slurm training budget.

The scope is deliberately infrastructure-first. It does not claim that DreamerV3 now trains
well on Robot SF, and it does not promote a checkpoint. It removes a major comparability gap:
the previous launcher was easier to run, but it did not expose the same scenario-matrix
curriculum/evaluation surface that the PPO benchmark path uses.

## Implementation decision

The branch implements a config-first RLlib launcher path rather than a custom DreamerV3 fork.
That is the conservative option because RLlib already owns the DreamerV3 world-model, replay,
actor, critic, learner, and EnvRunner behavior. The repo-owned work now focuses on the parts
that are Robot SF-specific:

- scenario-matrix sampling with deterministic worker seeds,
- scenario switching between episodes with observation/action-space compatibility checks,
- recursive Dict observation flattening so nested `socnav_struct` + occupancy-grid observations
  can be consumed by RLlib DreamerV3,
- config-first BR-08 gate/full profiles, and
- periodic cycle-order evaluation summaries written into the Dreamer run directory.

This follows the DreamerV3 architecture at the launcher boundary: DreamerV3 learns a world
model from replayed real interactions and trains actor/critic components on imagined rollouts.
The RLlib documentation also treats DreamerV3 as a model-based algorithm that scales through
EnvRunner actors and learner resources, so this branch keeps sampling/evaluation parity in
the launcher instead of replacing RLlib internals.

Primary references consulted:

- DreamerV3 implementation/project reference: https://github.com/danijar/dreamerv3
- RLlib DreamerV3 algorithm docs: https://docs.ray.io/en/master/rllib/rllib-algorithms.html#dreamerv3
- RLlib EnvRunner scaling docs: https://docs.ray.io/en/master/rllib/rllib-env.html#performance-and-scaling

## New config surfaces

- `configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_gate.yaml`
  - short validation profile,
  - offline W&B,
  - `socnav_struct` observation mode,
  - occupancy grid in observation,
  - scenario-matrix random sampling per reset,
  - periodic evaluation configured but disabled by default.
- `configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_full.yaml`
  - long Slurm/Auxme profile,
  - online W&B,
  - auto CPU/GPU resource resolution,
  - scenario-matrix random sampling per reset,
  - periodic cycle-order evaluation enabled every 100 training iterations.
- `docs/training/dreamerv3_br08_slurm_handoff.md`
  - gate-first Slurm handoff,
  - monitoring commands,
  - artifact locations,
  - stop conditions and post-run decision path.

Canonical gate command:

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_gate.yaml
```

Canonical full command:

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_full.yaml
```

## Validation on this branch

Executed:

```bash
scripts/dev/ruff_fix_format.sh
```

Result: all checks passed.

Executed:

```bash
uv run python -m py_compile scripts/training/train_dreamerv3_rllib.py \
  robot_sf/training/rllib_env_wrappers.py robot_sf/training/scenario_sampling.py
```

Executed:

```bash
uv run pytest tests/training/test_rllib_env_wrappers.py \
  tests/training/test_scenario_sampling.py \
  tests/training/test_train_dreamerv3_rllib_config.py \
  tests/training/test_train_dreamerv3_rllib_runtime.py -q
```

Result: 39 passed.

Executed:

```bash
uv run --extra rllib python - <<'PY'
from pathlib import Path
from scripts.training.train_dreamerv3_rllib import load_run_config, _make_env_creator

config = load_run_config(Path("configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_gate.yaml"))
env = _make_env_creator(config)({"worker_index": 0})
try:
    obs, _ = env.reset(seed=config.experiment.seed)
    print(type(env).__name__)
    print(env.observation_space)
    print(env.action_space)
    print(type(obs).__name__, getattr(obs, "shape", None))
finally:
    env.close()
PY
```

Result: exit code 0; constructed the wrapped gate env with a flattened float32 observation vector
of shape `(77088,)` and normalized action space `Box(-1.0, 1.0, (2,), float32)`.

Executed:

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_gate.yaml \
  --dry-run
```

Result: exit code 0.

Executed:

```bash
uv run --extra rllib python scripts/training/train_dreamerv3_rllib.py \
  --config configs/training/rllib_dreamerv3/benchmark_socnav_grid_br08_full.yaml \
  --dry-run
```

Result: exit code 0.

Executed:

```bash
uv run python scripts/dev/check_skills.py
```

Result: validated 28 skills and README coverage.

Executed:

```bash
uv run pytest tests/test_issue_workflow_batching.py -q
```

Result: 1 passed.

Executed:

```bash
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

Result: 2752 passed, 14 skipped, 1 warning; Ruff and readiness checks passed.

## Remaining risk and next issues

The main technical risk is no longer only launcher parity. The likely remaining limits are:

- Dreamer may still be sample-inefficient for this sparse, safety-heavy navigation reward.
- The flattened SocNav+grid vector may be a weak representation for Dreamer compared with
  architectures that preserve spatial structure.
- Periodic evaluation is lightweight and in-process; checkpoint promotion still needs the
  external benchmark/policy-analysis gate before making any paper-facing claim.
- Encoder/decoder or world-model pretraining from rollouts generated by PPO, ORCA, or other
  policies is not part of #578/#608/#609. It is plausible as a future research direction, but
  it should be opened as a dedicated design issue because it requires deciding whether to
  extend RLlib DreamerV3, export/import world-model weights, or train a separate representation
  model outside RLlib. Follow-up issue: #782.

Suggested immediate next step: run the BR-08 SocNav+grid gate profile first and inspect W&B plus
the generated periodic evaluation summaries. If the gate run does not show at least improving
success and decreasing collision/timeout trends, avoid launching the full Slurm run until there
is a new hypothesis.
