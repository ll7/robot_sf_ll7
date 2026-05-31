# Issue 1752 Decision Transformer Dataset Preflight

Related issue: [#1752](https://github.com/ll7/robot_sf_ll7/issues/1752)
Parent feasibility PR: [#1750](https://github.com/ll7/robot_sf_ll7/pull/1750)
Fallback policy: [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md)

## Decision

The expert trajectory collector now emits a Decision-Transformer-ready preflight schema while
remaining backwards-compatible for existing BC consumers that read only observations/actions.

Schema marker:

```text
trajectory_dataset.v2.decision_transformer_preflight
```

The collector still does not train a Decision Transformer. It only proves that Robot SF can produce
and validate the missing offline sequence labels that issue #1622 identified as a blocker.

## Dataset Fields

`scripts/training/collect_expert_trajectories.py` now writes these NPZ arrays:

- `positions`
- `actions`
- `observations`
- `rewards`
- `terminated`
- `truncated`
- `return_to_go`
- `episode_ids`
- `scenario_ids`
- `seeds`

Reward convention: `environment_step_reward`.
Return convention: `undiscounted_future_return_to_go`.

The return-to-go value at step `t` is the undiscounted sum of rewards from `t` through the end of
that recorded episode. The terminal flags are stored separately so a future offline-RL loader can
distinguish environment termination from truncation.

## Manifest Fields

Trajectory dataset manifests now receive additive metadata:

- `dataset_schema`
- `trajectory_fields`
- `splits`
- `reward_convention`
- `return_convention`
- `status_policy`
- `dataset_sha256`
- `durable_artifact_uri_policy`

The `splits.train` section records episode IDs, scenario IDs, and seeds. The issue #1752 slice uses
a single `train` split because it is a preflight producer; fair BC/PPO/oracle comparison splits are
still future experiment-design work.

The status policy explicitly lists excluded non-success statuses:

- `readiness_status`: `fallback`, `degraded`
- `availability_status`: `not_available`

Rows with those statuses must be excluded or explicitly labeled before any future training use.

## Validator

`robot_sf.benchmark.validation.trajectory_dataset.TrajectoryDatasetValidator` now has an explicit
Decision Transformer preflight mode. Legacy BC-style NPZ datasets still validate against the base
`positions` / `actions` / `observations` schema unless DT mode is requested or the dataset metadata
declares `trajectory_dataset.v2.decision_transformer_preflight`. In DT mode, the validator requires
reward, terminal, truncation, and return-to-go arrays. It also quarantines datasets when per-episode
trajectory lengths are misaligned or when fallback/degraded/not-available status rows are present
without an explicit status policy.

The new CLI is:

```bash
uv run python scripts/validation/validate_trajectory_dataset.py \
  --dataset-id dt1752_dry \
  --min-episodes 1 \
  --fail-on-quarantine
```

For ad-hoc legacy NPZ files that do not yet declare the schema, add
`--require-decision-transformer-fields` to force the stricter preflight.

## Proof

Focused tests:

```bash
uv run pytest \
  tests/training/test_collect_expert_trajectories_dt_preflight.py \
  tests/integration/test_expert_trajectory_dataset.py \
  tests/test_benchmark_imitation_manifest.py \
  tests/validation/test_validate_trajectory_dataset_cli.py \
  -q
```

Lint:

```bash
uv run ruff check \
  scripts/training/collect_expert_trajectories.py \
  scripts/validation/validate_trajectory_dataset.py \
  robot_sf/benchmark/validation/trajectory_dataset.py \
  tests/training/test_collect_expert_trajectories_dt_preflight.py \
  tests/integration/test_expert_trajectory_dataset.py \
  tests/test_benchmark_imitation_manifest.py \
  tests/validation/test_validate_trajectory_dataset_cli.py
```

Tiny executable preflight:

```bash
LOGURU_LEVEL=WARNING uv run python scripts/training/collect_expert_trajectories.py \
  --dataset-id dt1752_dry \
  --policy-id ppo_expert_v1 \
  --episodes 1 \
  --dry-run \
  --scenario-config configs/scenarios/classic_interactions.yaml \
  --seeds 1752
```

Observed result: the generated manifest under `output/benchmarks/expert_trajectories/` recorded
`quality_status=validated`, one episode, split seed `1752`, non-empty `dataset_sha256`, and the
schema/status-policy metadata above. The generated `output/` files are local proof artifacts only,
not durable training dependencies.
