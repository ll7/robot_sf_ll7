# Issue 863 SVG/Model Log Spam

## Goal

Reduce repeated observability noise seen in issue-791 PPO long runs without changing map geometry,
model resolution, or benchmark semantics.

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/863>

## Evidence

- `robot_sf/nav/svg_map_parser.py` repaired invalid obstacle polygons during SVG conversion and
  logged the same warning/info pair every time the same map was reparsed.
- `robot_sf/models/registry.py` logged `Using cached model artifact` on every W&B cache hit, which
  can repeat in predictive-planner or reset-heavy workflows.
- `scripts/training/train_ppo.py` evaluates PPO checkpoints by constructing a fresh environment for
  each evaluation episode in `_evaluate_policy`; that path can reparse SVG maps during long
  evaluation phases.
- Issue-reported configs:
  - `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22.yaml`
  - `configs/training/ppo/ablations/expert_ppo_issue_791_asymmetric_critic_promotion_10m_env22.yaml`

## Decision

- Deduplicate SVG obstacle repair diagnostics once per process, SVG filename, path id, and event
  type. The first warning/info remains visible so invalid geometry is not hidden.
- Deduplicate cached-model hit logs once per process and resolved cache path. Downloads and errors
  are unchanged.
- Add compact PPO evaluation phase markers: start, completion, and progress every 10 episodes plus
  the final episode. This keeps long evaluation/reset-heavy sections distinguishable from stalls
  without replacing SB3 training progress output.

## Validation

Targeted proof:

```bash
uv run pytest tests/test_svg_obstacle_self_intersection.py tests/models/test_registry.py tests/training/test_train_expert_ppo_contract.py -q
```

Result on 2026-04-30: `32 passed`.

## Remaining Risk

The issue evidence suggests repeated environment construction may also affect throughput. This
change only addresses log volume and phase observability. Any optimization to cache parsed maps or
reuse evaluation environments should be handled separately because it can affect reset isolation and
benchmark reproducibility.
