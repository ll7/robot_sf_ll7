# ORCA-Residual Behavior-Cloning Lineage Handoff

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1428>

## Boundary

This handoff stages the first ORCA-residual learned-policy lineage slice. The concrete issue #1475
smoke job is now prepared as a bounded Slurm wrapper, but this handoff still does not count the
existing `orca_residual_guarded_ppo_v0` runtime smoke as learned residual evidence.

Issue #2390 revises the rerun target to `orca_residual_guarded_ppo_progress_v1` after job `12749`
timed out with low progress and no guard saturation. The revision is still a smoke-only
launch-packet candidate until a bounded rerun produces durable artifacts.

The local preflight packet is:

```bash
uv run python scripts/validation/validate_orca_residual_lineage_packet.py \
  --config configs/training/orca_residual/orca_residual_bc_issue_1428.yaml --json
```

## Contract

- Objective: progress-probe behavior-cloning residual, predicting bounded
  `policy_action - ORCA_action` under the guarded ORCA runtime contract.
- Observation surface: current runtime `socnav_struct` state only.
- Forbidden inputs: scenario-future features, benchmark labels, future oracle trajectories, and
  privileged map-solution features.
- Residual bounds: linear `0.35`, angular `0.35`, matching
  `configs/policy_search/candidates/orca_residual_guarded_ppo_progress_v1.yaml`.
- Hard guard: authoritative in all evaluation rows.

## Required Outputs

The completion update for the bounded Slurm follow-up must include:

- residual dataset manifest,
- candidate YAML,
- checkpoint pointer,
- diagnostic report path,
- residual contribution and clipping diagnostics,
- guard veto and fallback/degraded status.

Fallback or degraded rows must not count as successful learned-policy evidence.

## Command Shapes

Dataset collection:

```bash
uv run python scripts/training/collect_expert_trajectories.py --policy-id \
  ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417 \
  --dataset-id issue_1428_orca_residual_bc_progress_v1_smoke --episodes 3 \
  --scenario-config configs/scenarios/single/planner_sanity_simple.yaml \
  --training-config configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity.yaml \
  --env-config configs/training/orca_residual/orca_residual_bc_issue_1475_smoke_pretrain.yaml \
  --seeds 111 112 113
```

Behavior-cloning training:

```bash
uv run python scripts/training/pretrain_from_expert.py \
  --config configs/training/orca_residual/orca_residual_bc_issue_1475_smoke_pretrain.yaml
```

Smoke evaluation after a concrete checkpoint pointer exists:

```bash
LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 uv run python \
  scripts/validation/run_policy_search_candidate.py --candidate orca_residual_guarded_ppo_progress_v1 \
  --stage smoke --workers 1
```

## Gate

Only submit the bounded nominal Slurm job after the local packet validates and the dataset/checkpoint
artifact pointers are concrete. If the smoke or nominal run reports fallback/degraded status, revise
the lineage path instead of escalating to stress/full evaluation.

## Issue #1475 Slurm Wrapper

Dry-run the prepared smoke job from the owning worktree:

```bash
scripts/dev/sbatch_orca_residual_bc_issue1475.sh --dry-run --no-status
```

Submit the smoke-only job from an allowed Auxme Slurm login node:

```bash
scripts/dev/sbatch_orca_residual_bc_issue1475.sh
```

The wrapper validates this lineage packet, collects three smoke episodes from the issue-791 PPO
leader, trains `issue_1428_orca_residual_bc_progress_v1_policy_smoke`, materializes a run-local
candidate registry that points at that checkpoint, and evaluates smoke before any optional nominal run. Set
`--run-nominal` only when the smoke gate should immediately escalate to `nominal_sanity`; the
wrapper fails closed if smoke success is below `1.0` or collision rate is above `0.0`.
