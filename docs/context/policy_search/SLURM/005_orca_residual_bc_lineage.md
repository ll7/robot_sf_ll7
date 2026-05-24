# ORCA-Residual Behavior-Cloning Lineage Handoff

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1428>

## Boundary

This handoff stages the first ORCA-residual learned-policy lineage slice. It does not submit Slurm,
train the residual policy, or count the existing `orca_residual_guarded_ppo_v0` smoke as learned
residual evidence.

The local preflight packet is:

```bash
uv run python scripts/validation/validate_orca_residual_lineage_packet.py \
  --config configs/training/orca_residual/orca_residual_bc_issue_1428.yaml --json
```

## Contract

- Objective: behavior-cloning residual, predicting bounded `policy_action - ORCA_action`.
- Observation surface: current runtime `socnav_struct` state only.
- Forbidden inputs: scenario-future features, benchmark labels, future oracle trajectories, and
  privileged map-solution features.
- Residual bounds: linear `0.25`, angular `0.35`, matching
  `configs/policy_search/candidates/orca_residual_guarded_ppo_v0.yaml`.
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
  --dataset-id issue_1428_orca_residual_bc_dataset --episodes <bounded-smoke-episodes> \
  --training-config configs/training/orca_residual/orca_residual_bc_issue_1428.yaml
```

Behavior-cloning training:

```bash
uv run python scripts/training/pretrain_from_expert.py \
  --config configs/training/orca_residual/orca_residual_bc_issue_1428.yaml
```

Smoke evaluation after a concrete checkpoint pointer exists:

```bash
LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 uv run python \
  scripts/validation/run_policy_search_candidate.py --candidate orca_residual_guarded_ppo_v0 \
  --stage smoke --workers 1
```

## Gate

Only submit the bounded nominal Slurm job after the local packet validates and the dataset/checkpoint
artifact pointers are concrete. If the smoke or nominal run reports fallback/degraded status, revise
the lineage path instead of escalating to stress/full evaluation.
