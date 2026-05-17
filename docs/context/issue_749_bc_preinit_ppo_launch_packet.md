# Issue 749 BC-Preinitialized PPO Launch Packet

Date: 2026-05-09

Related issue: `ll7/robot_sf_ll7#749`

## Goal

Prepare the reproducible launch packet for the BC-preinitialized PPO challenger under the v10
fine-tune contract, while making the current evidence gap explicit.

Issue #749 asks for a full experiment chain:

1. collect or validate expert trajectories from the promoted BR-06 v3 expert,
2. run behavioral-cloning pretraining,
3. fine-tune PPO under the maintained v10 contract,
4. compare the result against `27dbe5xu` and `b60iopxt`,
5. document whether the result improves sample efficiency or benchmark strength.

## Current Evidence State

No tracked #749 run artifact is available in this checkout. The existing parent campaign note says
no matching #749 W&B run URL was found by issue tag, group, or run name:

* `docs/context/issue_708_main_based_ppo_retrain_campaign.md`

The repository does have the necessary pipeline entry points and comparator references:

* `scripts/training/collect_expert_trajectories.py`
* `scripts/training/pretrain_from_expert.py`
* `scripts/training/train_ppo_with_pretrained_policy.py`
* `docs/imitation_learning_pipeline.md`
* `configs/baselines/ppo_v10_carry_forward_27dbe5xu.yaml`
* `model/registry.yaml`

The new issue-specific configs are:

* `configs/training/ppo_imitation/bc_pretrain_issue_749_v10_warm_start.yaml`
* `configs/training/ppo_imitation/ppo_finetune_issue_749_v10_warm_start.yaml`

## Canonical Command Path

Install the optional imitation stack before the BC stage:

```bash
uv sync --group imitation
```

Collect the expert trajectory dataset from the promoted BR-06 v3 expert policy:

```bash
uv run python scripts/training/collect_expert_trajectories.py \
  --dataset-id issue_749_b60iopxt_v10_eval_trajectories \
  --policy-id ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200 \
  --episodes 141 \
  --scenario-config configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml \
  --training-config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml \
  --env-config configs/training/ppo_imitation/ppo_finetune_issue_749_v10_warm_start.yaml \
  --seeds 111 112 113
```

The `--training-config` flag records the source expert provenance, while `--env-config` applies
the issue-specific v10 warm-start env contract and factory kwargs. Collection persists the
checkpoint-compatible observation contract for BC and PPO fine-tuning.

Run BC pretraining:

```bash
uv run --group imitation python scripts/training/pretrain_from_expert.py \
  --config configs/training/ppo_imitation/bc_pretrain_issue_749_v10_warm_start.yaml
```

Run PPO fine-tuning:

```bash
uv run python scripts/training/train_ppo_with_pretrained_policy.py \
  --config configs/training/ppo_imitation/ppo_finetune_issue_749_v10_warm_start.yaml
```

Compare the finished challenger against the carry-forward candidate and historical expert:

```bash
uv run python scripts/tools/policy_analysis_run.py \
  --policy ppo \
  --model-path output/benchmarks/expert_policies/issue_749_ppo_finetune_v10_warm_start_finetuned.zip \
  --seed-set eval \
  --max-seeds 3 \
  --output output/benchmarks/issue749_policy_analysis \
  --video-output output/recordings/issue749_policy_analysis \
  --all
```

Use `configs/baselines/ppo_v10_carry_forward_27dbe5xu.yaml` as the direct `27dbe5xu` comparator
surface. Use the `b60iopxt` registry entry in `model/registry.yaml` as the historical promoted v3
expert source.

## Artifact Boundary

The full #749 experiment is not complete until these durable artifacts exist:

* trajectory dataset NPZ and manifest for `issue_749_b60iopxt_v10_eval_trajectories`,
* BC checkpoint and manifest for `issue_749_bc_preinit_v10_policy`,
* PPO fine-tuned checkpoint and manifest for `issue_749_ppo_finetune_v10_warm_start`,
* policy-analysis comparison against `27dbe5xu` and `b60iopxt`,
* W&B or equivalent durable artifact pointers for any checkpoint or dataset used in paper-facing
  claims.

Local files under `output/` are cache/history until paired with a durable pointer.

## Validation In This PR

This PR validates the launch packet, not the long-running experiment outcome.

Dry-run commands:

```bash
uv run python scripts/training/collect_expert_trajectories.py \
  --dataset-id issue_749_b60iopxt_v10_eval_trajectories \
  --policy-id ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200 \
  --episodes 1 \
  --scenario-config configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml \
  --training-config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml \
  --env-config configs/training/ppo_imitation/ppo_finetune_issue_749_v10_warm_start.yaml \
  --seeds 111 \
  --dry-run
uv run --group imitation python scripts/training/pretrain_from_expert.py \
  --config configs/training/ppo_imitation/bc_pretrain_issue_749_v10_warm_start.yaml \
  --dry-run
uv run python scripts/training/train_ppo_with_pretrained_policy.py \
  --config configs/training/ppo_imitation/ppo_finetune_issue_749_v10_warm_start.yaml \
  --dry-run
```

The dry run proves that the issue-specific dataset id, BC policy id, and PPO fine-tune config are
wired through the repository-native entry points. It does not produce benchmark evidence and must
not be cited as a #749 result.

## Follow-Up Boundary

The expensive experiment execution remains open in `ll7/robot_sf_ll7#1108`. That follow-up should
launch the real dataset collection, BC pretraining, PPO fine-tuning, durable artifact promotion,
and policy-analysis comparison before #749 is treated as scientifically answered.
