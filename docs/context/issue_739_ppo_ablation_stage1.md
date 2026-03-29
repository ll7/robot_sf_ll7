# Issue 739: PPO Reward and Observation Ablation Stage 1

## Goal

Run a bounded first-pass ablation against the issue-708 PPO setup instead of continuing opaque
reward and observation tweaking.

This stage is intentionally smaller than the full 30M-step issue-708 retrain:

- short local training runs,
- deterministic full maintained-surface eval,
- conservative interpretation,
- enough evidence to choose the next ablation direction.

## Matrix

Canonical stage-1 configs:

- `configs/training/ppo/ablations/expert_ppo_issue_739_stage1_baseline.yaml`
  - short-run control
  - current issue-708 reward and observation stack
- `configs/training/ppo/ablations/expert_ppo_issue_739_stage1_reward_core.yaml`
  - same observation stack
  - reward reduced to progress, collision, timeout, terminal bonus
- `configs/training/ppo/ablations/expert_ppo_issue_739_stage1_obs_grid_goal.yaml`
  - same reward stack
  - grid extractor plus ego-goal vector only for the MLP branch
- `configs/training/ppo/ablations/expert_ppo_issue_739_stage1_obs_selective.yaml`
  - same reward stack
  - keep ego dynamics plus compact predictive summary features
  - drop high-dimensional pedestrian arrays

## Evaluation Contract

- training surface:
  - `configs/scenarios/classic_interactions_francis2023.yaml`
- eval surface:
  - `configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`
- eval cadence:
  - one deterministic evaluation at the end of the 8,192-step screening run
- fixed eval seed policy:
  - one fixed seed block for faster turnaround in stage 1

## Runner

```bash
uv run python scripts/validation/run_issue_739_stage1_ablations.py
```

Optional single-config execution:

```bash
uv run python scripts/validation/run_issue_739_stage1_ablations.py \
  --config configs/training/ppo/ablations/expert_ppo_issue_739_stage1_reward_core.yaml
```

## Result Recording

Write the final stage-1 summary to:

- `output/validation/issue_739_stage1_ablations/<run>/summary.json`
- `output/validation/issue_739_stage1_ablations/<run>/summary.md`

The recommendation for stage 2 should answer:

- did reward simplification help enough to justify a longer run?
- did observation simplification help or hurt?
- should predictive foresight stay enabled in the next comparison wave?

## Early Result: Iterations 3-4

Artifacts:

- `output/validation/issue_739_stage1_ablations/iter3/summary.json`
- `output/validation/issue_739_stage1_ablations/iter3/summary.md`
- `output/validation/issue_739_stage1_ablations/iter4_obs_only/summary.json`
- `output/validation/issue_739_stage1_ablations/iter4_obs_only/summary.md`
- `output/validation/issue_739_stage1_ablations/iter5_obs_selective/summary.json`
- `output/validation/issue_739_stage1_ablations/iter5_obs_selective/summary.md`
- `output/validation/issue_739_stage1_ablations/iter6_reward_tuned/summary.json`
- `output/validation/issue_739_stage1_ablations/iter6_reward_tuned/summary.md`

Observed screening result after `8,192` training steps and one deterministic full-surface eval:

| Variant | Success | Collision | SNQI | Eval return |
| --- | ---: | ---: | ---: | ---: |
| baseline | `0.1571` | `0.8429` | `-2.0380` | `-2.6803` |
| reward_core | `0.0571` | `0.8857` | `-2.3080` | `-13.2681` |
| obs_grid_goal | `0.1143` | `0.8143` | `-2.1082` | `-7.4426` |
| obs_selective | `0.0143` | `0.9000` | `-2.3900` | `-22.3000` |
| reward_tuned | `0.0143` | `0.9000` | `-2.3200` | `-16.1000` |

Interpretation:

- Removing the dense shaping terms did **not** help in this early-screening regime.
- The `reward_core` variant is worse on every primary eval metric than the baseline.
- The observation-lite variant reduced collisions relative to baseline, but it also reduced success
  and did not improve SNQI or return.
- The selective observation reduction is much worse than baseline on every primary metric.
- The narrower reward retuning also failed the screening gate and landed in the same low-success
  regime as the weakest observation variants.
- That does **not** prove the current issue-708 stack is globally optimal, but it does show that
  neither blunt reward pruning, the tested observation-pruning variants, nor this first reward
  retuning pass are easy wins.

## Stage-2 Recommendation

The next high-value ablation should be narrower and should keep the broad observation surface:

- keep the current reward stack and current observation stack as the control,
- keep predictive foresight enabled,
- stop broad reward changes for now,
- if observation work continues, test additive feature normalization or weighting before further
  feature removal,
- shift the next hypothesis toward scenario sampling / curriculum or optimizer-scale changes rather
  than more feature or reward removal.
