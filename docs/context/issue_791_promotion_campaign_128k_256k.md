# Issue 791 Promotion Campaign (128k/256k)

## Objective

Run a full-length promotion campaign for the three issue-791 quality-gate improvements
(reward curriculum, asymmetric critic, attention head with zero-pedestrian hardening) at
128-step-timestep horizons (optionally scaling to 256k if results warrant deeper analysis).

Compare all three ablations against an unchanged baseline to isolate the effect of each
improvement and decide which (if any) are ready for promotion into default training pipelines.

## Campaign Structure

### Phase 1: 128k Promotion (Active)

Four simultaneous 128k runs on partition `a30` with deterministic seeds and fixed scenarios:

| Policy | Config | Job ID | WandB Group | Notes |
|--------|--------|--------|-----------|-------|
| Baseline (unchanged) | `expert_ppo_issue_791_baseline_promotion_128k.yaml` | TBD | issue-791-baseline | Reference: no reward curriculum, no asymmetric critic, no attention |
| Reward Curriculum | `expert_ppo_issue_791_reward_curriculum_promotion_128k.yaml` | 11445 | issue-791-reward-curriculum | Progressive curriculum schedule from simple to complex reward |
| Asymmetric Critic | `expert_ppo_issue_791_asymmetric_critic_promotion_128k.yaml` | 11446 | issue-791-asymmetric-critic | Separate observation spaces for actor/critic |
| Attention Head | `expert_ppo_issue_791_attention_head_promotion_128k.yaml` | 11447 | issue-791-attention-head | Masked self-attention over pedestrian slots + asymmetric critic |

### Phase 2: 256k Promotion (Optional, decision-gated)

If 128k results show convergence torque or near-equal performance across ablations,
run optional 256k deeper-exploration runs:

```bash
ISSUE791_WANDB_POLICY=require sbatch --export=ALL,ISSUE791_TRAIN_CONFIG=configs/training/ppo/ablations/expert_ppo_issue_791_baseline_promotion_256k.yaml SLURM/Auxme/issue_791_reward_curriculum.sl
ISSUE791_WANDB_POLICY=require sbatch --export=ALL,ISSUE791_TRAIN_CONFIG=configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_256k.yaml SLURM/Auxme/issue_791_reward_curriculum.sl
ISSUE791_WANDB_POLICY=require sbatch --export=ALL,ISSUE791_TRAIN_CONFIG=configs/training/ppo/ablations/expert_ppo_issue_791_asymmetric_critic_promotion_256k.yaml SLURM/Auxme/issue_791_asymmetric_critic.sl
ISSUE791_WANDB_POLICY=require sbatch --export=ALL,ISSUE791_TRAIN_CONFIG=configs/training/ppo/ablations/expert_ppo_issue_791_attention_head_promotion_256k.yaml SLURM/Auxme/issue_791_attention_head.sl
```

## Reproducibility & Comparability

- **Scenarios**: Deterministic fixed set (`classic_interactions_francis2023.yaml`); no randomization.
- **Seeds**: Single deterministic seed `123` across all runs (reproducibility constraint).
- **Environment count**: `num_envs: 2` subproc (same as stage-1/32k baseline).
- **Evaluation episodes**: 70 per checkpoint on held-out canonical eval split.
- **Intermediate eval**: Reduced from every 8k/16k to every 65536 steps to cut wall-clock cost.
  Final eval remains deterministic and benchmark-faithful.
- **WandB**: Mandatory for all promotion runs (enforced in SLURM wrappers).

## Promotion Decision Criteria

### Promotion Ready
- **Success rate** increases repeatably over baseline (significance TBD, at minimum +3-5% absolute).
- **Collision rate** does not regress (target: ≤ baseline).
- **Stability**: no crashes, NaN, or divergence during training.
- Policy converges within 128k timesteps (plateau detection).

### Conditional / Further Tuning Needed
- Mixed signals (one metric improves, another regresses).
- Performance still flat after 128k; defer 256k before declaring failure.

### Not Promotion Ready
- Regression on success rate relative to baseline.
- Crash or NaN during evaluation.
- Collision rate materially higher (>+5% absolute).

## Monitoring

### Live WandB Dashboard
- Project: `robot_sf`  
- Groups: `issue-791-baseline`, `issue-791-reward-curriculum`, `issue-791-asymmetric-critic`, `issue-791-attention-head`
- Key metrics: `eval/success_rate`, `eval/collision_rate`, `eval/episode_length`, `eval/episode_return`
- Wall-clock target: ~2–3 hours per 128k run on A30 GPU (based on prior 32k/57m runtime).

### Automated Reporting
After each run completes:
1. Extract final eval metrics from WandB.
2. Compare against baseline using effect sizes (Cohens d or similar).
3. Record decision (promote/conditional/reject) in issue-791 ledger.

## Next Steps After 128k Finishes

1. If baseline passes without issues and ablations show mixed/flat results, record no clear improvement.
2. If any ablation shows measurable improvement, run optional 256k for confirmation.
3. Create final promotion decision issue and close issue 791 with recommendations for future work
   (reward tuning, critic architecture, attention normalization, etc.).

## Relevant Docs

- [Issue 791 Attention Head Gate](issue_791_attention_head_gate.md)
- [Issue 791 Reward Curriculum Gate](issue_791_reward_curriculum_gate.md)
- [Issue 791 Asymmetric Critic Gate](issue_791_asymmetric_critic_gate.md)
- [Benchmark Fallback Policy](issue_691_benchmark_fallback_policy.md)
- [WandB Integration](../../docs/dev_guide.md#monitoring-training-with-wandb)
- [SLURM Auxme Guidance](../../docs/training/)

## Config Artifact Lineage

```
└── configs/training/ppo/ablations/
    ├── expert_ppo_issue_791_baseline_promotion_128k.yaml
    ├── expert_ppo_issue_791_baseline_promotion_256k.yaml
    ├── expert_ppo_issue_791_reward_curriculum_promotion_128k.yaml
    ├── expert_ppo_issue_791_reward_curriculum_promotion_256k.yaml
    ├── expert_ppo_issue_791_asymmetric_critic_promotion_128k.yaml
    ├── expert_ppo_issue_791_asymmetric_critic_promotion_256k.yaml
    ├── expert_ppo_issue_791_attention_head_promotion_128k.yaml
    └── expert_ppo_issue_791_attention_head_promotion_256k.yaml
```

## Evidence Tracking

Issue-791 results ledger: `output/ai/autoresearch/issue-791/results.tsv`
- Current: stage-1 (8k) and follow-up (32k) outcomes
- Will append: promotion-128k and promotion-256k outcomes with final success_rate, collision_rate, and decisions
