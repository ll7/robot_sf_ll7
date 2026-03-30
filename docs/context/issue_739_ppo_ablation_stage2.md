# Issue 739: PPO Reward and Observation Ablation Stage 2

## Goal

Execute the stage-2 ablation plan recommended by stage-1 results.

Stage-1 showed that blunt reward pruning and observation removal were not easy wins.
The stage-2 plan shifts focus to optimizer-scale and scenario-sampling variants while
keeping the full observation stack.

## Matrix

Canonical stage-2 configs:

- `configs/training/ppo/ablations/expert_ppo_issue_739_stage2_opt_scale.yaml`
  - scaled batch size (512 vs 256) and epochs (8 vs 4)
  - doubled learning rate (1.5e-4 vs 7.5e-5)
  - target KL relaxed to 0.03
  - purpose: reduce gradient noise in early training
- `configs/training/ppo/ablations/expert_ppo_issue_739_stage2_sampling.yaml`
  - cycle scenario sampling instead of random
  - cycles through scenarios in order for equal exposure
  - purpose: simplify early learning curriculum by giving each scenario equal time

Both variants keep:
- full issue-708 observation stack (grid + socnav_struct)
- predictive foresight enabled
- route_completion_v3 reward with issue-708 weights
- same 8,192-step screening length for fast turnaround

## Evaluation Contract

- training surface:
  - `configs/scenarios/classic_interactions_francis2023.yaml`
- eval surface:
  - `configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`
- eval cadence:
  - one deterministic evaluation at the end of the screening run
- fixed eval seed policy:
  - one fixed seed block for faster turnaround

## Runner

```bash
uv run python scripts/validation/run_issue_739_stage2_ablations.py
```

Optional single-config execution:

```bash
uv run python scripts/validation/run_issue_739_stage2_ablations.py \
  --config configs/training/ppo/ablations/expert_ppo_issue_739_stage2_opt_scale.yaml
```

## Results

### Comparison with Stage 1 Baseline

| Variant | Success | Collision | SNQI | Eval Return |
| --- | ---: | ---: | ---: | ---: |
| **stage1_baseline** | **0.1571** | **0.8429** | **-2.0380** | **-2.6803** |
| stage2_opt_scale | 0.0143 | 0.8429 | -2.4159 | -24.5830 |
| stage2_sampling | 0.0571 | 0.8714 | -2.2327 | -15.3726 |

### Interpretation

- **Optimizer scaling (opt_scale)**: Scaling batch size, epochs, and learning rate did **not** help.
  - Success rate dropped from 15.7% to 1.4% (11x worse)
  - SNQI degraded from -2.04 to -2.42
  - Eval return collapsed from -2.68 to -24.58
  - The higher learning rate (1.5e-4) and larger batch (512) introduced instability

- **Cycle sampling (sampling)**: Cycling through scenarios in order was **not better** than random.
  - Success rate dropped from 15.7% to 5.7% (2.7x worse)
  - SNQI degraded from -2.04 to -2.23
  - Collision rate slightly increased (0.84 to 0.87)
  - The cycle strategy exposed the policy to more scenarios but didn't improve learning

### Stage 2 vs Stage 1 Comparison

Neither Stage 2 hypothesis improved over the Stage 1 baseline:

| Improvement Area | Tested | Result |
| --- | --- | --- |
| Reward simplification | Stage 1 | No - reward_core and reward_tuned failed |
| Observation simplification | Stage 1 | No - obs_grid_goal and obs_selective failed |
| Optimizer scaling | Stage 2 | No - opt_scale failed |
| Scenario sampling | Stage 2 | No - cycle sampling failed |

## Final Recommendation

After two stages of ablations (5 variants in Stage 1, 2 variants in Stage 2), **none of the tested simplifications or modifications improved over the baseline issue-708 configuration**.

**Recommendation for next steps:**

1. **Return to the baseline** (`expert_ppo_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch.yaml`) for any full 30M-step retrain
2. **The current stack is not obviously broken** - the complexity may be necessary for this task
3. **Future ablation directions** (if continued):
   - Feature normalization/standardization before the MLP
   - Different CNN architectures (smaller grids, different channel structures)
   - Predictive foresight ablation (test without foresight once #738 is resolved)
   - Different curriculum strategies (progressive difficulty rather than random/cycle)

**Do not proceed with:**
- Further reward term removal
- Further observation simplification
- Aggressive optimizer scaling
- Simple scenario cycling

## Artifacts

Results recorded:
- `output/validation/issue_739_stage2_ablations/latest/summary.json`
- `output/validation/issue_739_stage2_ablations/latest/summary.md`

Checkpoints:
- `output/benchmarks/expert_policies/checkpoints/ppo_expert_issue_739_stage2_opt_scale/`
- `output/benchmarks/expert_policies/checkpoints/ppo_expert_issue_739_stage2_sampling/`

## Definition of Done Status

| Requirement | Status |
| --- | --- |
| Bounded ablation matrix defined | ✅ 7 configs across Stage 1 & 2 |
| Simplified variants trained vs baseline | ✅ All variants completed |
| Results with conservative interpretation | ✅ Both stages documented |
| Recommendation on simplification | ✅ Return to baseline |

**Conclusion: Issue 739 is complete. The baseline configuration remains the best tested option.**
