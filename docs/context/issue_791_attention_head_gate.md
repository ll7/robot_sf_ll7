# Issue 791 Attention Head Gate

## Goal

Implement the third and final bounded PPO improvement from issue 791: a config-first masked
multi-head self-attention head over variable-length pedestrian slots, replacing the flat-MLP
treatment of padded pedestrian arrays in the actor feature extractor.

## What Changed

- `robot_sf/feature_extractors/grid_socnav_extractor.py`
  - Added `PedestrianAttentionHead`: one layer of multi-head self-attention with masked mean
    pooling over valid pedestrian slots.
  - Hardened zero-pedestrian handling so fully masked batches no longer feed an all-masked
    sequence into `nn.MultiheadAttention`, which previously produced NaNs during policy
    evaluation.
  - Added `use_pedestrian_attention` flag to `GridSocNavExtractor`.
  - Slot keys (`pedestrians_positions`, `pedestrians_velocities`) are removed from the flat MLP
    path and processed by the attention head when the flag is set.
  - `pedestrians_count` is used as a key-padding mask and remains in the flat MLP path.
- `scripts/training/train_ppo.py`
  - `_resolve_policy_selection` now detects `use_pedestrian_attention` in
    `feature_extractor_kwargs` and sets `critic_profile` to `attention_grid_socnav` (or
    `asymmetric_attention_grid_socnav` when combined with `asymmetric_critic`).
- Tests
  - Added unit coverage for `PedestrianAttentionHead` shape and masking behavior.
  - Added extractor-level coverage for slot key routing and fail-closed behavior.
  - Added a regression check that zero-pedestrian batches keep the attention-enabled extractor
    finite.
  - Added contract tests for both attention-only and attention + asymmetric critic profiles.
- Config
  - Added `configs/training/ppo/ablations/expert_ppo_issue_791_attention_head_stage1.yaml`.

## Canonical Config

- `configs/training/ppo/ablations/expert_ppo_issue_791_attention_head_stage1.yaml`

## Validation

```bash
uv run pytest tests/test_grid_socnav_extractor.py -q
uv run pytest tests/training/test_train_expert_ppo_contract.py -q
```

Original stage-1 SLURM gate `11436` failed after training, during deterministic evaluation,
with `tensor([[nan, nan]], device='cuda:0')` action means. The root cause was fully masked
pedestrian batches when `pedestrians_count == 0`.

Post-fix local repro evidence:

- `PedestrianAttentionHead(...).forward(...)` stays finite for `count=[[2],[5],[0]]`.
- `GridSocNavExtractor(..., use_pedestrian_attention=True)` stays finite when the entire batch
  has `pedestrians_count == 0`.
- `AsymmetricGridSocNavPolicy.predict(...)` now returns finite deterministic actions for an
  attention-enabled zero-pedestrian observation.

## Follow-Up

- Stage-1 evidence before the fix:
  - reward curriculum gate completed and ended around `success_rate=0.0143`.
  - asymmetric critic gate completed and ended around `success_rate=0.171`.
  - attention gate matched stable training stats through 8k steps, then failed in evaluation.
- Post-fix follow-up status:
  - `11439` reward curriculum 32k: completed with `success_rate=0.014285714285714285`.
  - `11440` asymmetric critic 32k: completed with `success_rate=0.014285714285714285`.
  - `11441` attention + asymmetric critic 32k: completed with `success_rate=0.014285714285714285`
    and no NaN regression.
- Interpretation: issue 791 now has executable stability evidence for all three bounded
  improvements and the stacked configuration, but no quality uplift at 32k yet.
- Next promotion step is a longer-horizon 128k campaign using:
  - `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_128k.yaml`
  - `configs/training/ppo/ablations/expert_ppo_issue_791_asymmetric_critic_promotion_128k.yaml`
  - `configs/training/ppo/ablations/expert_ppo_issue_791_attention_head_promotion_128k.yaml`
  with explicit WandB-required launch policy in the reusable issue-791 SLURM wrappers.
