# Issue 791 Attention Head Gate

## Goal

Implement the third and final bounded PPO improvement from issue 791: a config-first masked
multi-head self-attention head over variable-length pedestrian slots, replacing the flat-MLP
treatment of padded pedestrian arrays in the actor feature extractor.

## What Changed

- `robot_sf/feature_extractors/grid_socnav_extractor.py`
  - Added `PedestrianAttentionHead`: one layer of multi-head self-attention with masked mean
    pooling over valid pedestrian slots.
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

All targeted suites passed.

## Follow-Up

- Issue 791 is now fully implemented across all three gates.
- Benchmark training runs using the stage-1 config will determine whether each gate improves
  success rate; gates that do not improve should be documented as negative results.
- Do not extend this issue further; open new issues for any follow-on architecture work.
