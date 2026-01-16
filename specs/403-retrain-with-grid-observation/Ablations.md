# Planned Ablations for Issue 403

Purpose: Track **one‑variable** experiments to isolate which design choices matter.
All ablations should be compared against the **baseline** training run (simple
feature extractor, simple_reward, fixed grid config).

## Rules
- Change **one factor at a time**.
- Keep evaluation protocol and seeds constant.
- Record results in the same manifest/JSONL pipeline.

## Candidate Ablations (initial list)
1) **Observation ablation**
   - SocNav + grid (baseline) vs SocNav‑only (no grid).  
     *Why:* isolate contribution of obstacle context from the grid.
2) **Pedestrian encoder**
   - Flatten ped fields (baseline) vs ped encoder + pooling/attention.
3) **Reward shaping**
   - `simple_reward` (baseline) vs `punish_action_reward`.
4) **Curriculum**
   - Curriculum off (baseline) vs density curriculum on.
5) **Ped‑robot repulsion**
   - `prf_config.is_active=True` (baseline) vs `False`.
6) **Grid channel set**
   - `[OBSTACLES, PEDESTRIANS, COMBINED]` (baseline) vs reduced/expanded channel sets.

## Notes
- Add new ablations here as ideas arise; prioritize those that address reviewer‑facing
  questions (generalization, stability, interpretability).
