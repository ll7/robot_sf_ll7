# PPO BR-06 v2 Full Scenario Statistical Assessment

## Scope
- Model: `ppo_expert_br06_v2_15m_all_maps_20260303T074433` (W&B `c4s6142g`)
- Training config: `configs/training/ppo_imitation/expert_ppo_issue_576_br06_v2_15m_all_maps.yaml`
- Evaluation command: `policy_analysis_run.py --policy ppo --all --seed-set eval --max-seeds 3 --videos`
- Artifacts:
  - `episodes.jsonl`
  - `summary.json`
  - `report.md`
  - `videos/*.mp4`

## Headline Results
- Episodes: **141**
- Success rate: **0.766**
- Collision rate: **0.220**
- Failure reasons:
  - `collision`: 31 episodes
  - `terminated`: 1 episode
  - `error`: 1 episode

## Family-Level Breakdown
- `classic_*` scenarios:
  - success: **0.652**
  - collision: **0.333**
- `francis2023_*` scenarios:
  - success: **0.867**
  - collision: **0.120**

Interpretation: performance is materially weaker on classic interaction archetypes than on Francis scenarios.

## Worst Scenarios (collision-dominant)
- `classic_crossing_high`: collision 1.00, success 0.00
- `classic_crossing_low`: collision 1.00, success 0.00
- `classic_merging_low`: collision 1.00, success 0.00
- `classic_merging_medium`: collision 1.00, success 0.00
- `francis2023_narrow_doorway`: collision 1.00, success 0.00
- `francis2023_narrow_hallway`: collision 1.00, success 0.00

## Strong Scenarios
- Multiple bottleneck/head-on/group-crossing scenarios reached 1.0 success with 0.0 collision in this seed subset.

## Suspicious Findings
- One `error` episode due to integrity contradiction:
  - Scenario: `francis2023_pedestrian_overtaking`, seed `112`
  - Message indicates `outcome.route_complete=true` while `metrics.collisions>0` and `metrics.success<=0`.
- This indicates a remaining consistency bug in metric/outcome alignment for at least one trajectory path.

## Training-Dynamics Review (from W&B logs)
- Final step reached: ~`15.24M`.
- PPO stability indicators are numerically stable:
  - `train/approx_kl` around 0.012 (frequent early-stop inside PPO epochs from KL cap)
  - `train/clip_fraction` around 0.28–0.31
  - `train/explained_variance` around 0.96–0.98
- Despite stable optimization, eval quality plateaus in the 0.4–0.6 success / 0.4–0.6 collision regime.

## Is Training Long Enough?
- For this config, **more steps alone are unlikely to fix quality**.
- Reason: 15M already completed with persistent high collision on difficult families.

## Hyperparameters/Design Areas to Tune Next
1. **Entropy coefficient (`ent_coef`)**
- Current policy remains highly stochastic late in training (high action std).
- Try lower entropy pressure or entropy decay schedule.

2. **Reward profile weights (`route_completion_v2`)**
- Collision-heavy scenarios suggest safety penalties are still under-weighted relative to progress/terminal reward.
- Increase collision/near-miss/TTC penalties; verify via per-term reward logging.

3. **PPO rollout/trust-region shape**
- Sweep `n_steps`, `batch_size`, `clip_range`, `target_kl` jointly.
- Current frequent KL early-stop implies updates may be too constrained for late-stage refinement.

4. **Curriculum / sampling**
- Oversample hardest scenario families (crossing/merging/narrow hallway) during training.

## Suggested HPO Budgeting
- Stage A (coarse): 12–20 trials, each 4M steps.
- Stage B (narrow): top 4 trials to 8M steps.
- Stage C (final): top 1–2 trials to 15M steps + full all-scenario evaluation.

Rationale: current run shows quality signal early enough that full 15M for every candidate is inefficient.

## Issue #577 Readiness
- **Recommendation: do not move to #577 yet** as “model-ready”.
- Blockers:
  1. collision rate too high on critical classic scenarios,
  2. one integrity-contradiction error episode still present,
  3. no evidence yet that this checkpoint improves the hard families versus acceptance targets.

Proceed to #577 only after one tuned rerun clears explicit gates (e.g., overall collision <= 0.10 and no integrity contradictions).
