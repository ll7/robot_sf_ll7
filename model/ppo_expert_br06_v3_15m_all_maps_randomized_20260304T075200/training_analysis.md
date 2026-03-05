# PPO BR-06 v3 Final Statistical Assessment

## Run Provenance
- Run name: `ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200`
- W&B run id: `b60iopxt`
- W&B URL: https://wandb.ai/ll7/robot_sf/runs/b60iopxt
- Terminal state: `finished`
- Final global step: `15,000,000`
- Runtime: `27,775 s` (~`7.7 h`)

## Final Training Snapshot (W&B summary)
- `eval/success_rate`: `0.89`
- `eval/collision_rate`: `0.11`
- `eval/snqi`: `0.14550596`
- `eval/eval_episode_return`: `36.480503`
- `eval/eval_avg_step_reward`: `0.25476125`

## Full Policy Analysis (canonical local evidence)
Artifacts:
- `output/benchmarks/20260304_184717_policy_analysis_ppo/summary.json`
- `output/benchmarks/20260304_184717_policy_analysis_ppo/report.md`
- `output/benchmarks/20260304_184717_policy_analysis_ppo/episodes.jsonl`
- `output/recordings/20260304_184717_policy_analysis_ppo/`

Aggregate over 141 episodes:
- Success: `123/141` (`0.8723`)
- Collision: `18/141` (`0.1277`)
- Terminations: `success=123`, `collision=18`, `max_steps=0`, `error=0`
- Integrity contradictions (`termination=collision` with success signal): `0`

Collision composition:
- Pedestrian-collision episode rate: `0.0567`
- Obstacle-collision episode rate: `0.0000`

## Interpretation
1. This run is materially better than the previous BR-06 v2 candidate (higher success and lower collision).
2. Outcome integrity is clean under the corrected semantics (no contradictory records).
3. Remaining risk is concentrated in dense-interaction scenarios (`classic_doorway_high`,
   `francis2023_robot_crowding`, `classic_crossing_high`, `francis2023_perpendicular_traffic`).

## Promotion Decision
Promote this run as the default PPO model for BR-06 onward, with caveat that issue #584 is
still open and local artifacts remain authoritative for timeline-level eval interpretation.
