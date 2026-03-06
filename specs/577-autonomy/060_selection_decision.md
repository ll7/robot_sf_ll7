# 060 Selection Decision Log

## Promotion gates
- contradictions == 0
- reproducibility check passed
- success and safety metrics improved or justified

## Current decision
- Date: 2026-03-05
- Selected working champion: `guarded_ppo_v3`
- Evidence:
  - `output/tmp/planner_portfolio/guarded_ppo_compare/hard_summary.json`
  - `output/tmp/planner_portfolio/guarded_ppo_compare/global_summary.json`
- Gate status:
  - contradictions: pass (no contradiction findings in campaign artifacts)
  - reproducibility repeat: pending
  - success improvement: pass vs plain PPO (`global 0.227 -> 0.242`)
  - safety/clearance: pass vs plain PPO (ped collisions `7 -> 1`, mean min-distance `0.788 -> 0.851`)
- Promotion verdict:
  - `keep as champion candidate`, `do not promote to final` until repeat-run confirmation and targeted fixes for `classic_bottleneck_medium`.
