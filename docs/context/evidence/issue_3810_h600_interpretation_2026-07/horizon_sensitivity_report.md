# H600 vs H500 Horizon-Sensitivity Report

- Evidence status: `diagnostic-only`.
- Status: `ok`.
- H500 source status: `available_s10_h500_not_s20`.
- Caveat: this is not a causal horizon-only ablation.

| metric | planner_key | h500_rank | h600_rank | rank_delta | stability |
| --- | --- | --- | --- | --- | --- |
| snqi | goal | 10 | 3 | -7 | rank_flip |
| snqi | orca | 9 | 4 | -5 | rank_flip |
| snqi | ppo | 8 | 2 | -6 | rank_flip |
| snqi | prediction_planner | 7 | 1 | -6 | rank_flip |
| snqi | sacadrl | 11 | 6 | -5 | rank_flip |
| snqi | social_force | 12 | 7 | -5 | rank_flip |
| snqi | socnav_sampling | 6 | 5 | -1 | stable |
| success | goal | 11 | 4 | -7 | rank_flip |
| success | orca | 7 | 1 | -6 | rank_flip |
| success | ppo | 6 | 2 | -4 | rank_flip |
| success | prediction_planner | 8 | 3 | -5 | rank_flip |
| success | sacadrl | 10 | 5 | -5 | rank_flip |
| success | social_force | 12 | 7 | -5 | rank_flip |
| success | socnav_sampling | 9 | 6 | -3 | rank_flip |
| collision | goal | 11 | 5 | -6 | rank_flip |
| collision | orca | 6 | 1 | -5 | rank_flip |
| collision | ppo | 7 | 2 | -5 | rank_flip |
| collision | prediction_planner | 9 | 3 | -6 | rank_flip |
| collision | sacadrl | 12 | 6 | -6 | rank_flip |
| collision | social_force | 8 | 4 | -4 | rank_flip |
| collision | socnav_sampling | 10 | 7 | -3 | rank_flip |
| near_miss | goal | 5 | 5 | 0 | stable |
| near_miss | orca | 6 | 7 | 1 | stable |
| near_miss | ppo | 4 | 2 | -2 | rank_flip |
| near_miss | prediction_planner | 8 | 6 | -2 | rank_flip |
| near_miss | sacadrl | 3 | 3 | 0 | stable |
| near_miss | social_force | 2 | 4 | 2 | rank_flip |
| near_miss | socnav_sampling | 1 | 1 | 0 | stable |
