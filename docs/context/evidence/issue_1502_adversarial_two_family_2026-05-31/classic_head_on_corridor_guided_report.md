# Adversarial Route Generation Report (classic_head_on_corridor_low)

- objective_mode: `composite`
- optimizer: `optuna_tpe`
- seed: `123`
- trial_count: `100`
- valid_trial_count: `40`
- failed_trials: `60`
- best_score: `0.387602`

## Objective Components
- failure_proxy: `0.222222`
- delay_proxy: `0.550396`
- path_inefficiency: `0.000022`
- near_miss_stress: `1.000000`

## Rejections
- feasibility_rejection_counts: `{'invalid_start_or_goal': 60}`

## Replay
Add this to your scenario entry:
- `route_overrides_file: output/adversarial/issue_1502/issue1502-two-family-d4a49b26/classic_head_on_corridor/guided_route_search/classic_head_on_corridor_low_20260531_054246_045282/route_overrides.yaml`

## Visualizations
- `trajectories_overlay.png`
