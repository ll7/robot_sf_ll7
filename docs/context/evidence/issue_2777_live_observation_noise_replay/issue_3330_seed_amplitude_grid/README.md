# Issue #2777 Live Observation-Noise Replay

## Status

- Status: `live_replay`
- Classification: `behavior_sensitive_diagnostic_only`
- Rationale: All selected live replays completed. The predeclared seed/amplitude grid is diagnostic-only and does not support a robustness, sensor-realism, or planner-superiority claim.

## Claim Boundary

Stress-slice live planner/environment replay for one scenario, one scenario seed, and a predeclared observation-noise seed/amplitude grid. Treat persistence, disappearance, or mixed behavior sensitivity as diagnostic-only; do not infer robustness, sensor realism, or planner superiority.

## Fixture Contract

- `required_scenario`: `issue_2756_occluded_emergence`
- `required_family`: `occluded_emergence/deterministic_occluded_emergence`
- `first_visible_step`: `5`
- `delay_steps`: `2`
- `delay_only_expected_first_observed_step`: `7`
- `scenario_matrix`: `configs/scenarios/sets/issue_3323_occluded_emergence_near_field_live_replay.yaml`
- `satisfied`: `True`
- `blocker`: `None`

## Behavior Sensitivity Summary

- Interpretation: `behavior_sensitivity_is_seed_or_amplitude_specific_in_selected_grid`
- Behavior-sensitive conditions: `['medium_noise_seed_3328', 'high_noise_seed_3328', 'high_noise_seed_3330']`
- Policy-insensitive conditions: `['delay_only', 'medium_noise_seed_2755', 'medium_noise_seed_3330', 'high_noise_seed_2755']`
- Command-changed conditions: `['medium_noise_seed_3328', 'high_noise_seed_3328', 'high_noise_seed_3330']`
- Progress/risk-changed conditions: `['medium_noise_seed_3328', 'high_noise_seed_3328', 'high_noise_seed_3330']`
- Stop/yield proxy-changed conditions: `[]`
- Boundary: Diagnostic-only persistence summary over completed live replay cells. It is not a benchmark-strength robustness, sensor-realism, or planner-superiority claim.

## Conditions

| Condition | Status | Classification | Command changed | Progress/risk changed | Min distance changed | Collision/near-miss changed | Stop/yield proxy changed | Caveat |
|---|---|---|---:|---:|---:|---:|---:|---|
| `noop` | `live_replay` | `baseline` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` |  |
| `delay_only` | `live_replay` | `policy_insensitive` | `False` | `False` | `False` | `False` | `False` | Perturbation changed planner-input observations in a near-field trace, but selected commands and progress/risk summaries were identical. |
| `medium_noise_seed_2755` | `live_replay` | `policy_insensitive` | `False` | `False` | `False` | `False` | `False` | Perturbation changed planner-input observations in a near-field trace, but selected commands and progress/risk summaries were identical. |
| `medium_noise_seed_3328` | `live_replay` | `behavior_sensitive_diagnostic_only` | `True` | `True` | `True` | `False` | `False` | Perturbation changed selected commands or progress/risk fields. This is live behavior evidence, but the selected diagnostic slice is not enough for a benchmark-strength robustness claim. |
| `medium_noise_seed_3330` | `live_replay` | `policy_insensitive` | `False` | `False` | `False` | `False` | `False` | Perturbation changed planner-input observations in a near-field trace, but selected commands and progress/risk summaries were identical. |
| `high_noise_seed_2755` | `live_replay` | `policy_insensitive` | `False` | `False` | `False` | `False` | `False` | Perturbation changed planner-input observations in a near-field trace, but selected commands and progress/risk summaries were identical. |
| `high_noise_seed_3328` | `live_replay` | `behavior_sensitive_diagnostic_only` | `True` | `True` | `True` | `False` | `False` | Perturbation changed selected commands or progress/risk fields. This is live behavior evidence, but the selected diagnostic slice is not enough for a benchmark-strength robustness claim. |
| `high_noise_seed_3330` | `live_replay` | `behavior_sensitive_diagnostic_only` | `True` | `True` | `True` | `False` | `False` | Perturbation changed selected commands or progress/risk fields. This is live behavior evidence, but the selected diagnostic slice is not enough for a benchmark-strength robustness claim. |
