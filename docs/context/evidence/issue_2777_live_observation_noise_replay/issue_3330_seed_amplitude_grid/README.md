# Issue #2777 Live Observation-Noise Replay

## Status

- Status: `live_replay`
- Classification: `behavior_sensitive_diagnostic_only`
- Rationale: All selected live replays completed. One scenario fixture seed is diagnostic only and does not support a robustness, sensor-realism, or planner-superiority claim.

## Claim Boundary

Stress-slice live planner/environment replay for one scenario fixture seed and the selected perturbation condition set. Treat behavior-sensitive differences as diagnostic-only from this fixture; do not infer robustness, sensor realism, planner superiority, or scenario-general behavior.

## Fixture Contract

- `required_scenario`: `issue_2756_occluded_emergence`
- `required_family`: `occluded_emergence/deterministic_occluded_emergence`
- `first_visible_step`: `5`
- `delay_steps`: `2`
- `delay_only_expected_first_observed_step`: `7`
- `scenario_matrix`: `configs/scenarios/sets/issue_3323_occluded_emergence_near_field_live_replay.yaml`
- `satisfied`: `True`
- `blocker`: `None`

## Grid Interpretation

- Evidence status: `diagnostic-only`
- Label: `medium_amplitude_sensitive`
- Summary: Diagnostic-only grid classification: medium-amplitude-sensitive because at least one std=0.30 m, bound=0.60 m condition changed live behavior.
- Sensitive conditions: `medium_noise_3328, high_noise_3328, high_noise_3330`
- Medium-amplitude sensitive conditions: `medium_noise_3328`
- High-noise sensitive conditions: `high_noise_3328, high_noise_3330`
- Limitation: One scenario and one fixture seed only; this is diagnostic behavior-probe evidence, not a robustness, sensor-realism, or planner-superiority claim.

## Conditions

| Condition | Status | Classification | Command changed | Progress/risk changed | Min distance changed | Collision/near-miss changed | Stop/yield proxy changed | Caveat |
|---|---|---|---:|---:|---:|---:|---:|---|
| `noop` | `live_replay` | `baseline` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` |  |
| `delay_only` | `live_replay` | `policy_insensitive` | `False` | `False` | `False` | `False` | `False` | Perturbation changed planner-input observations in a near-field trace, but selected commands and progress/risk summaries were identical. |
| `medium_noise_2755` | `live_replay` | `policy_insensitive` | `False` | `False` | `False` | `False` | `False` | Perturbation changed planner-input observations in a near-field trace, but selected commands and progress/risk summaries were identical. |
| `medium_noise_3328` | `live_replay` | `behavior_sensitive_diagnostic_only` | `True` | `True` | `True` | `False` | `False` | Perturbation changed selected commands or progress/risk fields. This is live behavior evidence, but one seed is not enough for a benchmark-strength robustness claim. |
| `medium_noise_3330` | `live_replay` | `policy_insensitive` | `False` | `False` | `False` | `False` | `False` | Perturbation changed planner-input observations in a near-field trace, but selected commands and progress/risk summaries were identical. |
| `high_noise_2755` | `live_replay` | `policy_insensitive` | `False` | `False` | `False` | `False` | `False` | Perturbation changed planner-input observations in a near-field trace, but selected commands and progress/risk summaries were identical. |
| `high_noise_3328` | `live_replay` | `behavior_sensitive_diagnostic_only` | `True` | `True` | `True` | `False` | `False` | Perturbation changed selected commands or progress/risk fields. This is live behavior evidence, but one seed is not enough for a benchmark-strength robustness claim. |
| `high_noise_3330` | `live_replay` | `behavior_sensitive_diagnostic_only` | `True` | `True` | `True` | `False` | `False` | Perturbation changed selected commands or progress/risk fields. This is live behavior evidence, but one seed is not enough for a benchmark-strength robustness claim. |
