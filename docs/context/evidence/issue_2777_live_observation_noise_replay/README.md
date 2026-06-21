# Issue #2777 Live Observation-Noise Replay

## Status

- Status: `live_replay`
- Classification: `policy_insensitive`
- Rationale: All seven perturbation-family live replays completed.

## Fixture Contract

- Scenario: `issue_2756_occluded_emergence`
- Seed: `111`
- Matrix: `configs/scenarios/sets/issue_3323_occluded_emergence_near_field_live_replay.yaml`
- No-op first observed step: `5`
- Delay-only first observed step: `7`
- No-op closest robot-pedestrian distance: `1.65520666531114` m at step `14`
- Satisfied: `True`

## Claim Boundary

Stress-slice live planner/environment replay for one scenario, one seed, and seven #2755 perturbation families. Treat as benchmark-facing only when fixture_contract.satisfied is true; otherwise diagnostic-only.

Raw trace JSON and per-condition reports were generated locally to build this summary but are not committed.

## Conditions

| Condition | Status | First observed | Hidden observations | Command changed | Classification |
|---|---|---:|---:|---|---|
| `noop` | `live_replay` | `5` | `5` | `n/a` | `baseline` |
| `low_noise` | `live_replay` | `5` | `5` | `False` | `policy_insensitive` |
| `medium_noise` | `live_replay` | `5` | `5` | `False` | `policy_insensitive` |
| `missed_detection_only` | `live_replay` | `None` | `5` | `False` | `policy_insensitive` |
| `occlusion_only` | `live_replay` | `5` | `5` | `False` | `policy_insensitive` |
| `delay_only` | `live_replay` | `7` | `5` | `False` | `policy_insensitive` |
| `combined` | `live_replay` | `5` | `5` | `False` | `policy_insensitive` |

## Interpretation

- The wrapper executes on the intended issue #2756 occluded-emergence fixture boundary using the issue #3323 near-field route matrix.
- The no-op trace first observes the pedestrian at step 5, and the delay-only condition first observes it at step 7.
- The no-op trace reaches a closest robot-pedestrian distance of 1.6552 m, inside the <=2 m near-field target.
- Perturbations changed planner-input observations, but selected commands and progress/risk summaries stayed unchanged; this is policy-insensitive diagnostic stress evidence, not a robustness claim.

## Issue #3328 Behavior Probe

`issue_3328_behavior_probe/` preserves this same near-field fixture boundary and adds an opt-in
high-amplitude noise condition (`std=1.0`, `bound=2.0`, seed 3328). The probe is classified
`behavior_sensitive_diagnostic_only`: it changes selected commands at steps 6 and 9 plus
progress/min-distance summaries on one seed, while medium noise and delay-only remain
policy-insensitive. It is diagnostic stress evidence only, not a robustness, sensor-realism, or
planner-superiority claim.

## Issue #3330 Seed/Amplitude Grid

`issue_3330_seed_amplitude_grid/` preserves the same #2756/#3323 near-field fixture boundary and
adds an opt-in two-amplitude, three-perturbation-seed Gaussian grid. The grid is classified
`behavior_sensitive_diagnostic_only`: behavior sensitivity is mixed across the selected cells.
`medium_noise_seed_3328`, `high_noise_seed_3328`, and `high_noise_seed_3330` changed selected
commands plus progress/min-distance summaries, while `medium_noise_seed_2755`,
`medium_noise_seed_3330`, `high_noise_seed_2755`, and `delay_only` remained policy-insensitive.
This is diagnostic stress evidence only, not a robustness, sensor-realism, or planner-superiority
claim.
