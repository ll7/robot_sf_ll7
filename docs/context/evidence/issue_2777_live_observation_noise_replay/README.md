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
