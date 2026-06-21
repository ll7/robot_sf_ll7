# Issue #2777 Live Observation-Noise Replay

## Status

- Status: `live_replay`
- Classification: `scenario_too_weak`
- Rationale: All seven perturbation-family live replays completed.

## Fixture Contract

- Scenario: `issue_2756_occluded_emergence`
- Seed: `111`
- Matrix: `configs/scenarios/sets/issue_3320_occluded_emergence_live_replay.yaml`
- No-op first observed step: `5`
- Delay-only first observed step: `7`
- Satisfied: `True`

## Claim Boundary

Stress-slice live planner/environment replay for one scenario, one seed, and seven #2755 perturbation families. Treat as benchmark-facing only when fixture_contract.satisfied is true; otherwise diagnostic-only.

Raw trace JSON and per-condition reports were generated locally to build this summary but are not committed.

## Conditions

| Condition | Status | First observed | Hidden observations | Command changed | Classification |
|---|---|---:|---:|---|---|
| `noop` | `live_replay` | `5` | `5` | `n/a` | `baseline` |
| `low_noise` | `live_replay` | `5` | `5` | `False` | `scenario_too_weak` |
| `medium_noise` | `live_replay` | `5` | `5` | `False` | `scenario_too_weak` |
| `missed_detection_only` | `live_replay` | `None` | `5` | `False` | `scenario_too_weak` |
| `occlusion_only` | `live_replay` | `5` | `5` | `False` | `scenario_too_weak` |
| `delay_only` | `live_replay` | `7` | `5` | `False` | `scenario_too_weak` |
| `combined` | `live_replay` | `5` | `5` | `False` | `scenario_too_weak` |

## Interpretation

- The wrapper now executes on the intended issue #2756 occluded-emergence fixture boundary.
- The no-op trace first observes the pedestrian at step 5, and the delay-only condition first observes it at step 7.
- Perturbations changed planner-input observations, but selected commands and progress/risk summaries stayed unchanged; this is diagnostic stress evidence, not a robustness claim.
- The closest robot-pedestrian distance stayed outside the near-field target, so the scenario remains classified as too weak for a behavioral robustness conclusion.
