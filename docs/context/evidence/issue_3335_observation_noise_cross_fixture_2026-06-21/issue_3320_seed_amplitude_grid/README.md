# Issue #2777 Live Observation-Noise Replay

## Status

- Status: `fail_closed`
- Classification: `blocked`
- Rationale: Issue #3330 seed/amplitude grid requires no-op closest_robot_ped_distance <= 2.0 m; observed 21.803818559278636.

## Claim Boundary

Stress-slice live planner/environment replay for one scenario fixture seed and the selected perturbation condition set. Treat behavior-sensitive differences as diagnostic-only from this fixture; do not infer robustness, sensor realism, planner superiority, or scenario-general behavior.

## Fixture Contract

- `required_scenario`: `issue_2756_occluded_emergence`
- `required_family`: `occluded_emergence/deterministic_occluded_emergence`
- `first_visible_step`: `5`
- `delay_steps`: `2`
- `delay_only_expected_first_observed_step`: `7`
- `scenario_matrix`: `configs/scenarios/sets/issue_3320_occluded_emergence_live_replay.yaml`
- `satisfied`: `True`
- `blocker`: `None`

## Grid Interpretation

- Evidence status: `diagnostic-only`
- Label: `unavailable_fail_closed`
- Summary: Grid interpretation is unavailable/fail-closed because at least one required live replay condition did not complete under the fixture guardrails.
- Sensitive conditions: `none`
- Medium-amplitude sensitive conditions: `none`
- High-noise sensitive conditions: `none`
- Limitation: Diagnostic-only; no robustness claim is available.

## Conditions

| Condition | Status | Classification | Command changed | Progress/risk changed | Min distance changed | Collision/near-miss changed | Stop/yield proxy changed | Caveat |
|---|---|---|---:|---:|---:|---:|---:|---|
| `noop` | `live_replay` | `baseline` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` |  |
| `delay_only` | `live_replay` | `scenario_too_weak` | `False` | `False` | `False` | `False` | `False` | The live condition did not expose a near-field behavior difference against the no-op trace. |
| `medium_noise_2755` | `live_replay` | `scenario_too_weak` | `False` | `False` | `False` | `False` | `False` | The live condition did not expose a near-field behavior difference against the no-op trace. |
| `medium_noise_3328` | `live_replay` | `scenario_too_weak` | `False` | `False` | `False` | `False` | `False` | The live condition did not expose a near-field behavior difference against the no-op trace. |
| `medium_noise_3330` | `live_replay` | `scenario_too_weak` | `False` | `False` | `False` | `False` | `False` | The live condition did not expose a near-field behavior difference against the no-op trace. |
| `high_noise_2755` | `live_replay` | `scenario_too_weak` | `False` | `False` | `False` | `False` | `False` | The live condition did not expose a near-field behavior difference against the no-op trace. |
| `high_noise_3328` | `live_replay` | `scenario_too_weak` | `False` | `False` | `False` | `False` | `False` | The live condition did not expose a near-field behavior difference against the no-op trace. |
| `high_noise_3330` | `live_replay` | `scenario_too_weak` | `False` | `False` | `False` | `False` | `False` | The live condition did not expose a near-field behavior difference against the no-op trace. |

## Blockers

- Issue #3330 seed/amplitude grid requires no-op closest_robot_ped_distance <= 2.0 m; observed 21.803818559278636.
