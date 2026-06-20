# Issue #2777 Live Observation-Noise Replay

## Status

- Status: `fail_closed`
- Classification: `blocked`
- Rationale: No checked-in live scenario matrix preserving the #2755/#2756 occluded-emergence fixture boundary was found in the selected matrix.

## Claim Boundary

No benchmark-facing robustness claim. The command failed closed before live replay because the #2755/#2756 occluded-emergence fixture boundary could not be preserved.

## Fixture Contract

- `required_scenario`: `issue_2756_occluded_emergence`
- `required_family`: `occluded_emergence/deterministic_occluded_emergence`
- `first_visible_step`: `5`
- `delay_steps`: `2`
- `delay_only_expected_first_observed_step`: `7`
- `scenario_matrix`: `configs/scenarios/sets/issue_3201_pedestrian_dominated_observation_noise.yaml`
- `satisfied`: `False`
- `blocker`: `No checked-in live scenario matrix preserving the #2755/#2756 occluded-emergence fixture boundary was found in the selected matrix.`

## Conditions

| Condition | Status | Classification | Caveat |
|---|---|---|---|
| `noop` | `blocked` | `blocked` | No checked-in live scenario matrix preserving the #2755/#2756 occluded-emergence fixture boundary was found in the selected matrix. |
| `low_noise` | `blocked` | `blocked` | No checked-in live scenario matrix preserving the #2755/#2756 occluded-emergence fixture boundary was found in the selected matrix. |
| `medium_noise` | `blocked` | `blocked` | No checked-in live scenario matrix preserving the #2755/#2756 occluded-emergence fixture boundary was found in the selected matrix. |
| `missed_detection_only` | `blocked` | `blocked` | No checked-in live scenario matrix preserving the #2755/#2756 occluded-emergence fixture boundary was found in the selected matrix. |
| `occlusion_only` | `blocked` | `blocked` | No checked-in live scenario matrix preserving the #2755/#2756 occluded-emergence fixture boundary was found in the selected matrix. |
| `delay_only` | `blocked` | `blocked` | No checked-in live scenario matrix preserving the #2755/#2756 occluded-emergence fixture boundary was found in the selected matrix. |
| `combined` | `blocked` | `blocked` | No checked-in live scenario matrix preserving the #2755/#2756 occluded-emergence fixture boundary was found in the selected matrix. |

## Blockers

- No checked-in live scenario matrix preserving the #2755/#2756 occluded-emergence fixture boundary was found in the selected matrix.
