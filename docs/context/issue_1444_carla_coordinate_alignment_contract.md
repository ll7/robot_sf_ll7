# Issue #1444 CARLA Coordinate Alignment Contract

Issue: [#1444](https://github.com/ll7/robot_sf_ll7/issues/1444)

Parent issue: [#872 CARLA Oracle Replay Bridge Status](./issue_872_carla_oracle_replay_bridge_status.md)

Related contracts:
- [T0/T1 contract in #928](./issue_928_carla_t0_t1_replay_contract.md)
- [#1442](https://github.com/ll7/robot_sf_ll7/issues/1442) (active parity gate)

## Goal

Define the coordinate-alignment contract that governs whether a CARLA oracle replay
qualifies as a candidate for Robot-SF/CARLA metric parity. This note does not
create new parity claims; it documents the conservative replay-mode taxonomy and
the tolerance boundaries required before any future parity claim can be made.

## Replay Mode Taxonomy

Every CARLA replay attempt must report one of the following modes:

| Mode | Definition | Parity Eligible? |
|---|---|---|
| `native` | Robot-SF coordinates, kinematics, and geometry transfer to CARLA without any unplanned projection, remapping, or spawn correction. | Yes |
| `aligned` | An explicit, pre-documented coordinate transformation or map alignment is applied; the projection distance and semantic rationale are recorded in replay metadata. | Only if projection is within an explicit threshold and the transformation is reversible. |
| `adapted` | A material coordinate correction (e.g., CARLA map spawn projection, waypoint snap, or geometry remapping) is applied to make the replay run. | No |
| `failed` | Replay was attempted but violated the contract, crashed, timed out, or produced invalid/incomplete outputs. | No |
| `not-available` | CARLA runtime, requested map, assets, or certified scenario input is missing. | No |

Only `native` and `aligned` are eligible for metric-parity comparison.
`adapted`, `failed`, and `not-available` are diagnostic results and must not be
reported as successful benchmark outcomes.

## Projection Tolerance for Parity

### Native mode
- Native parity requires zero unplanned projection tolerance.
- If any spawn, waypoint, or obstacle coordinate is silently adjusted by CARLA
  or the bridge, the replay is reclassified as `adapted`.

### Aligned mode (if used)
- An explicit maximum projection threshold must be defined in the contract
  metadata before replay begins.
- The threshold, actual projection distance, transform rationale, and inverse
  transform must be recorded in the replay metadata.
- If the actual projection exceeds the threshold, the replay is reclassified as
  `adapted`.
- Aligned mode is opt-in and must not be treated as native parity.

## Required CARLA Output Metadata

For any replay that reports `native` or `aligned`, the output bundle must
include:

- `replay_mode`: one of the five modes above.
- `projection_meters`: total Euclidean displacement applied to the robot spawn
  (0.0 for `native`).
- `projection_rationale`: free-text description of why projection was applied,
  or `"none"` for `native`.
- `carla_map_name`: exact CARLA map identifier used.
- `carla_server_version`: semantic version of the CARLA server.
- `robot_sf_commit`: Robot-SF Git commit SHA that generated the certified
  scenario.
- `scenario_cert_id`: certified scenario identifier.
- `timestamp_utc`: ISO-8601 UTC timestamp of the replay.
- `bridge_version`: version of `robot_sf_carla_bridge` used.

For `aligned` mode, additionally:

- `alignment_transform`: the explicit transform (or pointer to its canonical
  definition).
- `alignment_threshold_meters`: the pre-agreed tolerance.
- `alignment_inverse_available`: boolean indicating whether the transform can be
  reversed for metric comparison.

## Parity Report Fields

A `parity_report` comparing Robot-SF and CARLA trajectories must include:

- **Trajectory-level fields**: success, collision, minimum distance, TTC,
  comfort, jerk, curvature, intervention rate, and SNQI-compatible fields when
  supported.
- **Mode provenance**: the replay mode and metadata listed above.
- **Exclusion statement**: explicit note if `adapted`, `failed`, or
  `not-available` caused the report to be withheld or partial.
- **Comparable subset**: which episodes/scenarios were compared and which were
  excluded due to mode ineligibility.

## Parity Gate: Issue #1442

Issue #1442 is the active parity gate. No Robot-SF/CARLA metric parity claim
should be made until #1442 is resolved with `native` or threshold-bounded
`aligned` replay evidence.

## Benchmark Claim Boundary

This document is conservative and creates no new parity claims. It defines
eligibility rules; it does not assert that parity currently exists.

## Validation

This note is documentation only. Validation:

```bash
git diff --check
```

Future implementation should prove each replay mode is correctly classified by
the bridge and that `adapted` replay is never silently promoted to `aligned`
or `native`.

## Links

- Parent epic: [Issue #872](./issue_872_carla_oracle_replay_bridge_status.md)
- T0/T1 contract: [Issue #928](./issue_928_carla_t0_t1_replay_contract.md)
- Active parity gate: [#1442](https://github.com/ll7/robot_sf_ll7/issues/1442)
- Spawn projection evidence: [Issue #1440](./issue_1440_carla_spawn_projection.md)
- Live replay parity blocker: [Issue #1430](./issue_1430_carla_live_parity.md)
