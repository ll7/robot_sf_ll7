# Issue #872 CARLA Oracle Replay Bridge Status

Issue: [#872](https://github.com/ll7/robot_sf_ll7/issues/872)

Status date: 2026-05-21

## Scope Boundary

Issue #872 is a parent epic for the CARLA oracle replay bridge (see the
[T0/T1 contract in #928](./issue_928_carla_t0_t1_replay_contract.md)). The tracked definition of
done includes neutral export, clear missing-CARLA failure behavior, at least one positive CARLA
oracle replay smoke on a CARLA-capable host, and trajectory-level metric comparison between
Robot-SF and CARLA outputs.

This note records the current implementation boundary. It does not close #872.

## Current State

Completed CARLA-free T0 bridge pieces:

- `robot_sf_carla_bridge/` is import-safe without CARLA installed.
- `carla-replay-export.v1` payloads can represent certified Robot-SF scenario data as neutral JSON.
- T0 helper APIs can build, read, write, and validate export payloads and local manifests.
- CLI entry points expose export, manifest validation, batch validation, availability checks, and
  schema catalog inspection.
- Missing-CARLA availability checks report explicit `not-available` metadata instead of importing
  optional CARLA dependencies at package import time.

Current live replay state:

- Setup-only smoke, Docker runtime substrate, live replay command/server connectivity, and
  static-geometry proxy support are implemented by the completed child issues.
- [#1430](https://github.com/ll7/robot_sf_ll7/issues/1430) attempted the post-#1329 certified live
  replay on the fallback host `imech156-u`. The pinned CARLA Docker runtime connected to CARLA
  `0.9.16` on `Town10HD_Opt`, then failed closed with `CARLA failed to spawn robot`.
- The #1110 parity adapter marks the #1430 report `unavailable` because CARLA status/mode is
  `failed`.

## Claim Boundary

The current stack supports CARLA-free export plumbing and optional-dependency readiness checks. It
is not yet simulator-transfer evidence.

Specifically:

- `not-available` missing-CARLA checks are expected setup results, not successful CARLA replay.
- T0 neutral export validates data contracts, not coordinate or physics parity.
- Setup-only T1 smoke coverage does not prove that a CARLA world can replay a certified scenario.
- The #1430 live replay reaches a CARLA server but fails before oracle-replay metrics exist, so it
  is runtime evidence and a concrete blocker, not parity or transfer evidence.

## Remaining Parent Gaps

The #872 definition of done still needs:

- A positive live-CARLA T1 oracle replay on a CARLA-capable host.
- Metric-producing Robot-SF/CARLA comparison output from that replay.
- Follow-up diagnosis for the #1430 robot actor-spawn failure in
  [#1437](https://github.com/ll7/robot_sf_ll7/issues/1437).
- Documentation that continues to separate setup, replay, degraded, failed, and metric-parity
  claims.

## Follow-Up Issues

- [#1430](https://github.com/ll7/robot_sf_ll7/issues/1430) records the post-#1329 live replay
  attempt and unavailable parity report.
- [#1437](https://github.com/ll7/robot_sf_ll7/issues/1437) should diagnose and fix or further
  narrow the `CARLA failed to spawn robot` blocker.

## Validation

This status note is documentation over the #1430 live evidence. PR validation should include the
CARLA bridge tests and docs proof consistency. The parent remains open because #1430 did not reach
`oracle-replay`.
