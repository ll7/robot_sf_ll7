# Issue #872 CARLA Oracle Replay Bridge Status

Issue: [#872](https://github.com/ll7/robot_sf_ll7/issues/872)

Status date: 2026-05-24

## Scope Boundary

Issue #872 is a parent epic for the CARLA oracle replay bridge (see the
[T0/T1 contract in #928](./issue_928_carla_t0_t1_replay_contract.md)). The tracked definition of
done includes neutral export, clear missing-CARLA failure behavior, at least one positive CARLA
oracle replay smoke on a CARLA-capable host, and trajectory-level metric comparison between
Robot-SF and CARLA outputs.

This note records the current implementation boundary. It does not close #872.

## Current State

Completed CARLA bridge pieces:

- `robot_sf_carla_bridge/` is import-safe without CARLA installed.
- `carla-replay-export.v1` payloads can represent certified Robot-SF scenario data as neutral JSON.
- T0 helper APIs can build, read, write, and validate export payloads and local manifests.
- CLI entry points expose export, manifest validation, batch validation, availability checks, and
  schema catalog inspection.
- Missing-CARLA availability checks report explicit `not-available` metadata instead of importing
  optional CARLA dependencies at package import time.
- Setup-only T1 smoke evidence is merged (#1111).
- The conservative CARLA-free metric comparison adapter is implemented (#1110).
- The Docker-backed pinned CARLA 0.9.16 runtime path is implemented (#1179).
- Live CARLA server connectivity and static-geometry proxy replay support are implemented
  (#1169 and #1329).
- Issue #1430 and Issue #1437 recorded the post-static-geometry live replay blocker and narrowed it
  to a robot vehicle spawn rejection.
- Issue #1440 proves the certified payload can reach live `oracle-replay` after an explicit CARLA
  map spawn projection for the robot vehicle.
- Issue #1442 / PR #1466 records fresh native/aligned replay evidence under the accepted #1444
  contract. The existing certified #1111 payload still requires adapted replay with a large
  robot-spawn projection, while a generated CARLA-aligned probe reaches native `oracle-replay`.
- Issue #1467 / PR #1468 emits native live-replay metrics from CARLA oracle replay summaries and
  makes the parity comparator consume Docker runtime summaries shaped as `replay.metrics`.

## Claim Boundary

The current stack now supports adapted live CARLA replay for the certified payload, native
CARLA-aligned replay probing, and limited native replay metric comparison. It is still not broad
simulator-transfer evidence.

Specifically:

- `not-available` and `failed` checks are diagnostic setup/runtime results, not successful replay.
- T0 neutral export validates data contracts, not coordinate or physics parity.
- Setup-only T1 smoke coverage does not prove live simulator replay.
- `oracle-replay-adapted` proves CARLA actor execution after a recorded projection, not native
  Robot-SF/CARLA metric parity.
- The #1440 projection moved the robot spawn about `18.191 m`, so comparable trajectory metrics are
  intentionally unavailable.
- The generated CARLA-aligned native probe from #1442 shows that native spawn can run, but it is
  not the same as regenerating the durable certified #1111 fixture as native metric-bearing
  evidence.
- Current comparable native replay metrics are limited to fields emitted by the live replay
  summary, such as success, collision, and intervention rate. They do not justify broader transfer
  claims.

## Remaining Parent Gaps

The #872 definition of done still needs:

- A native or explicitly map-aligned CARLA replay whose coordinate semantics are comparable to
  Robot-SF.
- Comparable Robot-SF vs CARLA trajectory metrics from that native/map-aligned replay.
- A regenerated native metric-bearing certified fixture, or an explicit maintainer decision that
  the generated CARLA-aligned native probe plus #1467 metrics are sufficient for the parent closure
  rule after the child PRs land.
- Documentation that separates setup, failed, adapted replay, native replay, degraded, and
  metric-parity claims. The coordinate-alignment contract in
  [#1444](./issue_1444_carla_coordinate_alignment_contract.md) now provides this taxonomy.

## Follow-Up Issues

- [#1430](https://github.com/ll7/robot_sf_ll7/issues/1430) records post-#1329 live replay parity
  evidence and the first concrete robot-spawn failure.
- [#1437](https://github.com/ll7/robot_sf_ll7/issues/1437) narrows the robot-spawn failure to the
  CARLA blueprint, spawn API, and transform.
- [#1440](https://github.com/ll7/robot_sf_ll7/issues/1440) adds explicit robot spawn projection and
  records adapted live replay evidence.
- [#1444](./issue_1444_carla_coordinate_alignment_contract.md) defines the conservative replay-mode
  taxonomy and projection tolerance required before any Robot-SF/CARLA metric parity claim.
- [#1442](https://github.com/ll7/robot_sf_ll7/issues/1442) / PR #1466 owns the native/aligned replay
  evidence gate.
- [#1467](https://github.com/ll7/robot_sf_ll7/issues/1467) / PR #1468 owns native replay metric
  emission and comparison support.
- Any broader multi-scenario CARLA replay campaign should remain a separate follow-up child issue
  rather than extending this parent closure bar.

## Validation

This status note is documentation only. The supporting implementation evidence for the current
state is recorded in
[`issue_1440_carla_spawn_projection.md`](issue_1440_carla_spawn_projection.md), PR #1466, and
PR #1468. The parent should not be marked complete from non-CARLA checks, setup-only evidence,
server-connectivity proof, static-geometry code support, adapted replay alone, or native probe
metrics that do not satisfy the accepted closure fixture boundary.
