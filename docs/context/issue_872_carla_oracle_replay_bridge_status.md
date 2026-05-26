# Issue #872 CARLA Oracle Replay Bridge Status

Issue: [#872](https://github.com/ll7/robot_sf_ll7/issues/872)

Status date: 2026-05-25

> Status: GitHub issue #872 closed on 2026-05-25 by the accepted issue-audit
> decision.
> Keep this note as the bounded v1 parent-closure record.
> Broader transfer-boundary follow-up now lives in
> [`issue_1485_carla_transfer_boundary_follow_up.md`](issue_1485_carla_transfer_boundary_follow_up.md).

## Scope Boundary

Issue #872 is a parent epic for the CARLA oracle replay bridge (see the
[T0/T1 contract in #928](./issue_928_carla_t0_t1_replay_contract.md)). The tracked definition of
done includes neutral export, clear missing-CARLA failure behavior, at least one positive CARLA
oracle replay smoke on a CARLA-capable host, and trajectory-level metric comparison between
Robot-SF and CARLA outputs.

This note records why the bounded parent could close without turning the CARLA
bridge into an open-ended simulator-transfer program.

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
- Issue #1467 adds conservative T1 replay metric emission for native replay outputs and proves the
  Docker-runtime JSON can produce comparable parity rows for a generated CARLA-aligned native
  metric probe.

## Closure Basis And Claim Boundary

The merged stack now supports:

- setup-only T1 smoke for the certified payload (#1111),
- adapted live CARLA replay for the certified payload (#1440 / #1466),
- a generated CARLA-aligned native replay probe with exact robot spawn (#1442 / PR #1466),
- and limited native replay metric comparison for that generated probe (#1467 / PR #1468).

That bounded v1 surface was accepted as enough to close the parent on 2026-05-25.
It is still not broad simulator-transfer evidence.

Keep these claim boundaries explicit:

- `not-available`, `failed`, and `degraded` remain fail-closed diagnostic
  setup/runtime results, not successful replay.
- T0 neutral export validates data contracts, not coordinate or physics parity.
- Setup-only T1 smoke does not prove live simulator replay.
- `oracle-replay-adapted` proves CARLA actor execution after a recorded
  projection, not native or aligned Robot-SF/CARLA metric parity.
- The #1440 / #1466 certified-payload replay still needed about `18.191 m`
  of robot-spawn projection, so its comparable trajectory metrics remain
  intentionally unavailable.
- The generated CARLA-aligned native probe from #1442 shows that native or
  explicitly aligned spawn can run, but it is not the same as regenerating the
  durable certified #1111 fixture as native metric-bearing evidence.
- Current comparable native replay metrics are limited to fields emitted by the
  live replay summary, such as success, collision, intervention_rate, and
  `min_distance_m` when geometry exists. They do not justify broader transfer
  claims by themselves.
- The #1467 native metric probe is comparable smoke evidence, not a broad CARLA
  transfer claim. It proves the metric-emission path for a bounded probe.

## Post-Closure Follow-Up Boundary

The parent is closed, so the following items are no longer parent blockers.
They are strengthening or expansion work that must stay outside #872:

- a durable certified native or explicitly aligned fixture that is not derived
  from ignored exploratory output,
- richer Robot-SF vs CARLA trajectory metrics from that native/aligned replay,
- broader multi-scenario replay evidence,
- and any paper-facing or benchmark-strength CARLA transfer claim.

Those follow-ups now belong to
[`issue_1485_carla_transfer_boundary_follow_up.md`](issue_1485_carla_transfer_boundary_follow_up.md)
or a separate benchmark issue when the scope expands beyond documentation.

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
- [#1485](https://github.com/ll7/robot_sf_ll7/issues/1485) owns the post-closure
  transfer-boundary wording cleanup and keeps broader replay planning out of
  this closed parent.
- Any broader multi-scenario CARLA replay campaign should remain a separate
  benchmark issue rather than extending this parent closure bar.

## Validation

This status note is documentation only. The supporting implementation evidence for the current
state is recorded in
[`issue_1440_carla_spawn_projection.md`](issue_1440_carla_spawn_projection.md),
[`issue_1442_carla_native_spawn_probe.md`](issue_1442_carla_native_spawn_probe.md),
[`issue_1467_carla_replay_metrics.md`](issue_1467_carla_replay_metrics.md),
PR #1466, PR #1468, and PR #1479.
The closed parent should not be reopened from non-CARLA checks, setup-only
evidence, server-connectivity proof, static-geometry code support, adapted
replay alone, or native probe metrics that do not satisfy a broader follow-up
issue's explicit contract.
