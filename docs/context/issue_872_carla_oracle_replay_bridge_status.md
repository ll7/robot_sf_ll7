# Issue #872 CARLA Oracle Replay Bridge Status

Issue: [#872](https://github.com/ll7/robot_sf_ll7/issues/872)

Status date: 2026-05-09

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

Next active smoke slice:

- [#1003](https://github.com/ll7/robot_sf_ll7/issues/1003) /
  [PR #1100](https://github.com/ll7/robot_sf_ll7/pull/1100) carries the setup-only T1 smoke path.
  Its purpose is dependency gating and fail-closed behavior, not live simulator replay proof.

## Claim Boundary

The current stack supports CARLA-free export plumbing and optional-dependency readiness checks. It
is not yet simulator-transfer evidence.

Specifically:

- `not-available` missing-CARLA checks are expected setup results, not successful CARLA replay.
- T0 neutral export validates data contracts, not coordinate or physics parity.
- Setup-only T1 smoke coverage does not prove that a CARLA world can replay a certified scenario.
- No current artifact compares CARLA oracle trajectories against Robot-SF trajectories.

## Remaining Parent Gaps

The #872 definition of done still needs:

- A positive live-CARLA T1 oracle replay smoke on a CARLA-capable host.
- A conservative Robot-SF vs CARLA trajectory metric comparison adapter.
- Evidence for the first replay target scenario and the exact environment used.
- Documentation that separates setup, replay, degraded, and metric-parity claims.

## Follow-Up Issues

- [#1111](https://github.com/ll7/robot_sf_ll7/issues/1111) runs the first live CARLA T1 oracle
  replay smoke on a CARLA-capable host.
- [#1110](https://github.com/ll7/robot_sf_ll7/issues/1110) adds the trajectory metric comparison
  adapter and conservative unavailable-field reporting.

## Validation

This status note is documentation only. PR validation for the note should include normal docs
format checks plus the CARLA bridge unit tests already used by the T0 stack. The live simulator
validation remains deferred to [#1111](https://github.com/ll7/robot_sf_ll7/issues/1111) because
this machine does not provide a CARLA runtime.
