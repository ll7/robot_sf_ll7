# Issue #1485 CARLA Transfer-Boundary Follow-Up

Issue: [#1485](https://github.com/ll7/robot_sf_ll7/issues/1485)

Closed parent predecessor:
[`issue_872_carla_oracle_replay_bridge_status.md`](issue_872_carla_oracle_replay_bridge_status.md)

Related contract:
[`issue_1444_carla_coordinate_alignment_contract.md`](issue_1444_carla_coordinate_alignment_contract.md)

Merged evidence:

- [`issue_1442_carla_native_spawn_probe.md`](issue_1442_carla_native_spawn_probe.md)
  / PR #1466
- [`issue_1467_carla_replay_metrics.md`](issue_1467_carla_replay_metrics.md)
  / PR #1468
- PR #1479 parent-status refresh

## Goal

Keep the CARLA transfer boundary clear after the bounded #872 parent closed on
2026-05-25. This note preserves the distinction between setup-only evidence,
adapted replay, native or aligned replay, metric parity, fail-closed statuses,
and any broader transfer claim.

## Current Boundary Taxonomy

| Boundary | Current evidence | What it proves | What it does **not** prove |
| --- | --- | --- | --- |
| Setup-only | [#1111](issue_1111_carla_setup_smoke.md) | CARLA Python/API path, T0 payload selection, and setup-only T1 smoke can run. | Live replay, actor spawning, metric parity, or transfer. |
| Adapted replay | [#1440](issue_1440_carla_spawn_projection.md), [#1442](issue_1442_carla_native_spawn_probe.md) | The certified #1111 payload can reach CARLA `oracle-replay` after an explicit map projection. | Native/aligned parity; the certified payload still needed about `18.191 m` of spawn projection. |
| Native/aligned replay | [#1442](issue_1442_carla_native_spawn_probe.md) / PR #1466 | A generated CARLA-aligned probe can replay with exact robot spawn and no adaptations. | Durable certified-fixture parity or broad transfer. |
| Metric parity | [#1467](issue_1467_carla_replay_metrics.md) / PR #1468 | The native probe can emit and compare a bounded metric subset (`success`, `collision`, `intervention_rate`, and `min_distance_m` when supported). | Full trajectory parity, richer comfort/TTC/SNQI-style coverage, or multi-scenario transfer. |
| Failed / unavailable / degraded | [Issue #691 fallback policy](issue_691_benchmark_fallback_policy.md) | The stack fails closed when runtime or contract conditions are not met. | Any benchmark-success or transfer claim. |
| Transfer claim | No current issue provides this. | Nothing beyond the bounded v1 replay/parity surface accepted for #872 closure. | Broad simulator-transfer evidence, paper-facing transfer language, or benchmark-strength CARLA validation. |

## Why #872 Could Close

The accepted 2026-05-25 audit decision treated #872 as a bounded parent for one
native or explicitly aligned parity attempt plus conservative documentation, not
as an indefinitely growing CARLA transfer program.

The closure basis was:

1. PR #1466: merged native or aligned replay evidence and kept the certified
   payload's adapted replay boundary explicit.
2. PR #1468: merged native replay metric emission and bounded parity-comparison
   support for the generated probe.
3. PR #1479: refreshed the parent-status note so the repo-local documentation
   matched the merged child evidence before closure.

This closure does **not** promote setup-only, adapted, failed, unavailable, or
degraded outcomes into transfer success.

## Deferred Work Boundary

If broader CARLA replay work is still desired, do not reopen #872 and do not
stretch this documentation issue into an implementation campaign.

Open a separate benchmark issue with its own explicit contract for:

- scenario list and scenario-family scope,
- seed policy and replay count,
- host/runtime assumptions,
- required replay mode (`native` vs `aligned`) and alignment tolerance,
- metric set and acceptance criteria,
- artifact policy, including durable evidence pointers instead of bulky local
  CARLA outputs,
- and fail-closed handling for `failed`, `unavailable`, or `degraded` rows.

Until that exists, keep broader CARLA work described as deferred follow-up, not
as existing transfer evidence.

## Artifact Boundary

Tracked CARLA evidence should stay compact and reviewable. Keep large runtime
artifacts, raw videos, Docker state, and ignored `output/` trees out of git.
If future replay work needs durable proof, promote only small manifests,
summaries, or checksummed evidence copies under `docs/context/evidence/`.

## Validation

This note is documentation only. Re-check:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
