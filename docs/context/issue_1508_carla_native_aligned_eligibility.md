# Issue #1508 CARLA Native/Aligned Eligibility Audit

Date: 2026-05-31

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1508>
- <https://github.com/ll7/robot_sf_ll7/issues/1491>
- <https://github.com/ll7/robot_sf_ll7/issues/1444>
- <https://github.com/ll7/robot_sf_ll7/issues/1442>
- <https://github.com/ll7/robot_sf_ll7/issues/1467>

Compact evidence:
[evidence/issue_1508_carla_eligibility_2026-05-31/candidate_eligibility_summary.json](evidence/issue_1508_carla_eligibility_2026-05-31/candidate_eligibility_summary.json)

## Scope

This audit answers the Issue #1508 pre-campaign question: which existing CARLA replay candidates
are eligible for a native or explicitly aligned replay campaign under the Issue #1444 contract?

It does not run a new CARLA replay, launch a multi-scenario campaign, or make a paper-facing
simulator-transfer claim. It classifies the existing durable evidence so Issue #1491 has a clear
next step before more expensive CARLA work.

## Eligibility Contract

The current replay-mode contract is
[issue_1444_carla_coordinate_alignment_contract.md](issue_1444_carla_coordinate_alignment_contract.md):

- `native` and `aligned` are the only replay modes eligible for metric parity comparison.
- `adapted`, `failed`, and `not-available` replay modes are not success evidence.
- Broader benchmark statuses such as `degraded`, fallback, or setup-only outputs are also
  non-success evidence under the fail-closed benchmark policy.
- A candidate row still needs durable fixture provenance and comparable metrics before it can become
  benchmark-strength CARLA/native parity evidence.

## Candidate Table

| Candidate | Current evidence | Eligibility | Why |
|---|---|---|---|
| Certified Issue #1111 payload | [Issue #1430](issue_1430_carla_live_parity.md), [Issue #1442](issue_1442_carla_native_spawn_probe.md) | `not_eligible` | The certified payload reached `oracle-replay-adapted`; Issue #1442 records about `18.191 m` of robot spawn projection. |
| Issue #1442 native-spawn probe | [Issue #1442](issue_1442_carla_native_spawn_probe.md) | `runtime_native_smoke_only` | Native spawn was proven for a generated inverse-coordinate probe, but comparable CARLA metrics were unavailable. |
| Issue #1467 native metric probe | [Issue #1467](issue_1467_carla_replay_metrics.md) | `comparable_smoke_only` | The generated native probe emitted comparable `success`, `collision`, and `intervention_rate`, but it is not a certified scenario row. |

## Result

No existing candidate is ready for a multi-scenario native/aligned CARLA parity campaign today.

The best available row is the Issue #1467 generated native metric probe. It proves the metric
emission/comparison path can work for a bounded smoke fixture, but it remains smoke evidence rather
than benchmark-strength evidence because the fixture is generated and not yet a certified scenario
contract.

The certified Issue #1111 payload remains blocked for native/aligned parity because it still needs
spawn projection. That is a fail-closed blocker, not a degraded success.

## Next Unlocked Child

Issue #1491 should next create or certify a durable CARLA-native or explicitly aligned fixture, then
run the metric-emitting live replay path from Issue #1467 against that fixture. A candidate should
not enter a multi-scenario campaign until the replay bundle records:

- replay mode `native` or `aligned`,
- zero unplanned projection for native mode, or an explicit reversible alignment for aligned mode,
- CARLA replay status `oracle-replay`,
- comparable Robot-SF and CARLA metrics,
- and a compact durable evidence bundle.

## Validation

Validation for this audit was intentionally documentation/evidence validation only:

```bash
python -m json.tool docs/context/evidence/issue_1508_carla_eligibility_2026-05-31/candidate_eligibility_summary.json
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```

No new live CARLA replay was run for this issue. The audit cites existing tracked evidence from
Issues #1430, #1442, and #1467.
