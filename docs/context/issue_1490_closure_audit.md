# Issue #1490 Closure Audit

Date: 2026-07-04
Issue: <https://github.com/ll7/robot_sf_ll7/issues/1490>

## Scope

This note audits whether Issue #1490 can be closed after the merged predictive planner v2
readiness and decision-packet pull requests. It does not run training, submit SLURM jobs, execute a
full benchmark campaign, or promote a benchmark or paper-facing claim.

Compact machine-readable evidence lives in
[`evidence/issue_1490_closure_audit_2026-07-04/summary.json`](evidence/issue_1490_closure_audit_2026-07-04/summary.json).

## Verdict

Status: `blocked_not_closable`.

Issue #1490 should remain open as a blocked historical umbrella. The old baseline, obstacle-only,
ego-only, and combined four-way matrix is not executable under the current issue contract. The
remaining blocker is a maintainer-selected revised hypothesis plus a provenance-backed closed-loop
coupling-gate artifact that recommends `continue`.

## Acceptance Criteria Audit

| Criterion | Evidence | Status |
| --- | --- | --- |
| Baseline, obstacle-only, ego-only, and combined variants use the same seed schedule and scenario matrix. | PR #1520 added the ego feature contract and same-seed config surfaces; PR #3769 added `validate_predictive_v2_comparison_readiness.py`, which checks variant completeness, provenance, ego-obstacle conditioning, and same-seed schedule metadata. | `preflighted_not_executed` |
| Ego-conditioned feature contract documented and preflighted. | PR #1520 added `configs/training/predictive/predictive_ego_features_contract_v1.yaml` and `docs/context/issue_1504_ego_feature_contract.md`; PR #3769 tests the readiness contract. | `met_for_contract` |
| Training and evaluation artifacts durable with manifests and checksums. | No four-way training/evaluation campaign has run under a gate-cleared hypothesis. PR #3833 and PR #4459 intentionally keep the decision packet `no_go` unless a future gate artifact includes durable evidence or provenance. | `blocked` |
| ADE/FDE and downstream navigation metrics reported separately. | PR #1549 recorded the obstacle-feature prerequisite audit and preserved forecast-vs-closed-loop interpretation boundaries. No new gate-cleared four-way result exists. | `partially_met_historical_only` |
| Hard-seed diagnostics included. | PR #1549 and PR #1901 record hard-seed outcomes for the prerequisite and coupling-gate evidence; both fail to justify old expansion. | `met_for_stop_boundary` |
| Missing artifact paths, sentinel-only rows, proxy-eval failures, fallback, degraded, and not-available rows explicit stage-gate outcomes. | PR #3769 added fail-closed readiness stages; PR #3833 added a decision packet; PR #4459 requires durable evidence or provenance before a `continue` gate can clear. | `met_for_readiness_gate` |
| A context note recommends continue, revise, or stop predictive planner v2. | PR #2287 added `docs/context/issue_2275_predictive_v2_fate.md`, selecting `stop_old_predictive_v2_expansion`; PR #2421 classified child issues under that decision. | `met_stop_old_expansion` |
| Parent-level acceptance: no routine four-way expansion without maintainer-selected revised hypothesis. | Issue comments on 2026-06-21 and 2026-07-04 reaffirm the issue is deferred behind the same-seed coupling gate, and merged readiness code enforces fail-closed behavior. | `met` |
| Parent-level acceptance: lane explicitly continued, revised, downgraded, or closed. | The lane is explicitly revised/deferred, not closed: it remains blocked until a maintainer-selected revised hypothesis plus provenance-backed `continue` artifact exists. | `blocked_not_closable` |

## Linked PR Evidence

| PR | Merge commit | Delivered evidence |
| --- | --- | --- |
| [#1520](https://github.com/ll7/robot_sf_ll7/pull/1520) | `6d0ffd21a09f87bf58dc0e614e195d33e5290ed0` | Ego-conditioned feature contract and same-seed config surfaces. |
| [#1549](https://github.com/ll7/robot_sf_ll7/pull/1549) | `7dcf7b263ddc3bd722190ff61e501a207a08d8dc` | Negative obstacle-transfer audit; closed-loop success did not improve. |
| [#1881](https://github.com/ll7/robot_sf_ll7/pull/1881) | `477fa07082ef166925696518a0bb6a604a034315` | Predictive coupling-gate preflight machinery. |
| [#1901](https://github.com/ll7/robot_sf_ll7/pull/1901) | `d0abb190d3aad78cc04a9391fb15a2b470a99789` | Recorded failed local closed-loop coupling-gate evidence. |
| [#2287](https://github.com/ll7/robot_sf_ll7/pull/2287) | `21b3632383df834409c63860efe6a5a9792fa7eb` | Predictive-v2 fate decision: stop old expansion, revise only through a new gate. |
| [#2421](https://github.com/ll7/robot_sf_ll7/pull/2421) | `fdf6e3059c6501dd2db95c47d05b96ed2c824750` | Child issue classification after the stop/revise decision. |
| [#3769](https://github.com/ll7/robot_sf_ll7/pull/3769) | `47ee331a531b698a594482dbd5c5125ba1a756c9` | Fail-closed readiness checker and CLI for the same-seed comparison. |
| [#3833](https://github.com/ll7/robot_sf_ll7/pull/3833) | `4243d354f872d2737399a5ebf7bdbcf8d3b77829` | Decision packet reporting `no_go` until a future gate clears. |
| [#4459](https://github.com/ll7/robot_sf_ll7/pull/4459) | `9637a504292db33db923612dcdc68208e1b50bb8` | Provenance hardening: a `continue` gate must include durable evidence or provenance. |

PR #1707 cross-referenced #1490 through the broader SLURM status ledger cleanup but did not change
the predictive-v2 acceptance contract.

## Residual Blocker

Closure would require one of these future actions:

1. A maintainer explicitly closes or downgrades #1490 as a historical stopped lane.
2. A maintainer selects a revised predictive-v2 hypothesis and supplies a provenance-backed
   closed-loop coupling-gate artifact that recommends `continue`; only then may the bounded
   same-seed comparison proceed.

Until one happens, no additional guardrail/checker slice is useful. The current smallest valid
state propagation is this closure audit, which preserves the criterion-to-evidence mapping and
prevents accidental stale execution.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_1490_closure_audit_2026-07-04/summary.json
test -f docs/context/issue_1490_closure_audit.md
test -f docs/context/evidence/issue_1490_closure_audit_2026-07-04/summary.json
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_1490_closure_audit.md \
  --path docs/context/evidence/issue_1490_closure_audit_2026-07-04/summary.json \
  --path docs/context/catalog.yaml \
  --path docs/context/evidence/README.md
git diff --check
```
