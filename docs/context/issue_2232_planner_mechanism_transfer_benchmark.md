# Issue #2232 Planner Mechanism Transfer Benchmark Protocol

Issue: [#2232](https://github.com/ll7/robot_sf_ll7/issues/2232)
Status: proposal/synthesis protocol, not benchmark evidence.
Date: 2026-06-04

## Goal

Define a conservative protocol for testing whether a planner mechanism discovered on one Robot SF
slice transfers to other scenario families, seeds, perturbations, observation tracks, kinematic
models, or AMV actuation envelopes. This note extends the held-out scenario-family transfer
contract in [issue_2128_heldout_scenario_family_transfer_protocol.md](issue_2128_heldout_scenario_family_transfer_protocol.md)
from whole planners to isolated planner mechanisms.

The protocol answers a narrow research question:

> Can Robot SF distinguish a transferable local-navigation mechanism from a slice-local fix?

No transfer claim is established until a follow-up benchmark executes this protocol, promotes
compact durable evidence, and passes the controls below.

## Claim Boundary

Allowed before execution:

- "planned planner-mechanism transfer protocol";
- "proposal for separating slice-local component effects from transfer evidence";
- "candidate mechanism selected for a follow-up transfer test".

Allowed after a valid follow-up run:

- "observed transfer behavior for the named mechanism, axes, seeds, planners, and metrics";
- "diagnostic mechanism-transfer evidence on the selected held-out slice";
- "transfer-supported on this protocol slice" only when the support rule below passes.

Disallowed:

- "OOD generalization", "real-world transfer", or "paper-facing mechanism causality" from this
  proposal alone;
- "transfer success" from fallback, degraded, failed, or not-available rows;
- causal mechanism claims from aggregate metrics without a matched ablation and, when needed,
  trace-backed explanation.

## Transfer Axes

A follow-up mechanism-transfer run should change exactly one primary axis at a time unless the
issue contract explicitly asks for an interaction test.

| Axis | Transfer question | Minimum control |
| --- | --- | --- |
| Scenario family | Does the mechanism help outside the discovery family? | Freeze training/tuning/discovery families separately from held-out families. |
| Seed set | Does the effect survive a new seed schedule? | Use the same new seed set for base and mechanism rows. |
| Perturbation family | Does the mechanism survive route, timing, density, or speed perturbations? | Keep perturbation magnitudes and paired baseline rows fixed. |
| Observation track | Does the mechanism depend on privileged or diagnostic-only observations? | Name the observation contract and exclude unavailable tracks. |
| Kinematic model | Does the mechanism survive holonomic/nonholonomic or AMV-style dynamics? | Keep planner base and command adapter constant except for the kinematic axis. |
| AMV actuation envelope | Does the mechanism survive command limits, lag, or update-rate stress? | Use the same actuation envelope and reporting metrics for base and mechanism rows. |

## Controls

Every valid mechanism-transfer row set must include:

- one base planner or candidate and one isolated mechanism toggle;
- identical scenario IDs, seeds, horizons, metrics, and availability policy across compared rows;
- a frozen discovery slice and frozen held-out slice before results are inspected;
- native execution, or an explicitly documented adapter mode whose adapter impact is not the
  mechanism under test;
- visible fallback, degraded, failed, and not-available rows, excluded from transfer support;
- no planner hyperparameter tuning, threshold tuning, reward tuning, checkpoint selection, or report
  design using the held-out slice;
- compact durable artifacts: source configs, row summaries, transfer deltas, leakage audit, and
  checksums or external artifact pointers.

## Minimum Evidence

| Evidence tier | Required evidence | Permitted conclusion |
| --- | --- | --- |
| `proposal` | This note plus referenced configs or source notes. | A follow-up can be scoped. |
| `diagnostic_transfer_smoke` | One mechanism, one held-out axis, matched base/toggle rows, visible row-status accounting. | Directional transfer behavior for that slice only. |
| `transfer_supported` | Native or eligible adapter rows show a consistent beneficial delta on the held-out slice without worsening the named safety gate, and the discovery-slice effect is not contradicted. | Mechanism transfer is supported for the named axis and slice. |
| `slice_local` | Discovery-slice benefit disappears or reverses on the held-out slice without a compensating safety benefit. | Mechanism should be treated as local to the discovery slice. |
| `trade_off` | Held-out slice improves one target metric while worsening a named safety, success, runtime, or clearance gate. | Mechanism needs a decision rule before promotion. |
| `inconclusive` | Rows are underpowered, mixed across seeds/families, unavailable, degraded, or missing required controls. | Do not update mechanism support; refine the proof step. |

## First Candidate

Use **static recentering** as the first mechanism-transfer candidate.

Rationale:

- [issue_2182_component_effect_synthesis.md](issue_2182_component_effect_synthesis.md) classifies
  `static_recenter_only_minus_base` and `escape_recenter_pair_minus_static_escape_only` as supported
  on the Issue #2180 h500 one-factor slice.
- [issue_2180_one_factor_h500.md](issue_2180_one_factor_h500.md) reports native local execution
  with zero failed jobs and directional success/average-speed gains without collision or near-miss
  penalties on the 18-row slice.
- Static escape alone was neutral, so recentering is a cleaner mechanism than the grouped static
  escape/recenter candidate.
- The claim boundary is still local diagnostic evidence: one-row and two-row success deltas over
  the six-scenario, three-seed h500 slice are not paper-facing causality.

Recommended first follow-up: issue
[#2221](https://github.com/ll7/robot_sf_ll7/issues/2221), the existing static-recentering transfer
benchmark issue.

- base row: the same hybrid-rule base used by the Issue #2170/#2180 one-factor manifest;
- mechanism row: static recentering enabled without adding static escape, corridor-transit terms, or
  selector behavior;
- transfer axis: scenario family, using the Issue #2128 held-out-family partition contract as the
  leakage-audit template;
- decision rule: classify `transfer_supported` only if the held-out slice preserves a beneficial
  success or low-progress delta without worsening collision or near-miss rate, and all comparable
  rows are native or explicitly eligible adapter rows.

## Reporting Contract

The follow-up PR or issue should include the research-result template from
[goal_driven_agent_loops_2026-05-13.md](goal_driven_agent_loops_2026-05-13.md):

- Target: whether static recentering transfers beyond the Issue #2180 discovery slice.
- Baseline/Comparator: matched hybrid-rule base row with the same planner base and no recentering.
- Evidence Tier And Claim Boundary: planned `diagnostic_transfer_smoke`; no paper-facing claim.
- Decision/Stop Rule: classify `transfer_supported`, `slice_local`, `trade_off`, or
  `inconclusive` using the table above.
- Evidence And Provenance Plan: configs, row summaries, row-status table, transfer deltas, leakage
  audit, and durable artifact pointers.
- Result Classification: one of the mechanism-transfer classes above.
- Update Surface: this note, the parent issue, and any mechanism dashboard or benchmark evidence map
  that tracks the follow-up.

## Follow-Up Boundary

This issue should close with the protocol only. The next smallest executable issue is a bounded
static-recentering transfer smoke that freezes the held-out scenario-family slice before execution;
that follow-up already exists as [#2221](https://github.com/ll7/robot_sf_ll7/issues/2221). Issue
[#2220](https://github.com/ll7/robot_sf_ll7/issues/2220) remains the better home for a broader
failure-mechanism taxonomy; do not duplicate that taxonomy here beyond the mechanism labels needed
for transfer interpretation.

## Validation

Docs-only validation for this protocol should include:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```

Also verify that every referenced note and config exists before opening the PR.
