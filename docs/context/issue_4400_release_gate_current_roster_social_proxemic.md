# Issue #4400 Release-Gate Current-Roster Social-Proxemic Pre-Registration

This note pre-registers the fresh current-roster release-gate campaign needed to make clearance and
personal-space gates evaluable. This is a configuration contract only: no campaign is submitted, no
release gate is recomputed, and no release approval claim is made.

## Purpose

Issue #4313 reported release-gate rows as `not_evaluable` because fields were not present in the
retained camera-ready packet. Issue #4331/#4334 reporting work preserves `min_clearance_m` and
`proxemic_intrusion_rate` when fresh episode rows contain source metrics. Issue #4400 config
declares that a fresh campaign should produce those rows with the social-proxemic metric group
explicitly enabled.

## Registered Inputs

- Campaign config: `configs/benchmarks/release_gate_current_roster_social_proxemic_issue_4400.yaml`
- Release-gate specification: `configs/benchmarks/release_gates/camera_ready_current_roster_gates.yaml`
- Retained baseline packet, intentionally unchanged:
  `docs/context/evidence/camera_ready_all_planners_2026-05-04/reports/campaign_summary.json`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Roster source: current 8-planner `camera_ready_all_planners` surface.

## Claim Boundary

The config supports gate evaluability only. It does not backfill the retained 2026-05-04 packet,
does not change release-gate thresholds, does not interpret a report, and does not certify a
release. The next empirical action is private-side campaign queueing followed by rerunning the
issue #4313 evaluator on the retrieved fresh rows.

## Closure Audit

The public repository criteria are mapped to merged PR #4419 in
[`docs/context/evidence/issue_4400_closure_audit_2026-07-04.md`](evidence/issue_4400_closure_audit_2026-07-04.md).
That audit records the remaining private-side campaign queueing and evaluator rerun as outside the
repository-only evidence boundary.
