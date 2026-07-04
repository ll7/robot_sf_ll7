# Issue #4400 Release-Gate Current-Roster Social-Proxemic Pre-Registration

This note pre-registers the fresh current-roster release-gate campaign needed to
make clearance and personal-space gates evaluable. It is a configuration
contract only: no campaign has been submitted, no release gate has been
recomputed, and no release approval claim is made.

## Purpose

Issue #4313 reported release-gate rows as `not_evaluable` for fields that were
not present in the retained camera-ready packet. The issue #4331/#4334 reporting
work preserves `min_clearance_m` and `proxemic_intrusion_rate` when fresh episode
rows contain the source metrics. The issue #4400 config declares the fresh
campaign that should produce those rows with the social-proxemic metric group
explicitly enabled.

## Registered Inputs

- Campaign config:
  `configs/benchmarks/release_gate_current_roster_social_proxemic_issue_4400.yaml`
- Release-gate specification:
  `configs/benchmarks/release_gates/camera_ready_current_roster_gates.yaml`
- Retained baseline packet, intentionally unchanged:
  `docs/context/evidence/camera_ready_all_planners_2026-05-04/reports/campaign_summary.json`
- Scenario matrix:
  `configs/scenarios/classic_interactions_francis2023.yaml`
- Roster source:
  the current 8-planner `camera_ready_all_planners` surface.

## Claim Boundary

The config supports gate evaluability only. It does not backfill the retained
2026-05-04 packet, does not change release-gate thresholds, does not interpret a
report, and does not certify a release. The next empirical action after merge is
private-side campaign queueing followed by rerunning the issue #4313 evaluator
on retrieved fresh rows.
