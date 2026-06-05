# Issue #2261 Static-Recenter Slice-Local Evidence

This directory contains compact reviewable evidence for Issue #2261. It summarizes why static
recentering should remain a slice-local diagnostic mechanism after the Issue #2221 held-out smoke.

Files:

- `summary.json`: validator-readable evidence summary with observed terminal parity, mechanism
  gates, scenario-fit interpretation, recommendation, and missing evidence.

Source evidence:

- `docs/context/issue_2180_one_factor_h500.md`
- `docs/context/issue_2221_static_recenter_transfer.md`
- `docs/context/issue_2266_static_recenter_activation.md`
- `docs/context/evidence/issue_2221_static_recenter_transfer_2026-06-04/comparison_summary.json`
- `docs/context/evidence/issue_2266_static_recenter_activation_2026-06-05/summary.json`
- `configs/policy_search/candidates/issue_2170_static_recenter_only.yaml`
- `configs/algos/hybrid_rule_v3_teb_like_rollout.yaml`
- `configs/scenarios/archetypes/classic_station_platform.yaml`
- `configs/scenarios/single/francis2023_intersection_wait.yaml`
- `robot_sf/planner/hybrid_rule_local_planner.py`

Claim boundary: analysis-only diagnostic evidence. The per-step activation trace is still missing.
