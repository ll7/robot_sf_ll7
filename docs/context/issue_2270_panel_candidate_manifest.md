# Issue #2270 Panel Candidate Manifest

Issue: [#2270](https://github.com/ll7/robot_sf_ll7/issues/2270)
Parent issue: [#2227](https://github.com/ll7/robot_sf_ll7/issues/2227)
Date: 2026-06-05
Status: analysis-only candidate manifest; panel rendering remains blocked on durable trace exports.

## Goal

Locate durable trace-pair candidates for the Issue #2227 mechanism panels before rendering any
visual artifacts. The requested panels are diagnostic-only and should explain static-recentering
non-transfer and topology primary-route-only behavior without promoting aggregate summaries as
visual trace evidence.

## Result

Two useful panel candidates are identifiable from the current tracked evidence, but neither is
panel-ready:

- static-recentering held-out non-transfer:
  `hybrid_rule_v3_fast_progress` versus `issue_2170_static_recenter_only` on
  `classic_station_platform_medium`, seed `111`, horizon `500`;
- topology primary-route-only behavior:
  `hybrid_rule_v3_fast_progress` versus `topology_guided_hybrid_rule_v0` on
  `classic_realworld_double_bottleneck_high`, seed `111`, horizon `160`.

Both candidates are blocked because the repository has compact summaries, inventories, and score
examples, but not matched mechanism-specific `simulation_trace_export.v1` trace pairs for the
baseline/intervention episodes. This preserves the Issue #2227 boundary: do not render generic or
decorative panels from unrelated fixtures.

## Manifest

The candidate manifest is tracked at:

- [evidence/issue_2270_panel_candidate_manifest_2026-06-05/panel_candidate_manifest.yaml](evidence/issue_2270_panel_candidate_manifest_2026-06-05/panel_candidate_manifest.yaml)
- [evidence/issue_2270_panel_candidate_manifest_2026-06-05/README.md](evidence/issue_2270_panel_candidate_manifest_2026-06-05/README.md)

The manifest records each candidate's source evidence, required trace exports, missing fields, and
recommended next command shape.

## Evidence Inputs

- Parent artifact-gap audit:
  [issue_2227_mechanism_panels.md](issue_2227_mechanism_panels.md)
- Parent gap manifest:
  [evidence/issue_2227_mechanism_panels_2026-06-04/artifact_gap_manifest.json](evidence/issue_2227_mechanism_panels_2026-06-04/artifact_gap_manifest.json)
- Static-recentering terminal outcome smoke:
  [issue_2221_static_recenter_transfer.md](issue_2221_static_recenter_transfer.md)
- Static-recentering activation gap:
  [issue_2266_static_recenter_activation.md](issue_2266_static_recenter_activation.md)
- Topology primary-route audit:
  [issue_2258_topology_primary_route_audit.md](issue_2258_topology_primary_route_audit.md)
- Topology selection instrumentation:
  [issue_2282_topology_selection_instrumentation.md](issue_2282_topology_selection_instrumentation.md)

## Follow-Up Boundary

The next child should generate or export the missing baseline/intervention
`simulation_trace_export.v1` traces for the selected candidate rows, then run
`scripts/tools/render_trajectory_panels.py` with an explicit selection CSV. Until that exists,
Issue #2227 remains a trace-input readiness lane, not panel evidence and not paper-facing mechanism
proof.

## Validation

Reference and consistency checks:

```bash
rtk rg -l 'simulation_trace_export\.v1' docs tests
rtk bash scripts/dev/check_docs_proof_consistency_diff.sh
rtk git diff --check
```

The recursive `simulation_trace_export.v1` search found only generic fixtures and schema/tooling
references, plus the prior Issue #2227 artifact-gap note. It did not find tracked mechanism-specific
trace pairs for the #2270 candidates.
