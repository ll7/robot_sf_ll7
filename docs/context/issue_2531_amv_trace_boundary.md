# Issue #2531 AMV Trace Boundary Decision

Issue: [#2531](https://github.com/ll7/robot_sf_ll7/issues/2531)
Status: current decision; summary-timeline-only for the AMV actuation-aware timeout lane.

## Decision

For the AMV actuation-aware hybrid-rule timeout slice reviewed in
[issue_2443_amv_trace_review.md](issue_2443_amv_trace_review.md) and the #2522 AMV why-first
report, the durable evidence remains summary-timeline-only. Do not cite that lane as a trace-level
mechanism explanation until a new promoted `simulation_trace_export.v1` artifact exists for the
same candidates, scenario, and seed.

The existing #2405/#2428 loader-valid `simulation_trace_export.v1` artifacts belong to a different
AMMV Social Force diagnostic lane: `default_social_force` vs `ammv_social_force` on
`classic_head_on_corridor_low`, seed `111`. They are useful diagnostic trace-panel evidence, but
they do not provide frame/event IDs for the #2443 AMV actuation-aware hybrid-rule slice:
`hybrid_rule_v3_fast_progress` vs `actuation_aware_hybrid_rule_v0` on `classic_cross_trap_high`,
seed `101`.

## Evidence

| Surface | Current evidence | Boundary |
| --- | --- | --- |
| [issue_2443 summary](evidence/issue_2443_amv_trace_review_2026-06-07/summary.json) | `trace_export_paths` is empty, `frame_ids` and `event_ids` are empty, and `frame_event_id_status` is `blocked_not_in_compact_artifact`. | #2443 remains compact summary-timeline evidence. |
| [#2522 AMV why-first report](evidence/issue_2522_why_first_diagnostics/amv_actuation_why_first_report.md) | The report cites #2308/#2404/#2443 summaries and records raw frame/event IDs as unavailable. | Why-first AMV interpretation must stay summary-timeline-only. |
| [#2428 trace exports](evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/ammv_social_force_trace_export.json) | Loader-valid `simulation_trace_export.v1` files exist for the AMMV Social Force lane, with 20 frames per selected trace and planner `ammv` metadata. | Separate diagnostic trace-panel lane; not a trace proof for the #2443 actuation-aware hybrid-rule row. |

## Claim Boundary

The #2443/#2522 AMV actuation result can support the diagnostic statement that command feasibility
improved while route/task progress stayed blocked. It cannot support frame-level or event-level
mechanism claims, benchmark-strength claims, paper-facing claims, calibrated AMV claims, or planner
ranking claims.

The #2405/#2428 AMMV Social Force exports can still be cited as diagnostic trace-panel evidence for
that separate lane, with their own limitations.

## Validation

```bash
uv run python -m json.tool docs/context/evidence/issue_2443_amv_trace_review_2026-06-07/summary.json
uv run python - <<'PY'
from pathlib import Path
from robot_sf.analysis_workbench.simulation_trace_export import load_simulation_trace_export

for path in [
    Path("docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/default_social_force_trace_export.json"),
    Path("docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/ammv_social_force_trace_export.json"),
]:
    trace = load_simulation_trace_export(path).to_dict()
    assert trace["schema_version"] == "simulation_trace_export.v1"
    assert trace["source"]["scenario_id"] == "classic_head_on_corridor_low"
    assert trace["source"]["seed"] == 111
    assert len(trace["frames"]) == 20
PY
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
