# Issue #2443 AMV Actuation Trace Review

Issue: [#2443](https://github.com/ll7/robot_sf_ll7/issues/2443)
Status: analysis-only trace review from compact tracked summaries.

## Question

For the matched AMV actuation baseline/intervention pair, did reduced command clipping fail to
improve success because route progress stayed blocked, because the speed envelope remained binding,
or because the navigation deadlock was unrelated to actuation feasibility?

## Selected Pair

Source: [Issue #2308 AMV timeout trace summary](evidence/issue_2308_amv_timeout_trace_2026-06-05/summary.json)
and [Issue #2404 decomposition decision](evidence/issue_2404_amv_timeout_decomposition_2026-06-06/summary.json).

| Field | Value |
| --- | --- |
| Scenario | `classic_cross_trap_high` |
| Seed | `101` |
| Stage | `amv_actuation_smoke` |
| Horizon | `80` |
| Baseline | `hybrid_rule_v3_fast_progress` |
| Intervention | `actuation_aware_hybrid_rule_v0` |
| Synthetic actuation profile | `amv-actuation-stress-v0` |

## Timeline

| Candidate | Window | Progress delta (m) | Clip fraction | Yaw saturation |
| --- | ---: | ---: | ---: | ---: |
| `hybrid_rule_v3_fast_progress` | `0-19` | `2.8884` | `0.75` | `0.0` |
| `hybrid_rule_v3_fast_progress` | `20-39` | `3.6421` | `0.10` | `0.0` |
| `hybrid_rule_v3_fast_progress` | `40-59` | `3.7380` | `0.15` | `0.0` |
| `hybrid_rule_v3_fast_progress` | `60-79` | `2.0724` | `0.10` | `0.0` |
| `actuation_aware_hybrid_rule_v0` | `0-19` | `2.9276` | `0.40` | `0.0` |
| `actuation_aware_hybrid_rule_v0` | `20-39` | `3.2348` | `0.20` | `0.0` |
| `actuation_aware_hybrid_rule_v0` | `40-59` | `3.6002` | `0.00` | `0.0` |
| `actuation_aware_hybrid_rule_v0` | `60-79` | `2.5209` | `0.15` | `0.0` |

## Classification

Result: `feasibility_improved_but_route_blocked`.

The actuation-aware row reduced command clipping from `22` to `15` steps and reduced the first
window clip fraction from `0.75` to `0.40`, but both candidates still ended
`timeout_low_progress`. Final route progress was effectively unchanged: the actuation-aware row
finished with `-0.0321 m` less final route progress and `+0.0321 m` more distance to goal than the
baseline. Average speed changed by only `+0.0047 m/s`, and yaw saturation stayed `0.0` for both
rows, so speed-cap binding is weak as the primary explanation.

The best supported driver is route/task progress remaining insufficient after command feasibility
improved. Scoring conservativeness remains only partially instrumented; unrelated deadlock is not
instrumented because the tracked summary has no deadlock or oscillation detector.

## Frame/Event Boundary

The durable source artifacts are compact summaries, not raw `simulation_trace_export.v1` files. The
review therefore leaves `trace_export_paths` empty and records
`trace_export_status: raw_simulation_trace_export_v1_unavailable` in
[summary.json](evidence/issue_2443_amv_trace_review_2026-06-07/summary.json). The compact summaries
preserve candidate IDs and 20-step window labels, but not raw frame IDs or step-event IDs. The
Issue #2443 frame/event requirement therefore fails closed as `blocked_not_in_compact_artifact`.

## Recommendation

Keep `actuation_aware_hybrid_rule_v0` diagnostic-only on this slice. Do not propose another broad
actuation-aware planner variant from this result. The next useful research direction is a
route-progress geometry or task-completion blocker analysis, or a raw trace-export artifact that
preserves frame/event IDs when a publication-style trace panel is needed.

## Claim Boundary

This is analysis-only evidence over one synthetic AMV smoke pair. It is not benchmark-strength,
calibrated AMV, hardware, safety, planner-ranking, or paper-facing evidence.

## Validation

```bash
uv run python -m json.tool docs/context/evidence/issue_2443_amv_trace_review_2026-06-07/summary.json
python - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(Path("docs/context/evidence/issue_2443_amv_trace_review_2026-06-07/progress_clipping_timeline.csv").open()))
assert len(rows) == 8
assert {row["candidate"] for row in rows} == {"hybrid_rule_v3_fast_progress", "actuation_aware_hybrid_rule_v0"}
PY
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/issue_2443_amv_trace_review.md
git diff --check
```
