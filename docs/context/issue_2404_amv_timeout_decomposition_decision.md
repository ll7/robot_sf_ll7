# Issue #2404 AMV Timeout Decomposition Decision

Issue: [#2404](https://github.com/ll7/robot_sf_ll7/issues/2404)
Date: 2026-06-06
Status: current, diagnostic-only synthesis.

## Goal

Classify the AMV actuation-smoke timeout against the decomposition fields requested by Issue #2404,
while checking the related AMV timeout issues first so the repository does not rerun an already
instrumented trace slice unnecessarily.

Related issues checked:

- [#2259](issue_2259_amv_clipping_success_boundary.md) established that reduced clipping did not
  change success and kept the AMV actuation scorer diagnostic-only.
- [#2268](issue_2268_amv_timeout_decomposition.md) showed aggregate command-feasibility
  improvement with unchanged `timeout_low_progress` failures, but noted that per-step traces were
  still missing.
- [#2308](issue_2308_amv_timeout_trace_analysis.md) reran the same matched AMV smoke with per-step
  route-progress and command-feasibility fields and promoted the required trace block.

## Decision

Decision output: `feasibility_improved_but_route_blocked`.

Issue #2308 already executed the smallest relevant trace rerun after Issue #2268 on:

- Stage: `amv_actuation_smoke`
- Scenario: `classic_cross_trap_high`
- Seed: `101`
- Horizon: `80`
- Baseline: `hybrid_rule_v3_fast_progress`
- Intervention: `actuation_aware_hybrid_rule_v0`
- Synthetic actuation profile: `amv-actuation-stress-v0`

The rerun found that actuation-aware scoring reduced command clipping and early-window command gaps,
while both candidates still timed out about `3.6 m` from the goal. Yaw-rate saturation stayed zero,
average speed did not reveal a success-relevant slowdown, and final route progress was effectively
unchanged. The timeout driver therefore remains route/task progress, not persistent command
infeasibility.

## Field Coverage

The Issue #2404 requested fields are present in the #2308 promoted summary and are re-expressed in
the compact #2404 bundle:

| Requested field | Status | Evidence |
| --- | --- | --- |
| `progress_over_time` | produced | Windowed progress traces for both candidates; final route progress differed by only `-0.0321 m` for actuation-aware. |
| `clipping_over_time` | produced | Total clip steps fell from `22` to `15`; first-window clip fraction fell from `0.75` to `0.40`. |
| `saturation_over_time` | produced | Yaw-rate saturation stayed `0.0` for both candidates across all windows. |
| `command_speed_profile` | produced | Requested/applied linear-speed profiles are tracked per window and in aggregate. |
| `route_progress_blocked` | produced | Both rows ended `timeout_low_progress` and remained about `3.6 m` from goal. |
| `speed_cap_binding` | partially instrumented | Weak as primary: average speed was nearly unchanged and yaw saturation stayed zero, but late-episode speed-cap binding is not separately instrumented. |
| `scoring_too_conservative` | partially instrumented | Not selected as primary: lower requested/applied speeds did not create a success-relevant progress loss, but scoring conservativeness is not directly instrumented. |
| `unrelated_deadlock` | not instrumented | No deadlock/oscillation detector is present in the tracked trace; route/task progress is the best-supported driver from available fields. |

## Claim Boundary

This is analysis-only evidence over one synthetic AMV smoke row. It is not benchmark-strength,
calibrated AMV, hardware, safety, planner-ranking, or paper-facing evidence. It narrows the next
research direction: do not escalate broad AMV actuation scoring from this slice; investigate
route-progress geometry or horizon/task-completion blockers separately.

## Evidence

- Field-mapped #2404 summary:
  [evidence/issue_2404_amv_timeout_decomposition_2026-06-06/summary.json](evidence/issue_2404_amv_timeout_decomposition_2026-06-06/summary.json)
- Requested-field coverage table:
  [evidence/issue_2404_amv_timeout_decomposition_2026-06-06/decomposition_fields.csv](evidence/issue_2404_amv_timeout_decomposition_2026-06-06/decomposition_fields.csv)
- Decision-output table:
  [evidence/issue_2404_amv_timeout_decomposition_2026-06-06/decision_outputs.csv](evidence/issue_2404_amv_timeout_decomposition_2026-06-06/decision_outputs.csv)
- Source diagnostic summary:
  [evidence/issue_2308_amv_timeout_trace_2026-06-05/summary.json](evidence/issue_2308_amv_timeout_trace_2026-06-05/summary.json)

## Validation

Validated the promoted artifacts and links with:

```bash
uv run python -m json.tool docs/context/evidence/issue_2404_amv_timeout_decomposition_2026-06-06/summary.json
python - <<'PY'
import csv
from pathlib import Path
base = Path("docs/context/evidence/issue_2404_amv_timeout_decomposition_2026-06-06")
for name in ("decomposition_fields.csv", "decision_outputs.csv"):
    rows = list(csv.DictReader((base / name).open()))
    assert rows, name
PY
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_2404_amv_timeout_decomposition_decision.md \
  --path docs/context/evidence/issue_2404_amv_timeout_decomposition_2026-06-06/README.md \
  --path docs/context/evidence/issue_2404_amv_timeout_decomposition_2026-06-06/summary.json \
  --path docs/context/catalog.yaml
git diff --check
```
