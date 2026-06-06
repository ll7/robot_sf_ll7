# Issue #2402 Static-Recenter Activation Decision

Issue: [#2402](https://github.com/ll7/robot_sf_ll7/issues/2402)
Date: 2026-06-06
Status: current, diagnostic-only synthesis.

## Goal

Classify the static-recenter held-out smoke against the activation fields requested by Issue #2402,
while checking the related static-recenter issues first so the repository does not rerun an already
instrumented diagnostic slice unnecessarily.

Related issues checked:

- [#2261](issue_2261_static_recenter_slice_local.md) preserved the slice-local conclusion and
  identified the missing activation trace boundary.
- [#2266](issue_2266_static_recenter_activation.md) showed that the tracked terminal outcomes were
  identical but activation instrumentation was missing.
- [#2306](issue_2306_static_recenter_activation_trace.md) reran the same held-out smoke with
  activation-level instrumentation and promoted the required field block.

## Decision

Decision outputs:

- `mechanism_inactive` for the unsolved `classic_station_platform_medium` row.
- `comparator_already_solved_case` for the `francis2023_intersection_wait` row.

Issue #2306 already executed the smallest relevant instrumented rerun after Issue #2266 on:

- Baseline: `hybrid_rule_v3_fast_progress`
- Mechanism candidate: `issue_2170_static_recenter_only`
- Scenario matrix: `configs/scenarios/sets/issue_2128_heldout_family_transfer_pilot_eval.yaml`
- Seed: `111`
- Horizon: `500`

The rerun found zero static-recenter activations on both held-out rows. On the unsolved station
platform row, terminal outcome parity therefore comes from non-activation rather than active-but-
irrelevant behavior. On the intersection-wait row, both baseline and mechanism candidate already
succeeded, so the row cannot demonstrate a recentering rescue.

## Field Coverage

The Issue #2402 requested fields are present in the #2306 promoted summary and are re-expressed in
the compact #2402 bundle:

| Requested field | Status | Evidence |
| --- | --- | --- |
| `activation_count` | produced | `0` on both held-out rows. |
| `first_activation_step` | produced | `null` on both rows because activation count was zero. |
| `selected_command_source` | produced | `[]` on both rows because no static-recenter command was selected. |
| `command_source_changed` | produced | `false` on both rows. |
| `progress_delta_after_activation` | produced | `null` on both rows because no activation occurred. |
| `trajectory_delta` | produced | `0.0 m` on both rows. |
| `terminal_outcome_changed` | produced | `false` on both rows. |

## Claim Boundary

This is analysis-only evidence. It does not show a static-recenter planner improvement, held-out
transfer, benchmark mitigation, or paper-facing result. It narrows the research direction: the
current held-out static-recenter smoke is an inactive-mechanism negative for the unsolved row, so
future work should either predeclare a slice where the recenter gate should activate or prioritize
other route-progress mechanisms.

## Evidence

- Field-mapped #2402 summary:
  [evidence/issue_2402_static_recenter_activation_2026-06-06/summary.json](evidence/issue_2402_static_recenter_activation_2026-06-06/summary.json)
- Requested-field coverage table:
  [evidence/issue_2402_static_recenter_activation_2026-06-06/activation_fields.csv](evidence/issue_2402_static_recenter_activation_2026-06-06/activation_fields.csv)
- Decision-output table:
  [evidence/issue_2402_static_recenter_activation_2026-06-06/decision_outputs.csv](evidence/issue_2402_static_recenter_activation_2026-06-06/decision_outputs.csv)
- Source diagnostic summary:
  [evidence/issue_2306_static_recenter_activation_trace_2026-06-05/summary.json](evidence/issue_2306_static_recenter_activation_trace_2026-06-05/summary.json)

## Validation

Validated the promoted artifacts and links with:

```bash
uv run python -m json.tool docs/context/evidence/issue_2402_static_recenter_activation_2026-06-06/summary.json
python - <<'PY'
import csv
from pathlib import Path
base = Path("docs/context/evidence/issue_2402_static_recenter_activation_2026-06-06")
for name in ("activation_fields.csv", "decision_outputs.csv"):
    rows = list(csv.DictReader((base / name).open()))
    assert rows, name
PY
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_2402_static_recenter_activation_decision.md \
  --path docs/context/evidence/issue_2402_static_recenter_activation_2026-06-06/README.md \
  --path docs/context/evidence/issue_2402_static_recenter_activation_2026-06-06/summary.json \
  --path docs/context/catalog.yaml
git diff --check
```
