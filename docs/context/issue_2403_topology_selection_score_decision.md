# Issue #2403 Topology Selection-Score Decision

Issue: [#2403](https://github.com/ll7/robot_sf_ll7/issues/2403)
Date: 2026-06-06
Status: current, diagnostic-only synthesis.

## Goal

Classify the topology primary-route diagnostic against the explicit selection-score fields requested
in Issue #2403, while checking the related topology issues first so the repository does not rerun an
already promoted diagnostic slice unnecessarily.

Related issues checked:

- [#2258](issue_2258_topology_primary_route_audit.md) established that alternatives were present,
  but the compact trace still lacked per-hypothesis score and rejection detail.
- [#2282](issue_2282_topology_selection_instrumentation.md) added score components, rank, margin,
  outcome, and rejection instrumentation.
- [#2307](issue_2307_topology_score_diagnostic.md) reran the same topology diagnostic slice after
  Issue #2282 and promoted the required field block.
- [#2393](issue_2393_topology_selection_preflight.md) used #2307 to name the next near-parity
  diversity-gate revision candidate.

## Decision

Decision output: `primary_route_overselected`.

The strongest current classification is score-surface over-selection of `primary_route`, not absent
alternatives, invalid alternatives, or topology irrelevance for the diagnostic case. Issue #2307
already executed the smallest relevant rerun after score instrumentation on:

- Candidate: `topology_guided_hybrid_rule_v0`
- Scenario: `classic_realworld_double_bottleneck_high`
- Seed: `111`
- Horizon: `160`
- Stage: `full_matrix`

The rerun found 98 scored alternative opportunities. Every rejected scored alternative used
`lower_topology_selection_score`, and the only non-primary route-selector choice was a numerical tie
that did not become topology-command influence. Topology-command wins selected `primary_route` 33
times and never switched to a non-primary hypothesis.

## Field Coverage

The Issue #2403 requested fields are present in the #2307 promoted summary and are re-expressed in
the compact #2403 bundle:

| Requested field | Status | Evidence |
| --- | --- | --- |
| `per_frame_hypothesis_count` | produced | 70 insufficient-hypothesis frames and 90 ok frames. |
| `alternative_hypothesis_count` | produced | 98 steps with scored alternative hypotheses. |
| `selected_hypothesis` | produced | Route selector: 97 `primary_route`, 1 `masked_cell_87_79` numerical tie; topology command influence: 33 `primary_route`. |
| `score_margin_to_primary_route` | produced | 98 alternative margins; median 0.4056854309944198, max 8.214213684774414. |
| `rejection_reason` | produced | 98 `lower_topology_selection_score` rejected alternatives. |
| `switch_opportunity_count` | produced | 98 switch opportunities; 0 real hypothesis switches. |

## Claim Boundary

This is analysis-only evidence. It does not show a topology-guided planner improvement, benchmark
mitigation, or paper-facing performance result. It only narrows the next research direction:
revise the upstream topology hypothesis scoring/selection surface before retuning downstream command
selection.

## Evidence

- Field-mapped #2403 summary:
  [evidence/issue_2403_topology_selection_score_2026-06-06/summary.json](evidence/issue_2403_topology_selection_score_2026-06-06/summary.json)
- Requested-field coverage table:
  [evidence/issue_2403_topology_selection_score_2026-06-06/field_coverage.csv](evidence/issue_2403_topology_selection_score_2026-06-06/field_coverage.csv)
- Decision-output table:
  [evidence/issue_2403_topology_selection_score_2026-06-06/decision_outputs.csv](evidence/issue_2403_topology_selection_score_2026-06-06/decision_outputs.csv)
- Source diagnostic summary:
  [evidence/issue_2307_topology_score_diagnostic_2026-06-05/summary.json](evidence/issue_2307_topology_score_diagnostic_2026-06-05/summary.json)

## Validation

Validated the promoted artifacts and links with:

```bash
uv run python -m json.tool docs/context/evidence/issue_2403_topology_selection_score_2026-06-06/summary.json
python - <<'PY'
import csv
from pathlib import Path
for name in ("field_coverage.csv", "decision_outputs.csv"):
    rows = list(csv.DictReader(
        (Path("docs/context/evidence/issue_2403_topology_selection_score_2026-06-06") / name).open()
    ))
    assert rows, name
PY
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_2403_topology_selection_score_decision.md \
  --path docs/context/evidence/issue_2403_topology_selection_score_2026-06-06/README.md \
  --path docs/context/evidence/issue_2403_topology_selection_score_2026-06-06/summary.json \
  --path docs/context/catalog.yaml
git diff --check
```
