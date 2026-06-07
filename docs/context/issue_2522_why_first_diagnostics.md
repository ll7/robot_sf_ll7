# Issue #2522 Why-First Diagnostics

Issue: [#2522](https://github.com/ll7/robot_sf_ll7/issues/2522)
Status: current analysis-tool application; not new benchmark evidence.

## Purpose

This note records the Issue #2522 why-first report generation pass for two active diagnostic
threads:

- topology near-parity selection, using the Issue #2518 diagnostic acceptance and Issue #2530
  corrective-behavior smoke;
- AMV actuation feasibility, using the Issue #2308/#2404 timeout decomposition and Issue #2443
  matched clipping-vs-progress trace review.

The generated reports convert compact existing evidence into a consistent why-first narrative. They
do not add runtime evidence, change benchmark rows, or promote either mechanism to a planner
improvement claim.

## Generated Reports

| Diagnostic | Compact input | Generated report | Decision |
| --- | --- | --- | --- |
| Topology near-parity corrective diagnostics | [topology_near_parity_input.json](evidence/issue_2522_why_first_diagnostics/topology_near_parity_input.json) | [topology_near_parity_why_first_report.md](evidence/issue_2522_why_first_diagnostics/topology_near_parity_why_first_report.md) | `revise`: near-parity gating is a real diagnostic signal, but the corrective-behavior smoke still exhausted the horizon. |
| AMV actuation timeout diagnostics | [amv_actuation_input.json](evidence/issue_2522_why_first_diagnostics/amv_actuation_input.json) | [amv_actuation_why_first_report.md](evidence/issue_2522_why_first_diagnostics/amv_actuation_why_first_report.md) | `revise`: command clipping improved, but route progress and success did not move on the matched slice. |

## Interpretation

Topology remains diagnostic-only. The Issue #2518 near-parity gate moved route selection and local
command influence away from the primary route, but Issue #2530 still ended
`horizon_exhausted`. The report therefore routes the next proof to the Issue #2563 primary-route
reuse-penalty hypothesis before any benchmark, transfer, or leaderboard claim.

AMV actuation-aware scoring remains a feasibility diagnostic rather than an improvement mechanism.
The matched trace review reduced command clipping from 22 to 15 steps, but both candidates timed out
with nearly identical final route progress. The report therefore routes follow-up work toward
route-progress geometry or task-completion blockers before another broad actuation-aware scorer.

## Claim Boundary

This is an analysis/reporting pass over already tracked compact evidence. It should be cited as a
why-first interpretation aid, not as an additional benchmark run, trace export, metric proof, or
paper-facing result. The report generator itself states that each report strength is limited to the
compact input evidence.

## Validation

```bash
rtk uv run python scripts/tools/generate_why_first_report.py --input docs/context/evidence/issue_2522_why_first_diagnostics/topology_near_parity_input.json --output /tmp/topology_near_parity_why_first_report.gen.md
rtk diff -u /tmp/topology_near_parity_why_first_report.gen.md docs/context/evidence/issue_2522_why_first_diagnostics/topology_near_parity_why_first_report.md
rtk uv run python scripts/tools/generate_why_first_report.py --input docs/context/evidence/issue_2522_why_first_diagnostics/amv_actuation_input.json --output /tmp/amv_actuation_why_first_report.gen.md
rtk diff -u /tmp/amv_actuation_why_first_report.gen.md docs/context/evidence/issue_2522_why_first_diagnostics/amv_actuation_why_first_report.md
rtk uv run pytest tests/tools/test_generate_why_first_report.py -q
rtk uv run python -m json.tool docs/context/evidence/issue_2522_why_first_diagnostics/topology_near_parity_input.json
rtk uv run python -m json.tool docs/context/evidence/issue_2522_why_first_diagnostics/amv_actuation_input.json
rtk uv run python -m json.tool docs/context/evidence/issue_2522_why_first_diagnostics/summary.json
rtk uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
BASE_REF=origin/main rtk scripts/dev/check_docs_proof_consistency_diff.sh
rtk git diff --check
```
