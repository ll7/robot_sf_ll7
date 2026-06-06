# Issue #2259 AMV Clipping Versus Success Boundary

This directory contains compact reviewable evidence for Issue #2259. It synthesizes the Issue
Issue #2224 ranking diagnostic and Issue #2268 timeout decomposition into a parent-lane
recommendation.

Files:

- `summary.json`: validator-readable summary of the feasibility-vs-success split, classification,
  missing trace evidence, and claim boundary.

Source evidence:

- `docs/context/issue_2224_amv_actuation_ranking.md`
- `docs/context/issue_2268_amv_timeout_decomposition.md`
- `docs/context/issue_2230_amv_actuation_evidence_ladder.md`
- `docs/context/evidence/issue_2224_amv_actuation_ranking_2026-06-04/comparison.json`
- `docs/context/evidence/issue_2268_amv_timeout_decomposition_2026-06-05/summary.json`
- `docs/context/evidence/issue_2268_amv_timeout_decomposition_2026-06-05/timeout_decomposition.csv`

Claim boundary: synthetic diagnostic evidence only. It does not move calibrated AMV actuation out
of `blocked`.
