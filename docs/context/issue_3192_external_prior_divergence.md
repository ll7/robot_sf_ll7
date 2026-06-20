# Issue #3192 External-Prior Divergence De-risk

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/3192>

## Claim Boundary (Read First)

- **Analysis-only / `inconclusive` current state.** This adds the *machinery* to compare authored
  priors against published external-dataset statistics, plus an honest current verdict. It is **not**
  a realism claim, not planner ranking, and not a substitute for a raw-data pilot.
- No licensed data is downloaded or redistributed. Comparison uses only *published, citable summary
  statistics*, and none have been sourced yet — so the current verdict for every dataset is
  `inconclusive-need-pilot` by construction (fail-closed).

## What This Adds

- `scripts/tools/external_prior_divergence.py` — computes per-statistic divergence between authored
  priors and published reference statistics, and assigns each dataset one of three canonical
  verdicts: `priors-sufficient-for-diagnostic`, `raw-data-materially-changes-scope`,
  `inconclusive-need-pilot`. A statistic is only `comparable` when BOTH a cited reference value and
  an authored value exist; uncited reference values are reported `not-comparable` and never treated
  as agreement.
- `configs/research/external_prior_reference_stats.yaml` — reference scaffold for SDD, ETH/UCY, and
  AMV command-response statistics. All values are `null` with `source_citation: NEEDS_CITATION`
  until sourced from real papers.
- `configs/research/authored_prior_summary_stats.yaml` — authored-side scaffold keyed identically,
  to be populated from tracked scenario parameter bounds.
- `tests/research/test_external_prior_divergence.py` — proves the verdict logic and the honesty
  contract (uncited / missing values → not-comparable → inconclusive).

## Current Verdict (this PR)

Running the tool on the shipped scaffolds yields `inconclusive-need-pilot` for **sdd**,
**eth_ucy_family**, and **amv_command_response** (0 comparable statistics each), because no published
reference statistic has been cited yet. Report:
`docs/context/evidence/issue_3192_external_prior_divergence/current_verdict.json`.

## Decision Routing

Until a cited-sources pass populates the reference manifest, the de-risk decision stands as:
**diagnostic claims proceed on authored + repository-trace priors; licensed-data calibration is
deferred.** This is the input the active draft critical-path needs to avoid blocking on licensed
datasets. The blocked staging issues (#1497 SDD, #1498 ETH, #2000 AMV) remain blocked; this tool
will produce a real `priors-sufficient` / `materially-changes-scope` recommendation per dataset once
published statistics are cited.

## Validation Evidence

- `uv run pytest tests/research/test_external_prior_divergence.py` → 5 passed.
- `ruff check` / `ruff format` → clean.

## Follow-up

- A cited-sources pass to populate `external_prior_reference_stats.yaml` from published papers (and
  the authored side from tracked scenario bounds). This benefits from domain/maintainer input on
  acceptable published sources and is intentionally separated from this machinery PR.
- Builds toward #2919 (authored vs trace-derived prior comparison) and informs #3161.
