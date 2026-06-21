# Issue #2726 Scenario Priors Calibration Report

## Scope

This directory contains evidence and report artifacts for calibrating scenario priors from trace clusters.
It processes simulation trace exports from the repository, extracts features, groups them deterministically,
and generates scenario prior candidate cards.

## Evidence Status

- `schema`: `scenario-prior-card-registry.v1`
- `claim_boundary`: `repository_trace_grounded_not_real_world_calibrated: this report and generated prior cards are derived entirely from deterministic simulation trace clusters. They do not claim real-world validity, representativeness, or generalizability to real-world pedestrian behavior. Refer to issues #3161 and #2918 for real-world staging and calibration requirements.`
- `validation_reference`: Refer to #3161 and #2918.

## Files

- [scenario_prior_cards_issue_2726.yaml](scenario_prior_cards_issue_2726.yaml): Generated prior cards
- [report.md](report.md): Human-readable Markdown summary report
- [report.json](report.json): Fully detailed cluster statistics and traces mapping JSON

## Reproducible Command

```bash
uv run python scripts/analysis/calibrate_scenario_priors_from_traces_issue_2726.py \
    --trace-dir tests/fixtures/analysis_workbench/simulation_trace_export_v1 \
    --output-dir docs/context/evidence/issue_2726_scenario_prior_trace_clusters
```
