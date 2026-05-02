# Issue 692 Scenario Difficulty Analysis

This note defines the conservative pilot workflow for separating scenario-driven benchmark difficulty
from planner-quality or adapter-mismatch effects in camera-ready campaigns.

## Goal

Use existing camera-ready artifacts to answer two questions without changing the benchmark contract:

1. Which scenarios or scenario families are hard for most core planners?
2. Which weak planner results look like planner-specific mismatch even on relatively easier scenarios?

The workflow is analysis-first and artifact-driven. It does not replace the full benchmark with a
small subset, and it does not treat difficulty labels as ground truth.

## Artifact Inputs

Run the post-campaign analyzer:

```bash
source .venv/bin/activate
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id>
```

The analyzer consumes the existing campaign artifacts:

* `reports/campaign_summary.json`
* `reports/scenario_breakdown.csv`
* `reports/seed_variability_by_scenario.json`
* `preflight/preview_scenarios.json`

It writes:

* `reports/campaign_analysis.json`
* `reports/campaign_analysis.md`
* `reports/scenario_difficulty_analysis.json`
* `reports/scenario_difficulty_analysis.md`

## Primary Proxy

The current pilot proxy is `consensus_outcome_rank_v1` .

Definition:

* Restrict the consensus pool to core planners that are benchmark-success rows.
* Compute per-scenario consensus means for:
  + `success_mean`
  + `collisions_mean`
  + `near_misses_mean`
  + `time_to_goal_norm_mean`
* Convert each metric to a within-campaign hardness rank.
  + Lower success is harder.
  + Higher collisions, near-misses, and normalized time-to-goal are harder.
* Combine the metric ranks with fixed weights.

Interpretation:

* Higher `difficulty_score` means the scenario is hard for the consensus planner set.
* `snqi_mean` is supporting evidence only for the pilot. It is not the primary label.
* Seed confidence intervals and coefficient-of-variation fields stay separate from the difficulty
  score so noise is visible instead of hidden inside the label.

## Residual Logic

Planner residuals compare each planner's per-scenario outcome row against the consensus row for the
same scenario.

Use the residual view to separate two cases:

* hard-for-everyone: high scenario difficulty but no large planner-specific underperformance
* planner-mismatch: relatively easier scenario plus large residual underperformance for one planner

The analyzer summarizes:

* per-scenario ranks
* family-level rollups
* planner residual summaries
* planner-family residual summaries

Treat these residuals as diagnostic evidence, not as a readiness label by themselves.

## Verified-Simple Recommendation Rule

The default calibration candidate is the existing subset:

* `configs/scenarios/sets/verified_simple_subset_v1.yaml`

Recommendation rule:

* If the current campaign does not overlap with the verified-simple scenarios, report
`rerun_required` and keep the subset as a debugging or promotion gate only.
* If a bounded pilot exists, check whether the subset broadly preserves planner ordering and keeps
  seed noise comparable to the full campaign.
* Only treat the subset as a calibration aid when ordering is preserved and noise does not inflate
  materially.

Current conservative recommendation:

* Keep verified-simple as a gate, not a replacement benchmark.
* Promote it to a calibration aid only after a bounded pilot shows ordering preservation and stable
  seed noise.
* If the subset mostly shifts absolute scores or adds noise, keep it as a debugging surface rather
  than a benchmark interpretation surface.

## Limits

* Difficulty is an observed campaign property, not a hand-labeled taxonomy.
* Rankings are only as planner-agnostic as the selected consensus pool.
* Preview metadata can be truncated; missing static context should not be overread.
* This workflow does not change fallback policy, planner categories, or benchmark success rules.

## Validation Note

Successful end-to-end validation was rerun locally on April 1, 2026 with ORCA available via the
`rvo2` binding:

* Campaign:
  `output/benchmarks/camera_ready/camera_ready_baseline_safe_issue592_baseline_safe_rvo2_20260401_135310`
* Runner outcome:
  `423` episodes, `3/3` successful planner runs, `benchmark_success=true`
* Analyzer outputs written:
  + `reports/campaign_analysis.json`
  + `reports/campaign_analysis.md`
  + `reports/scenario_difficulty_analysis.json`
  + `reports/scenario_difficulty_analysis.md`

Observed difficulty ordering on that run remained plausible for the pilot proxy, with
`francis2023_robot_crowding`, `francis2023_robot_overtaking`, and
`francis2023_exiting_elevator` at the top of the hardest-scenario table.

The rerun also exposed one analyzer semantics caveat that is now tracked separately: non-paper
baseline-safe campaigns can produce a successful all-planner consensus while the planner metadata
still marks every row as `planner_group=experimental`, so the analysis must describe that fallback
explicitly instead of implying a true core-planner consensus.

The generated scenario-difficulty markdown now mirrors the JSON payload by including the
primary-proxy description line, so fallback runs stay visibly labeled as fallback runs in both
machine-readable and human-readable outputs.
