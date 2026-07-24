# Issue #5302 Selection Ceilings & Oracle Gap Report Schema

This document details the report schema and contract produced by `robot_sf/benchmark/issue_5302_oracle_gap.py` and `scripts/analysis/compute_issue_5302_oracle_gap.py`.

## Overview

The analysis module consumes complete, native 6-arm benchmark rows and computes:
1. Four selection ceilings:
   - `best_fixed_planner`: Single best planner across selection split / overall.
   - `best_planner_per_scenario_family`: Best planner selected for each scenario family.
   - `best_planner_per_scenario_cell`: Best planner selected for each scenario cell.
   - `hindsight_per_episode_oracle`: Best planner selected per individual episode.
2. Paired gaps (`family_gap`, `cell_gap`, `oracle_gap`) relative to `best_fixed_planner`.
3. 3-level hierarchical bootstrap (family -> cell -> episode) with seed 5302 and 95% confidence intervals.
4. Hierarchical-bootstrap Pareto dominance probabilities between planners and ceilings. The
   probability matrix is the fraction of replicates where the row entity strictly dominates the
   column entity across completion, collision, severe intrusion, and tail clearance.
5. Maximum normalized regret for each fixed planner relative to hindsight oracle.
6. Pre-registered claim gate decision rule (`STOP_SELECTOR` vs `PROCEED_TO_SELECTOR_ISSUE`).

## Fail-Closed Rules

The runner enforces strict fail-closed constraints:
- Non-native rows (`execution_mode != 'native'`) block output (`NonNativeRowError`).
- Non-successful evidence rows (`row_status != 'successful_evidence'`) block output (`InvalidRowStatusError`).
- Incomplete 6-arm episodes (missing or extra planners) block output (`IncompleteEpisodeError`).
- Missing selection/evaluation families, unknown split values, invalid provenance, non-finite
  numeric metrics, mixed repository commits, or cross-arm scenario/seed identity drift block
  output.
- Split leakage (overlap between selection and evaluation scenario families) blocks output (`SplitLeakageError`).
- Safety constraint ordering (`collision_rate` -> `severe_intrusion_rate` -> `selection_score`) prevents completion gains from compensating safety violations.
- Zero bootstrap samples and missing/non-finite completion-gap intervals block claim evaluation.
- The frozen 2-percentage-point claim gate uses paired `completion_rate` gaps and never emits
  `universally best`.

## Required Report Artifacts

The analysis produces 10 report files under `output/benchmarks/issue_5302_oracle_gap/reports/`:

1. `preflight.json`: Preflight validation metadata and row completeness checks.
2. `ceiling_summary.json`: Top-level JSON summary of selection ceilings, gaps, and claim gate.
3. `ceiling_summary.csv`: Tabular point estimates and CIs for each ceiling.
4. `family_breakdown.csv`: Breakdown of metrics grouped by scenario family and planner / ceiling.
5. `cell_breakdown.csv`: Breakdown of metrics grouped by scenario cell and planner / ceiling.
6. `failure_mechanism_map.csv`: Categorized failure modes (collision, severe intrusion, timeout, success).
7. `runtime_tail.csv`: Compute time percentiles (p50, p90, p95, p99, max) per entity.
8. `pareto_dominance.json`: Pairwise point-dominance matrix plus
   `pareto_dominance_probability` values in `[0, 1]` and the hierarchical bootstrap sample count.
9. `normalized_regret.csv`: Normalized regret distribution per planner.
10. `bootstrap_intervals.json`: Complete 95% hierarchical bootstrap intervals for all metrics and gaps.
