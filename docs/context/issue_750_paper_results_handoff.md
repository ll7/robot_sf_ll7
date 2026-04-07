# Issue 750 Paper Results Handoff

Date: 2026-04-07

## Purpose

`ll7/amv_benchmark_paper#62` needs a benchmark-owned export that restores confidence-interval
language in `paper/sections/results.tex` without reconstructing benchmark semantics in the paper
repository. The handoff is a sidecar export over one frozen benchmark source, not a benchmark rerun.

## Canonical Source

Use the final March 18 canonical publication bundle:

- `output/benchmarks/publication/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407_publication_bundle`

Do not use the superseded issue-580 reruns for paper Results values.

## Command

```bash
uv run python scripts/tools/paper_results_handoff.py \
  --source output/benchmarks/publication/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407_publication_bundle
```

Default output:

- `output/benchmarks/publication/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407_paper_results_handoff/paper_results_handoff.json`
- `output/benchmarks/publication/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407_paper_results_handoff/paper_results_handoff.csv`

## Export Contract

Schema: `paper-results-handoff.v1`

Rows are planner-summary rows keyed by `planner_key` and `kinematics`. They cover the current
Results-section planner set from the frozen bundle, including headline/control and non-headline
rows.

Metrics:

- `success`
- `collisions`
- `near_misses`
- `time_to_goal_norm`
- `snqi`

Each metric includes:

- `<metric>_mean`
- `<metric>_std`
- `<metric>_count`
- `<metric>_ci_low`
- `<metric>_ci_high`
- `<metric>_ci_half_width`
- `<metric>_source_table_mean` when the frozen campaign table exposes the rounded mean

Each row also includes:

- `episode_count`
- `seed_count`
- `seed_list`
- planner metadata such as `planner_group`, `readiness_tier`, `readiness_status`,
  `preflight_status`, and `status`
- provenance fields such as `campaign_id`, `config_hash`, `git_hash`, `seed_policy_mode`,
  and `seed_policy_seed_set`
- CI metadata: `confidence_method`, `confidence_level`, `bootstrap_samples`, and `bootstrap_seed`

## Interval Method

The exporter recomputes intervals from the frozen episode JSONL records. It groups episodes by
planner, kinematics, and seed, computes per-seed means, then bootstraps over seed means.

Default settings:

- method: `bootstrap_mean_over_seed_means`
- confidence: `0.95`
- bootstrap samples: `400`
- bootstrap seed: `123`

## Downstream Mapping

Use `paper_results_handoff.csv` for Results-table values and CI text. The manuscript-facing mapping
is direct:

- `Succ.` -> `success_mean`, `success_ci_low`, `success_ci_high`
- `Coll.` -> `collisions_mean`, `collisions_ci_low`, `collisions_ci_high`
- `Near miss` -> `near_misses_mean`, `near_misses_ci_low`, `near_misses_ci_high`
- `TTG norm.` -> `time_to_goal_norm_mean`, `time_to_goal_norm_ci_low`,
  `time_to_goal_norm_ci_high`
- `SNQI` -> `snqi_mean`, `snqi_ci_low`, `snqi_ci_high`

Use `seed_count`, `seed_list`, and `episode_count` for seed/repeat sensitivity wording.
