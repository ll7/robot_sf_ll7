# Job 13274 Headline Confidence-Interval and Rank-Stability Evidence

This bundle preserves the completed job 13274 analysis needed to review which headline planner
comparisons remain uncertain at the available seed budget.

## Claim boundary

**Evidence status: diagnostic-only.** The verified 8,640-episode S20 campaign was analyzed with no
fallback or degraded cells admitted, but this bundle does not promote a planner ranking, benchmark
conclusion, paper claim, or dissertation claim. The constraints-first result remains subject to
claim-card/domain review and the S30 decision. Social Navigation Quality Index (SNQI) ranks are
`blocked_invalid_metric` because the source campaign failed its warning-level SNQI contract.

## Provenance

- Source: completed Slurm job `13274`, campaign `issue3216_s20_headline_ci`.
- Campaign commit: `e7730c46de33399d709d58c3dcb97854bb0f1d71`.
- Analysis commit: `220e30db7e100c9c2d4b3e6f8dd253b0ad7197fd`.
- Config:
  `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml`.
- Input: 8,640 episodes, 9 expected planners, 315 counted scenario-planner cells, 0 excluded
  cells, and a minimum of 20 seeds per cell.
- Method: 95% per-cell bootstrap confidence intervals (2,000 samples, seed 123) and 500 rank
  resamples (seed 123), using the canonical constraints-first profile with success as the rank
  metric and collision/near-miss availability as required safety constraints.

The runner was invoked after hydrating the verified harvest under its logical ignored output path:

```bash
uv run python scripts/analysis/run_issue3216_rank_stability.py \
  --campaign-root output/issue3216-13274-harvest/issue3216_s20_headline_ci \
  --harvest-log /path/to/verified/harvest_13274.log \
  --output-dir output/issue3216-13274-harvest/analysis
```

Two consecutive invocations produced identical hashes:

- `result.json`: `4e9c9a44d86190f19bd3735693aa2b8a504ba9a1ff6575ad041bd8b9ab580ce2`.
- `report.md`: `d9fe051d53300409f7ebbdb3211a35700f8e0f2671ce2908f6fbcf87af0acfe0`.
- `analysis_provenance.json`:
  `7de137b6da59bd7ea5b23bb424c20b109a8dc52067a78d85f25839c5c26d1252`.

## Headline Result

| Review field | Observed result |
| --- | --- |
| Generated classification | `blocked_until_run` |
| Manuscript-table status | `ready_for_table_review_no_claim_promotion` |
| S30 decision | `needs_review` |
| Constraints-first metric gaps | 0 |
| Identifiable scenario rankings | 35 of 35 |
| Scenarios with any resampled rank movement | 32 of 35 |
| Scenarios with stable top-1 planner | 12 of 35 |
| Adjacent success-rank comparisons with non-overlapping CIs | 45 of 280 |
| Adjacent comparisons not distinguishable at this budget | 235 of 280 |

`blocked_until_run` is the report harness's conservative legacy classification: empirical
execution is complete, while claim promotion is still blocked on review. Issue #5607 tracks this
misleading state name. The decision packet's two reasons for `needs_review` are
`adjacent_rank_ci_overlap_requires_claim_downgrade_or_more_data` and
`rank_resampling_instability_present`.

The exact SNQI failure is `rank_alignment_spearman=-0.2`, below the `0.3` fail threshold, under
`snqi_contract.enforcement=warn`. SNQI therefore remains explanatory only and cannot support an
adjacent-rank claim.

## Artifact Policy

- `result.json` is the complete canonical per-cell confidence-interval, rank-resampling, and
  adjacent-claim result.
- `report.md` is the generated human-readable rendering of the same result.
- `analysis_provenance.json` records the input/output hashes and exact SNQI failure.
- Raw episode rows, the harvest log, and the 204 MiB harvested campaign tree remain ignored local
  artifacts and are not copied into Git.
