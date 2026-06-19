# Campaign Comparison Report

- Schema: `campaign-comparison-report.v1`
- Study: `issue-3063-fixture`
- Report status: `analysis_only`
- Claim boundary: descriptive report only; row-status caveats and sample gates must be reviewed before benchmark or paper-facing claims

## Row Status Caveats

| row_status | count | interpretation |
|---|---:|---|
| adapter | 1 | benchmark_valid_denominator |
| degraded | 1 | excluded_or_limited |
| fallback | 1 | excluded_or_limited |
| native | 1 | benchmark_valid_denominator |

- Benchmark-valid denominator: `2`
- Excluded or limited rows: `2`

## Planner Metric Summaries

| planner | row_status_counts | metric | denominator | mean | ci95_low | ci95_high |
|---|---|---|---:|---:|---:|---:|
| goal | fallback:1, native:1 | success | 2 | 0.5000 | -0.4800 | 1.4800 |
| goal | fallback:1, native:1 | collision | 2 | 0.5000 | -0.4800 | 1.4800 |
| goal | fallback:1, native:1 | snqi | 2 | -0.1000 | -0.6880 | 0.4880 |
| goal | fallback:1, native:1 | comfort_exposure | 2 | 0.4500 | -0.2360 | 1.1360 |
| orca | adapter:1, degraded:1 | success | 2 | 0.5000 | -0.4800 | 1.4800 |
| orca | adapter:1, degraded:1 | collision | 2 | 0.0000 | 0.0000 | 0.0000 |
| orca | adapter:1, degraded:1 | snqi | 2 | 0.3500 | -0.1400 | 0.8400 |
| orca | adapter:1, degraded:1 | comfort_exposure | 2 | 0.2500 | 0.1520 | 0.3480 |

## Metric Visual Summaries

### success (core)

| planner | denominator | mean | visual |
|---|---:|---:|---|
| goal | 2 | 0.5000 | `####################` |
| orca | 2 | 0.5000 | `####################` |

### collision (core)

| planner | denominator | mean | visual |
|---|---:|---:|---|
| goal | 2 | 0.5000 | `####################` |
| orca | 2 | 0.0000 | `.` |

### snqi (social_compliance)

| planner | denominator | mean | visual |
|---|---:|---:|---|
| goal | 2 | -0.1000 | `######` |
| orca | 2 | 0.3500 | `####################` |

### comfort_exposure (social_compliance)

| planner | denominator | mean | visual |
|---|---:|---:|---|
| goal | 2 | 0.4500 | `####################` |
| orca | 2 | 0.2500 | `###########` |

## Statistical Hooks

| comparison | metric | mean_delta | denominators | sample_gate | interpretation |
|---|---|---:|---|---|---|
| orca - goal | success | 0.0000 | 2/2 | met | descriptive_only_formal_test_not_run |
| orca - goal | collision | -0.5000 | 2/2 | met | descriptive_only_formal_test_not_run |
| orca - goal | snqi | 0.4500 | 2/2 | met | descriptive_only_formal_test_not_run |
| orca - goal | comfort_exposure | -0.2000 | 2/2 | met | descriptive_only_formal_test_not_run |

## Limitations

- confidence intervals are normal-approximation descriptive intervals, not a significance claim
- fallback, degraded, unavailable, failed, and diagnostic-only rows are surfaced as limitations instead of successful benchmark evidence
