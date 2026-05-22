# Issue #1454 Stage A Gate

Date: 2026-05-22
Related:
- GitHub issue: <https://github.com/ll7/robot_sf_ll7/issues/1454>
- Draft PR: <https://github.com/ll7/robot_sf_ll7/pull/1455>
- SocNavBench re-entry gate: [issue_562_socnav_bench_reentry.md](issue_562_socnav_bench_reentry.md)
- Stage A plan: [issue_1454_s10_robustness_plan.md](issue_1454_s10_robustness_plan.md)
- Compact evidence:
  [issue_1454_stage_a_fixed_h100_2026-05-22](evidence/issue_1454_stage_a_fixed_h100_2026-05-22/README.md)

## Decision

Stage B is not gated in from the 2026-05-22 Stage A run.

The Stage A S10 fixed-h100 campaign produced a comparable table for seven broader-baseline planner
rows, but the `socnav_bench` row failed closed before episode execution because required
SocNavBench assets are missing locally. That leaves a campaign-level unresolved failure, so the
scenario-horizon h500 Stage B comparison should not be run or interpreted as the final fixed-vs-h500
answer for issue #1454.

## Stage A Evidence

Command:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1454_s10_fixed_h100_broader_baselines.yaml \
  --campaign-id issue1454-s10-fixed-h100
```

Campaign summary:

- Campaign ID: `issue1454-s10-fixed-h100`
- Git commit: `17b179007e6292dd0365c53cff719cccea1276a9`
- Runtime: `2722.0126` seconds
- Total episodes: `3360`
- Planner rows: `8`
- Successful rows: `7`
- Campaign success: `false`

Completed rows:

- `goal`: `480` episodes
- `social_force`: `480` episodes
- `orca`: `480` episodes
- `ppo`: `480` episodes
- `prediction_planner`: `480` episodes
- `socnav_sampling`: `480` episodes
- `sacadrl`: `480` episodes

Failed row:

- `socnav_bench`: `0` episodes, fail-closed prerequisite error:
  SocNavBench control-pipeline parameters could not load because required data directories are
  missing, starting with `third_party/socnavbench/wayptnav_data`.

This is an actionable dependency failure, not fallback benchmark evidence and not a planner-quality
result.

## Analysis Outputs

Analyzer command:

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/camera_ready/issue1454-s10-fixed-h100 \
  --output-json output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/campaign_analysis.json \
  --output-md output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/campaign_analysis.md
```

Automated consistency checks reported no additional inconsistencies beyond the failed
`socnav_bench` campaign status. The slowest row was `prediction_planner` at `1341.0938` seconds;
the top slow scenarios were the high-density double-bottleneck and t-intersection slices.

SNQI diagnostics remain interpretive rather than decisive:

- `snqi_contract_status=warn`
- Spearman rank alignment: `0.3214`
- Outcome separation: `0.2090`
- Dominant component: `time_penalty`
- Claim scope: benchmark aggregate, not a universal ground-truth utility

## Comparison To May 4 Fixed-H100 Reference

Comparison command:

```bash
uv run python scripts/tools/compare_camera_ready_campaigns.py \
  --base-campaign-root docs/context/evidence/camera_ready_all_planners_2026-05-04 \
  --candidate-campaign-root output/benchmarks/camera_ready/issue1454-s10-fixed-h100 \
  --output-json output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/may4_fixed_h100_vs_issue1454_s10_comparison.json \
  --output-md output/benchmarks/camera_ready/issue1454-s10-fixed-h100/reports/may4_fixed_h100_vs_issue1454_s10_comparison.md
```

The comparison found no planner coverage gaps. `socnav_bench` failed with `0` episodes in both the
May 4 reference and the issue #1454 S10 candidate. Other rows were expected to drift because the
candidate uses S10 rather than the smaller May 4 seed set.

Planner-level aggregate deltas, candidate minus May 4 reference:

| planner | success delta | collision delta | SNQI delta |
| --- | ---: | ---: | ---: |
| `goal` | -0.0076 | +0.0035 | -0.0182 |
| `orca` | +0.0006 | +0.0257 | +0.0092 |
| `ppo` | -0.0250 | +0.0326 | -0.0220 |
| `prediction_planner` | -0.0069 | +0.0146 | -0.0131 |
| `sacadrl` | +0.0104 | -0.0097 | -0.0277 |
| `social_force` | +0.0000 | +0.0167 | -0.0169 |
| `socnav_sampling` | -0.0132 | -0.0486 | -0.0533 |

These deltas are useful robustness context for the seven completed rows, but they are not a
fixed-h100-vs-h500 recommendation because Stage B was not run.

## Recommendation

Do not execute Stage B or S20 from this state.

The next benchmark-strengthening step is either:

1. hydrate SocNavBench assets, satisfy the issue #562 re-entry probe, and rerun Stage A so all
   broader-baseline rows have comparable S10 evidence; or
2. make an explicit maintainer decision to scope issue #1454 to the seven runnable rows, then run a
   matched Stage B comparison under that narrowed row set.

Until one of those happens, issue #1454 has a complete Stage A fail-closed gate result, but it does
not have a valid fixed-h100 versus scenario-horizon h500 verdict.
