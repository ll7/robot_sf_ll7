# Issue #1454 S10 H500 Candidate Comparison

Date: 2026-05-23

## Scope

This note records the interpretation of the exploratory S10 scenario-horizon h500 candidate
campaign:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1454_s10_scenario_horizons_h500_candidates.yaml \
  --campaign-id issue1454-s10-h500-candidates
```

The campaign intentionally excludes the asset-blocked `socnav_bench` row and compares the seven
functioning issue #1454 Stage A planner rows against local policy-search candidate rows. It is a
candidate robustness comparison, not a replacement for the full issue #1454 all-planner gate.

Related evidence:

- GitHub issue: <https://github.com/ll7/robot_sf_ll7/issues/1454>
- Stage A fixed-h100 gate:
  [issue_1454_stage_a_gate_2026-05-22.md](issue_1454_stage_a_gate_2026-05-22.md)
- Compact h500 candidate evidence:
  [evidence/issue_1454_s10_h500_candidates_2026-05-23/README.md](evidence/issue_1454_s10_h500_candidates_2026-05-23/README.md)
- Raw artifact release:
  <https://github.com/ll7/robot_sf_ll7/releases/tag/artifact/issue1454-s10-h500-candidates-2026-05-23>
- Earlier scenario-horizon h500 evidence:
  [evidence/issue_1023_scenario_horizons_local_full_2026-05-06/README.md](evidence/issue_1023_scenario_horizons_local_full_2026-05-06/README.md)
- Earlier policy-search h500 candidate evidence:
  [evidence/policy_search_h500_2026-05-06/README.md](evidence/policy_search_h500_2026-05-06/README.md)

## Result Summary

The campaign completed successfully:

- Campaign ID: `issue1454-s10-h500-candidates`
- Rows: `12`
- Episodes: `5760` total, `480` per row
- Runtime: `34507.6930` seconds, about `9.6` hours
- Campaign status: `benchmark_success=true`

The main result is positive but bounded: the candidate family remains the strongest observed
success/safety tradeoff on this h500 surface, while the larger seed sample softens the earlier
policy-search optimism.

Best direct-metric rows from
`evidence/issue_1454_s10_h500_candidates_2026-05-23/reports/campaign_table.md`:

| planner | success | collision | near misses | SNQI |
| --- | ---: | ---: | ---: | ---: |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | `0.8771` | `0.0250` | `18.9146` | `-0.0972` |
| `scenario_adaptive_hybrid_orca_v1` | `0.8729` | `0.0333` | `20.7771` | `-0.1037` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | `0.8729` | `0.0333` | `20.7771` | `-0.1037` |
| `hybrid_rule_v3_fast_progress_static_escape` | `0.8646` | `0.0354` | `21.8875` | `-0.1069` |
| `hybrid_rule_v3_fast_progress` | `0.7875` | `0.0292` | `21.9917` | `-0.1160` |
| `ppo` | `0.7854` | `0.1938` | `6.5146` | `-0.2415` |
| `orca` | `0.7750` | `0.1604` | `13.8250` | `-0.2476` |

Interpretation: the best candidate reaches roughly `0.88` success with about `0.025` collisions.
`ppo` and `orca` have similar success ceilings under h500, but with much higher collision rates.
This is the central evidence that the candidate rows improve the h500 success/safety tradeoff.

## Compared With Earlier H500

The seven shared non-candidate rows are broadly consistent with the issue #1023 h500 local
campaign, now with `480` episodes per row instead of `144`.

| planner | issue #1023 h500 success | issue #1454 S10/h500 success | change |
| --- | ---: | ---: | ---: |
| `orca` | `0.7569` | `0.7750` | `+0.0181` |
| `ppo` | `0.8056` | `0.7854` | `-0.0202` |
| `prediction_planner` | `0.4931` | `0.5312` | `+0.0381` |
| `socnav_sampling` | `0.4028` | `0.4000` | `-0.0028` |

Conclusion: the new campaign is best read as a stronger reproduction of the existing h500 shape,
not a surprising new baseline result.

## Compared With Fixed H100

Against the issue #1454 Stage A fixed-h100 surface, h500 strongly increases success but generally
also increases collision rates. It changes the benchmark question from tight-horizon completion to
longer-horizon completion with accumulated safety debt.

| planner | fixed h100 success | h500 success | fixed h100 collisions | h500 collisions |
| --- | ---: | ---: | ---: | ---: |
| `orca` | `0.1812` | `0.7750` | `0.0604` | `0.1604` |
| `ppo` | `0.2250` | `0.7854` | `0.1229` | `0.1938` |
| `prediction_planner` | `0.0625` | `0.5312` | `0.2229` | `0.4000` |
| `socnav_sampling` | `0.1604` | `0.4000` | `0.4792` | `0.6000` |

Conclusion: h500 is not simply "better evidence" than fixed h100. It is a different question. The
candidate result matters because it improves h500 success without paying the same collision penalty
as `orca`, `ppo`, or the other standard runnable rows.

## Compared With Policy-Search Candidates

The S10/h500 campaign is a useful robustness check on the earlier policy-search h500 summaries.
The candidates survive, but the wider seed sample makes the earlier evidence look somewhat
optimistic.

| candidate | policy-search success | S10/h500 success | change | policy collision | S10/h500 collision |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hybrid_rule_v3_fast_progress` | `0.8264` | `0.7875` | `-0.0389` | `0.0139` | `0.0292` |
| `hybrid_rule_v3_fast_progress_static_escape` | `0.9028` | `0.8646` | `-0.0382` | `0.0208` | `0.0354` |
| `scenario_adaptive_hybrid_orca_v1` | `0.9097` | `0.8729` | `-0.0368` | `0.0208` | `0.0333` |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | `0.9028` | `0.8729` | `-0.0299` | `0.0139` | `0.0333` |

Conclusion: the old policy-search surface was somewhat flattering, but the candidates still remain
ahead after the S10 rerun.

## Caveats And Claim Boundary

- `socnav_bench` is excluded because the local SocNavBench assets were unavailable. This is not a
  full all-planner verdict.
- Candidate rows are experimental adapter rows, not baseline-ready rows.
- SNQI diagnostics for this campaign report contract status `fail`, including negative rank
  alignment. Use direct success, collision, and near-miss metrics for claims; treat SNQI as
  diagnostic only.
- The compact campaign evidence predates explicit `benchmark_track` and `track_schema_version`
  metadata. Treat it as `legacy_track_unknown` per
  [issue_1721_benchmark_track_metadata_audit.md](issue_1721_benchmark_track_metadata_audit.md):
  valid for this historical S10/h500 surface, but not safe to aggregate with track-aware result rows
  or compare across observation tracks.
- The h500 surface should not replace fixed h100 in issue #1454 without an explicit scope decision.
  The fixed h100 Stage A gate remains failed at campaign level because `socnav_bench` failed closed.

## Recommended Wording

Use wording close to:

> The exploratory h500 candidates remain ahead of the existing runnable planner set on the main
> success/safety tradeoff. The best candidate reaches roughly `0.88` success with `0.025`
> collisions over `480` episodes. This result is robust enough to treat as useful challenger
> evidence, but it is not a full issue #1454 all-planner verdict because `socnav_bench` is excluded
> and SNQI fails its contract on this slice.
