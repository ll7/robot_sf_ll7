<!-- AI-GENERATED (robot_sf#6962, 2026-08-13) - NEEDS-REVIEW -->

# Issue #6962 lane-formation sensitivity diagnostic

Plain-language summary: a native Social Force Model diagnostic varied corridor geometry,
population, run duration, and speed calibration across the same ten seeds used by the existing
#5149 evidence bundle. No tested cell produced clear lane formation reproducibly, while weak
signals varied with seed and cell. This narrows the interpretation of the original non-result but
does not establish a model-wide absence of lane formation.

## Claim boundary

This is diagnostic-only local evidence. It is not benchmark-matrix evidence, paper-facing evidence,
a released-default change, a parameter recommendation, or a dissertation admission. The existing
lane-segregation and lane-purity metric semantics and released simulator defaults were unchanged.

## Protocol and provenance

- Code: `a11db4cba1b58478a2b205b6bf23bbeeaa5c9d35` on base `080536c8ef12269753a4c3bc4a48f5bc006d68c1`.
- Command: `uv run python scripts/validation/run_issue_6962_lane_formation_sensitivity.py --output-dir output/diagnostics/issue_6962_final --seeds 5149,5150,5151,5152,5153,5154,5155,5156,5157,5158 --generated-at 2026-08-13T06:30:00Z`.
- Native execution: 320/320 rows `native:computed`; no fallback, degraded, adapter, unavailable,
  or failed rows.
- Surface: lengths 16/24 m; corridor widths 3.5/5 m; populations 16/24; 200/400 steps;
  released-default and literature-typical speeds; ten seeds per cell.
- Threshold sensitivity: lane-segregation index at 0.15/0.3/0.5 and lane purity at 0.4/0.6/0.8.
- Historical comparator: `issue_5149_emergent_phenomena_multiseed_2026-08` was generated on
  macOS/arm64. Comparisons to it are qualified by platform drift; paired conclusions here use
  same-seed cells within this Linux run.

## Observed result

| Calibration | All 160 rows: mean LSI | LSI range | Cell-mean range | Maximum clear hit rate | Default cell mean LSI |
| --- | ---: | ---: | ---: | ---: | ---: |
| released-default | 0.1605 (SD 0.1040) | 0.0212–0.4659 | 0.1219–0.1833 | 0% | 0.1582 (0/10 clear) |
| literature-typical | 0.1492 (SD 0.0909) | 0.0104–0.3808 | 0.1162–0.1634 | 0% | 0.1513 (0/10 clear) |

The default cell is length 24 m, width 5 m, 24 pedestrians, and 400 steps. Its weak-threshold
(`LSI >= 0.15`) hit rates were 5/10 for released-default and 4/10 for literature-typical. Across
all non-default cells, paired same-seed mean differences versus that default cell ranged from
−0.0362 to +0.0251 for released-default and −0.0350 to +0.0121 for literature-typical. The
100,000-resample percentile intervals for every non-default cell included zero.

The result is therefore a mechanism screen, not a verdict: the tested geometry/population/
duration changes did not reliably activate the clear threshold, and the weak signal was noisy and
cell-dependent. The threshold grid is useful sensitivity information, but it is not a calibrated
reference threshold.

The complete compact cell surface is in [`summary.json`](summary.json). Raw rows and logs remain
ignored local output; their hashes and classifications are in
[`artifact_provenance.json`](artifact_provenance.json).

## Next research direction

Do not promote or retune the released configuration from this result. The smallest discriminating
next step is a threshold-calibration/reference case with an explicit warm-up or sustained-flow
control. If that reference still fails to produce stable separation, then #6969's Social Force
Model parameter sweep becomes the next bounded diagnostic; if it produces separation, revisit the
measurement cell and threshold before tuning force parameters. Keep #6962 and #6969 open until
that distinction is independently reviewed.
