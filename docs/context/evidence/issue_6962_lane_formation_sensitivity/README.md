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

## Reference-control follow-up and closeout

The requested threshold-calibration and sustained-flow check is now retained in the [#6969
reference package](../issue_6969_lane_formation_reference/README.md). It contains 12 native rows
covering mixed sustained flow and an initialized separated-lane control, with 100 warm-up steps,
200 observation steps, deterministic boundary recycling, and sampling strides 1/2/4. The mixed
condition produced no clear-threshold hits in either calibration (0/3 in each); the separated
control produced 3/3 in each calibration. This shows that the metric distinguishes the retained
known controls under that protocol. The separated condition is initialized, so it is not evidence
of spontaneous lane emergence.

The reference package narrows the original interpretation boundary: the tested mixed sustained
flow still did not produce a robust clear profile, but the result remains diagnostic-only and does
not distinguish a scenario limitation from a model property. The #6962 result-interpretation
fixture now binds the #6969 compact summary alongside the #6962 summary, context, and provenance;
the packet remains `diagnostic_only` with `inconclusive` decisions for downstream use by #7029 and
#7032.

## Next research direction

Do not promote or retune the released configuration from these diagnostics. The immediate
measurement-control gate is complete for the retained reference conditions, but #6969 Stage B
parameter search remains stopped until a candidate-selection rule, fidelity/cost contract, and
domain review are separately approved. Any further spontaneous-emergence campaign must preserve
the same claim boundary and be authorized as a new research decision. This issue needs no additional
sensitivity run; after this packet/context closeout is reviewed, close #6962 and keep any tuning or
released-default decision in #6969.
