<!-- AI-GENERATED (robot_sf_ll7#6969, 2026-08-13) - NEEDS-REVIEW -->

# Issue #6969 lane-formation reference and Stage A diagnostic

Plain-language summary: the lane metrics distinguish known mixed and separated
controls, and the native sustained-flow screen found no robust clear lane-forming
parameter profile in its frozen eight-cell space-filling sample. One profile crossed
the clear threshold for one of three seeds, which is a diagnostic lead—not a confirmed
lane-forming regime and not a tuning recommendation.

## Claim boundary

This is diagnostic-only local evidence. It is not benchmark evidence, paper evidence,
dissertation evidence, a model-wide statement, or a released-default change. The
separated-lane condition is initialized as a positive control and must not be read as
spontaneous lane emergence. No profile was selected for held-out confirmation because
none met the predeclared three-seed clear-hit rule.

## Protocol and provenance

- Reference implementation commit: `e298120d5ffd9782046c8401c8163877a1e245bc`.
- Stage A implementation commit: `8273fc384210a453b09e86d898141f7ae5d358c0`.
- Runtime: Python 3.13.14, PySocialForce 2.0.0, Linux x86_64.
- Reference command: `uv run python scripts/validation/run_issue_6969_lane_formation_reference.py --output-dir output/diagnostics/issue_6969_lane_formation_reference --generated-at 2026-08-13T08:40:00Z`.
- Stage A command: `uv run python scripts/validation/run_issue_6969_parameter_screen.py --output-dir output/diagnostics/issue_6969_parameter_screen --generated-at 2026-08-13T08:45:00Z`.
- Native protocol: 3 seeds (`5149, 5150, 5151`), 100 warm-up steps discarded, 200 observation steps, boundary recycling for sustained occupancy, sampling strides 1/2/4.
- Reference package: 12 native rows across mixed sustained flow and initialized separated-lane control, plus a synthetic metric audit.
- Stage A package: eight frozen Latin-hypercube profiles over seven declared factors plus released-default and literature-typical anchors; 30 native rows.

## Observed result

The synthetic metric audit passed at all sampling strides: the mixed fixture had LSI
approximately 0 and the separated fixture approximately 1, while sampling changed the
separated LSI by at most 0.00001 in that fixture. In the native package, within-run
sampling changes were at most 0.00251 for LSI and 0.00588 for lane purity.

| Condition/profile | Seeds | Mean LSI | LSI range | Clear LSI hits |
| --- | ---: | ---: | ---: | ---: |
| mixed, released anchor | 3 | 0.1857 | 0.0688–0.2753 | 0/3 |
| mixed, literature anchor | 3 | 0.0950 | 0.0536–0.1690 | 0/3 |
| separated positive control, released anchor | 3 | 0.7688 | 0.7104–0.8602 | 3/3 |
| separated positive control, literature anchor | 3 | 0.9360 | 0.9320–0.9384 | 3/3 |
| Stage A `lhs_01` | 3 | 0.2663 | 0.0444–0.3973 | 0/3 |
| Stage A `lhs_02` | 3 | 0.3255 | 0.2763–0.3758 | 0/3 |
| Stage A `lhs_03` | 3 | 0.0503 | 0.0180–0.1035 | 0/3 |
| Stage A `lhs_04` | 3 | 0.1667 | 0.0599–0.3204 | 0/3 |
| Stage A `lhs_05` | 3 | 0.3035 | 0.1213–0.5346 | 1/3 |
| Stage A `lhs_06` | 3 | 0.1988 | 0.0723–0.2864 | 0/3 |
| Stage A `lhs_07` | 3 | 0.2680 | 0.1372–0.3667 | 0/3 |
| Stage A `lhs_08` | 3 | 0.1827 | 0.0995–0.2329 | 0/3 |

The clear threshold is `lane_segregation_index >= 0.5`. The single `lhs_05` hit was
seed `5149`; seeds `5150` and `5151` were 0.1213 and 0.2545. The profile was therefore
not promoted to a held-out confirmation stage.

## Interpretation and next action

The reference stage establishes that the metric can detect a declared separated
trajectory and that the native observation is not materially changed by the tested
sampling stride. It does not establish that the model spontaneously forms lanes. The
Stage A screen is an outcome-free mechanism screen with no robust clear regime in its
tested envelope; it is inconclusive about regimes outside that envelope.

Stop here for this diagnostic package. Any continuation must separately freeze a
candidate-selection and held-out confirmation design, add fidelity outcomes (arching,
doorway oscillation, throughput, overlap/collision diagnostics), and obtain domain
review before a parameter recommendation or released-default decision. Raw rows,
manifests, and checksums remain ignored local artifacts; this directory contains only
the compact tracked synthesis and provenance pointers.
