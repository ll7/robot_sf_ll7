# Issue 838 SNQI Calibration And Normalization Robustness

Date: 2026-04-16

Related issues:

- `robot_sf_ll7#838` Research: SNQI calibration and normalization robustness follow-up after v3 contract
- `robot_sf_ll7#822` Strengthen SNQI as an operational multi-objective aggregation metric
- `robot_sf_ll7#635` SNQI v3 paper-facing contract delta and canonical field mapping

## Goal

Check whether the fixed SNQI v3 paper contract should stay fixed, be demoted further, or seed a
future v4 candidate by replaying frozen benchmark episodes through controlled weight and
normalization variants.

This note does not change the current paper-facing S3/S5 numbers, scenario matrix, benchmark
semantics, or v3 asset contract.

## Inputs And Limitation

The exact issue-635 publication bundle path was not mounted in this checkout:

- `output/benchmarks/publication/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407_publication_bundle`

The executable proof in this branch used the closest available frozen camera-ready campaign:

- `output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_rel_0_0_2_full_rerun_20260414_081244`

Versioned SNQI assets:

- `configs/benchmarks/snqi_weights_camera_ready_v3.json`
- `configs/benchmarks/snqi_baseline_camera_ready_v3.json`

Interpretation:

- the analysis is valid evidence for the implemented calibration workflow and for this frozen local
  campaign slice,
- but manuscript-facing conclusions should be rerun against the canonical publication bundle when
  that artifact is mounted.

## Command

```bash
uv run python scripts/tools/analyze_snqi_calibration.py \
  --campaign-root output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_rel_0_0_2_full_rerun_20260414_081244 \
  --weights configs/benchmarks/snqi_weights_camera_ready_v3.json \
  --baseline configs/benchmarks/snqi_baseline_camera_ready_v3.json \
  --epsilon 0.15 \
  --output-json output/ai/autoresearch/snqi_issue838/snqi_calibration_analysis.json \
  --output-md output/ai/autoresearch/snqi_issue838/snqi_calibration_analysis.md \
  --output-csv output/ai/autoresearch/snqi_issue838/snqi_calibration_variants.csv
```

Artifacts:

- `output/ai/autoresearch/snqi_issue838/snqi_calibration_analysis.json`
- `output/ai/autoresearch/snqi_issue838/snqi_calibration_analysis.md`
- `output/ai/autoresearch/snqi_issue838/snqi_calibration_variants.csv`

## Perturbation Protocol

The workflow compares:

- baseline `v3_fixed` weights and anchors,
- simplex-preserving `+-15%` local perturbations around each v3 weight,
- `uniform_simplex` as a broad weight alternative,
- `component_subset_no_jerk` as a component-subset ablation,
- dataset-derived normalization anchors: median/p95, median/p90, median/max, and p25/p75.

Metrics reported:

- planner rank correlation versus v3,
- episode rank correlation versus v3,
- mean absolute planner rank shift,
- component-to-SNQI Spearman alignment,
- dominant component contribution magnitude,
- contract health under the existing rank/separation/dominance thresholds.

Implementation note:

- anchor variants are generated for the baseline-normalized terms used by
  `robot_sf.benchmark.snqi.compute.normalize_metric`; the canonical implementation still consumes
  `time_to_goal_norm` directly as an already-normalized episode metric.

## Results

Frozen local campaign size:

- episodes: `987`
- planners: `7`

Baseline v3 ordering:

| Rank | Planner | Mean SNQI |
|---:|---|---:|
| 1 | `socnav_sampling` | `-0.094980` |
| 2 | `ppo` | `-0.138824` |
| 3 | `orca` | `-0.142374` |
| 4 | `goal` | `-0.146781` |
| 5 | `sacadrl` | `-0.156746` |
| 6 | `prediction_planner` | `-0.173891` |
| 7 | `social_force` | `-0.174008` |

Baseline v3 diagnostics on this slice:

- contract status: `pass`
- rank alignment Spearman: `0.607143`
- outcome separation: `0.205539`
- aligned variable metrics: `6 / 6`
- dominant component: `time_penalty`
- dominant component mean absolute contribution: `0.092598`

Sensitivity summary:

- local weight minimum planner-rank correlation: `0.928571`
- local weight minimum episode-rank correlation: `0.997252`
- anchor minimum planner-rank correlation: `0.607143`
- anchor minimum episode-rank correlation: `0.916550`
- local weight order-change count: `7`
- anchor order-change count: `2`
- non-baseline order-change count: `11`

Interpretation:

- The v3 component alignment is healthy on this slice.
- Local weight perturbations are mostly stable at the episode level and remain above the planner-rank
  demotion threshold, so the v3 weights are not the main weakness.
- Normalization-anchor changes are more sensitive: p25/p75 anchors reduce planner-rank correlation
  to `0.607143`, and median/p90 anchors change planner order with planner-rank correlation
  `0.857143`.
- No tested variant met the stricter future-v4 criterion of improving component alignment by at
  least `0.10` while preserving v3 planner ranks.

## Recommendation

Decision: `demote_snqi_further`.

Meaning:

- keep the fixed SNQI v3 contract intact for any already-frozen paper bundle,
- do not replace it with a v4 candidate from this analysis,
- avoid promoting SNQI as a main narrative claim beyond a supporting synthesis aid,
- rerun this exact workflow on the canonical publication bundle before making any manuscript-level
  calibration claim.

Why:

- v3 is component-aligned and reproducible,
- but anchor sensitivity is strong enough that the fixed contract should not be presented as a
  generally robust objective without additional frozen-bundle evidence,
- and the analysis did not identify a clearly superior replacement contract.

Conservative manuscript wording:

- "SNQI is reported as a fixed, versioned synthesis aid using the v3 contract."
- "Sensitivity analysis supports component alignment on the evaluated frozen slice, but
  normalization-anchor dependence remains a limitation."
- "Planner rankings and primary conclusions should continue to be interpreted through component
  metrics, not SNQI alone."

Avoid:

- claiming SNQI v3 is locally optimal,
- retroactively changing current paper-facing values,
- promoting SNQI as a universal social-navigation utility scalar.

## Validation

Targeted tests:

```bash
uv run pytest tests/unit/benchmark/test_snqi_calibration.py tests/tools/test_analyze_snqi_calibration.py
```

Result:

- `5 passed`

Frozen-campaign analysis:

```bash
uv run python scripts/tools/analyze_snqi_calibration.py \
  --campaign-root output/benchmarks/camera_ready/paper_experiment_matrix_all_planners_v1_rel_0_0_2_full_rerun_20260414_081244 \
  --weights configs/benchmarks/snqi_weights_camera_ready_v3.json \
  --baseline configs/benchmarks/snqi_baseline_camera_ready_v3.json \
  --epsilon 0.15 \
  --output-json output/ai/autoresearch/snqi_issue838/snqi_calibration_analysis.json \
  --output-md output/ai/autoresearch/snqi_issue838/snqi_calibration_analysis.md \
  --output-csv output/ai/autoresearch/snqi_issue838/snqi_calibration_variants.csv
```

Result:

- recommendation: `demote_snqi_further`
- episodes: `987`
- planners: `7`

## Follow-Up Boundary

No follow-up issue is required from this branch unless maintainers want the canonical publication
bundle remounted and rerun as a separate artifact-validation task. The implemented command is ready
for that rerun without changing benchmark semantics.
