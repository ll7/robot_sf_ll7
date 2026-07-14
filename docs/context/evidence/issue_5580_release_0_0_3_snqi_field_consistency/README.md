<!-- AI-GENERATED (robot_sf#5580, 2026-07-14) - NEEDS-REVIEW -->

# Issue #5580 — Release 0.0.3 Per-Episode `snqi` Field vs Diagnostics-Basis Consistency

## Plain-language summary

Issue #5580 reports that the per-episode `metrics["snqi"]` field in the 0.0.3 publication
bundle disagrees with the `snqi_diagnostics.json` `planner_ordering` computed under the
declared `camera_ready_v3` weights/baseline, and that the two surfaces can elect a different
SNQI-best arm.

Root-cause confirmed by direct reproduction against the released asset (SHA-256
`3cfefaaa39aab6cae541cece9573848a7e0afc5e1d9e4c9a7bbf48df2330b1a7`): the two surfaces are
computed by **two different SNQI formulas**:

- the per-episode `snqi` field is baked at episode-capture time by
  `robot_sf.benchmark.metrics.snqi` (the campaign-aware scalarizer that includes the
  `curvature_mean` / `w_curvature` term);
- the `snqi_diagnostics.json` `planner_ordering` is computed by
  `robot_sf.benchmark.snqi.compute.compute_snqi_v0` (which has **no** curvature term).

Because the formulas differ, the stored per-episode field matches a curvature-aware
recomputation at **100% (20,160 / 20,160)**, but matches the `compute_snqi_v0` diagnostics
basis at only **~33.9%** — exactly the figure reported in the issue. The two bases also elect
different SNQI-best arms: the per-episode-field basis ranks
`scenario_adaptive_hybrid_orca_v1` first, while the diagnostics basis ranks `ppo` first.

## Canonical-basis decision (ask #1 / #2)

The per-episode field is the surface that downstream `campaign_table.csv` SNQI means are built
from, and it uses the same curvature-aware `metrics.snqi` scalarizer that the aggregation path
(`robot_sf.benchmark.aggregate._ensure_snqi`) uses. Therefore the **curvature-aware
`metrics.snqi` basis is canonical** for the release: the per-episode field is correct, and the
diagnostics `planner_ordering` must be reconciled to the same formula (successor fix tracked
separately under #4364, the broader release re-base). The stored field is **not** legacy — it
is the canonical per-episode SNQI; only the diagnostics ordering basis must be aligned to it.

## Release gate (ask #3)

This PR adds `scripts/validation/check_release_snqi_field_consistency.py`, a fail-closed gate
that recomputes every per-episode `snqi` with the curvature-aware basis and asserts the
per-episode field and the `snqi_diagnostics.json` `planner_ordering` cannot drift. It checks:

1. **per-episode field integrity** — stored `metrics.snqi` equals the curvature-aware
   recomputation (0 mismatches on the real bundle);
2. **field-vs-recomputed ordering** — the arm ranking derived from the field equals the
   curvature-aware recompute ranking;
3. **field-vs-diagnostics ordering** — the field arm ranking equals the diagnostics
   `planner_ordering` (this is the check that **fails** on the real 0.0.3 bundle, reproducing
   the reported drift at the ordering level).

The gate reads the archive without extracting it and fails closed on archive/hash drift,
missing `snqi_diagnostics.json`, SHA-256 mismatch with the configured weights/baseline, or any
ordering disagreement.

## Classification and claim boundary

- `schema_version`: `release_snqi_field_consistency.v1`
- `status`: `fail` on the 0.0.3 bundle (drift reproduced); `pass` on a reconciled bundle
- `result_classification`: `snqi_field_inconsistent` / `snqi_field_consistent`
- `evidence_grade`: `diagnostic-only`
- `diagnostic_only`: `true`
- `benchmark_promotion`: `false`
- `paper_facing`: `false`
- `provenance`: `seeds`: S30 release seed budget; `config`: pinned v3 weights/baseline
  SHA-256; `hash`: archive SHA-256 recorded below.
- `claim_boundary`: `Diagnostic-only reconciliation of the per-episode metrics.snqi field against the snqi_diagnostics.json planner_ordering basis, recomputed with the same curvature-aware SNQI scalarizer that produced the stored field. A pass is not release-wide benchmark success, planner-ranking evidence, SNQI contract validity, or a paper claim.`

## Reproduction

Download the named asset from release 0.0.3 and verify its SHA-256 digest is
`3cfefaaa39aab6cae541cece9573848a7e0afc5e1d9e4c9a7bbf48df2330b1a7`, then run:

```bash
uv run python scripts/validation/check_release_snqi_field_consistency.py \
  --bundle <downloaded-release-asset.tar.gz> \
  --expected-bundle-sha256 3cfefaaa39aab6cae541cece9573848a7e0afc5e1d9e4c9a7bbf48df2330b1a7 \
  --expected-release-tag 0.0.3 \
  --source-url https://github.com/ll7/robot_sf_ll7/releases/download/0.0.3/paper_experiment_matrix_v2_h600_s30_extended_release_v0_0_3_final_publication_bundle.tar.gz \
  --output docs/context/evidence/issue_5580_release_0_0_3_snqi_field_consistency/summary.json
```

The unit/fixture tests live in
`tests/validation/test_check_release_snqi_field_consistency.py` (3 deterministic fixture tests
plus a gated real-bundle reproduction test behind `ROBOT_SF_0_0_3_BUNDLE`).

## Reproduction result (real 0.0.3 bundle)

- `status`: `fail`
- `violation_counts`: `field_vs_diagnostics_ordering: 1`
- `counts.rows`: `20160`, `episode_field_present`: `20160`, `snqi_recomputed_rows`: `20160`
- `snqi_field_recompute_mismatch`: `0` (per-episode field is internally consistent with the
  curvature-aware canonical basis)
- field SNQI-best arm: `scenario_adaptive_hybrid_orca_v1::differential_drive`
- diagnostics SNQI-best arm: `ppo::differential_drive`

## Remaining work (not in this slice)

- Reconcile the `snqi_diagnostics.json` `planner_ordering` to the curvature-aware basis so the
  gate passes on the next re-based release (tracked under #4364). This slice delivers the gate
  and the canonical-basis decision; it does not re-bake the published bundle.

No benchmark campaign was rerun, no Slurm/GPU job was submitted, and no paper/dissertation
claim was edited or promoted for this audit.
