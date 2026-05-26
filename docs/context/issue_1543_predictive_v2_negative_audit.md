# Issue #1543 Predictive v2 Negative Audit

Related issues and PRs:

- Issue #1543: <https://github.com/ll7/robot_sf_ll7/issues/1543>
- Issue #1427: <https://github.com/ll7/robot_sf_ll7/issues/1427>
- Issue #1490: <https://github.com/ll7/robot_sf_ll7/issues/1490>
- PR #1480: <https://github.com/ll7/robot_sf_ll7/pull/1480>

## Decision

**Recommendation: revise coupling/objective before expansion.**

The durable #1427 same-seed evidence is negative for obstacle-feature promotion. The obstacle-feature
variant preserved the same scenario matrix, seed schedule, training budget, and evaluation surface,
but it finished with lower closed-loop success than the baseline while hard-seed success remained
zero for both variants. The most likely mechanism is **prediction-to-control coupling failure**, not
"no forecast improvement."

This note is intentionally limited to tracked/durable evidence. It does **not** infer per-row values
that are absent from git-tracked artifacts.

## Why this note exists

Issue #1490 proposes a four-way predictive-v2 expansion (baseline, obstacle-only, ego-only,
ego+obstacle). Issue #1543 asks whether the completed #1427 obstacle-only prerequisite already
provides enough negative evidence to stop or redirect that expansion before more training spend.

## Durable evidence inventory

| Surface | What it preserves | Evidence tier |
| --- | --- | --- |
| `docs/context/issue_1167_predictive_obstacle_pipeline.md` | final #1427 rerun outcome, stage-gate status, paired run IDs, aggregate success/min-distance metrics, best planner-grid row, interpretation | issue-scoped durable note |
| `docs/context/evidence/issue_1427_predictive_same_seed_handoff_2026-05-21/manifest.json` | canonical config pair, shared seed manifest, scenario matrix, hard-seed manifest, planner grid, fail-closed contract | compact tracked artifact manifest |
| `configs/training/predictive/predictive_br07_same_seed_issue_1427.yaml` | baseline run contract | canonical config |
| `configs/training/predictive/predictive_obstacle_features_same_seed_issue_1427.yaml` | obstacle-feature run contract | canonical config |
| `configs/training/predictive/predictive_same_seed_issue_1427_base_seed_manifest.yaml` | exact paired scenario/seed schedule | canonical manifest |
| PR #1480 body | launcher/artifact fixes plus the same final #1427 metrics | merged PR summary |

## Same-seed comparison contract

Tracked artifacts preserve that the paired #1427 comparison used:

- the same scenario matrix: `configs/scenarios/classic_interactions.yaml`
- the same hard-seed manifest: `configs/benchmarks/predictive_hard_seeds_v1.yaml`
- the same planner grid: `configs/benchmarks/predictive_sweep_planner_grid_v1.yaml`
- the same base seed manifest:
  `configs/training/predictive/predictive_same_seed_issue_1427_base_seed_manifest.yaml`
- the same training budget: 40 epochs, batch 128, lr `3e-4`, hidden dim 128, message passing 3
- the same evaluation surface: `horizon=120`, `dt=0.1`, `workers=1`, `campaign_workers=2`

That removes ordinary "different seeds" or "different budget" explanations for the observed
negative result.

## Paired scenario/seed rows preserved durably

The committed seed manifest preserves the paired row schedule for **23 scenarios / 92 total seeds**,
but no git-tracked #1427 artifact preserves per-row outcome values for those rows.

| Scenario | Seeds | Durable baseline row metrics | Durable obstacle row metrics |
| --- | --- | --- | --- |
| `classic_bottleneck_low` | `11000, 11001, 11002, 11003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_bottleneck_medium` | `111000, 111001, 111002, 111003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_bottleneck_high` | `211000, 211001, 211002, 211003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_realworld_double_bottleneck_high` | `311000, 311001, 311002, 311003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_station_platform_medium` | `411000, 411001, 411002, 411003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_cross_trap_low` | `511000, 511001, 511002, 511003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_cross_trap_medium` | `611000, 611001, 611002, 611003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_cross_trap_high` | `711000, 711001, 711002, 711003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_doorway_low` | `811000, 811001, 811002, 811003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_doorway_medium` | `911000, 911001, 911002, 911003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_doorway_high` | `1011000, 1011001, 1011002, 1011003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_group_crossing_low` | `1111000, 1111001, 1111002, 1111003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_group_crossing_medium` | `1211000, 1211001, 1211002, 1211003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_group_crossing_high` | `1311000, 1311001, 1311002, 1311003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_head_on_corridor_low` | `1411000, 1411001, 1411002, 1411003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_head_on_corridor_medium` | `1511000, 1511001, 1511002, 1511003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_merging_low` | `1611000, 1611001, 1611002, 1611003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_merging_medium` | `1711000, 1711001, 1711002, 1711003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_overtaking_low` | `1811000, 1811001, 1811002, 1811003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_overtaking_medium` | `1911000, 1911001, 1911002, 1911003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_t_intersection_low` | `2011000, 2011001, 2011002, 2011003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_t_intersection_medium` | `2111000, 2111001, 2111002, 2111003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |
| `classic_urban_crossing_medium` | `2211000, 2211001, 2211002, 2211003` | not tracked in git-tracked #1427 artifacts | not tracked in git-tracked #1427 artifacts |

## Durable baseline vs obstacle-feature comparison

The tracked record supports the following aggregate comparison.

| Signal | Baseline predictive | Obstacle-feature predictive | Durable source / caveat |
| --- | ---: | ---: | --- |
| Final status | failed stage gate | failed stage gate | both runs produced complete artifacts, but `evaluation_ok: false`; `campaign_ok: true`; `hard_seed_diagnostics_ok: true` |
| Success rate | `0.1304` | `0.1014` | from the final #1427 rerun summary in `docs/context/issue_1167_predictive_obstacle_pipeline.md` |
| Collision rate | not durably tracked | not durably tracked | no git-tracked #1427 row table or compact summary preserves exact collision-rate values |
| Near-miss rate | not durably tracked | not durably tracked | same caveat |
| Low-progress / timeout rate | not durably tracked | not durably tracked | same caveat |
| Mean min distance (final eval) | `2.1931` | `2.2105` | preserved in the final rerun summary |
| Hard-seed success | `0.0000` | `0.0000` | preserved in the campaign best-variant summary |
| Selected planner-grid variant | `risk_aware_adaptive` | `baseline_like` | preserved in the campaign best-variant summary |
| Global mean min distance for selected row | `2.1931` | `2.2081` | preserved in the campaign best-variant summary |
| Forecast metrics (ADE/FDE) | exact final values not durably summarized here | note states forecast loss / clearance signals improved and ADE/FDE quality gates passed; exact final values not durably summarized here | use only as qualitative obstacle-side evidence; do not invent exact ADE/FDE deltas |

## Evidence-synthesis table

| Mechanism | Source issue | Evidence tier | Config | Seeds | Artifacts | Metrics | Verdict | Caveats |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| Forecast improved but closed-loop success worsened | #1427 / PR #1480 | durable issue note + merged PR summary | `predictive_obstacle_features_same_seed_issue_1427.yaml` vs `predictive_br07_same_seed_issue_1427.yaml` | shared 23-scenario / 92-seed base schedule plus shared hard-seed manifest | `docs/context/issue_1167_predictive_obstacle_pipeline.md`, PR #1480, `docs/context/evidence/issue_1427_predictive_same_seed_handoff_2026-05-21/manifest.json` | success `0.1304 -> 0.1014`; mean min distance `2.1931 -> 2.2105`; hard success `0.0000 -> 0.0000` | strong evidence for transfer failure, not for planner improvement | exact ADE/FDE deltas are not durably summarized |
| Seed-distribution mismatch is unlikely as the primary explanation | #1427 | canonical config/manifest contract | same paired config pair | identical shared seed manifest and hard-seed manifest | same config/manifest surfaces | no seed-schedule difference between variants | rejected as primary mechanism | per-row outcomes are not durably preserved |
| Planner-grid selection artifact is a secondary caveat, not the main explanation | #1427 | durable aggregate campaign summary excerpt | same pair; shared planner grid | same paired seeds | final #1427 summary in `issue_1167` | best row changed from `risk_aware_adaptive` to `baseline_like`, but both hard-success values remained `0.0000` and both failed the same success gate | plausible contributor, insufficient as primary mechanism | no tracked per-row planner-grid table to isolate its effect |

## Mechanism classification

**Likely mechanism: prediction-to-control coupling failure.**

Why this is the best fit:

1. The obstacle-feature variant was run under the same seeds, same scenario matrix, same training
   budget, and same evaluation surface as the baseline.
2. The durable record explicitly says the obstacle-feature model improved validation forecast
   loss/clearance signals, so the negative result is not best explained by "insufficient forecast
   improvement."
3. Closed-loop success got worse (`0.1304` to `0.1014`) while hard-seed success stayed at `0.0000`
   for both variants, which is the classic signature of predictor gains not transferring into better
   planner choices.
4. The min-distance increase is small, so the tracked evidence is too weak to call this excessive
   conservatism as the primary mechanism.
5. The best planner-grid row changed, so planner-grid interaction may matter, but that does not
   overturn the stronger transfer-failure signal because both variants still failed the same success
   gate and neither solved the hard seeds.

## Recommendation boundary

**Recommendation: revise coupling/objective before expansion.**

Concretely, do **not** treat #1427 as support for immediate predictive-v2 four-way expansion. The
next justified step is to revise the planner-side coupling or planner-aligned training objective,
then re-establish a bounded preflight before spending more training budget on rows 3-4 of #1490.

This recommendation is stronger than "proceed with caveats" because:

- the only durable same-seed obstacle-only prerequisite is already negative,
- the hard-seed result is zero for both variants,
- and the tracked evidence does not show a closed-loop safety or success gain large enough to
  justify a larger matrix.

## Exact proposed #1490 update text

```md
Issue #1543 audit outcome: **revise coupling/objective before expansion**.

Using only durable #1427 / PR #1480 evidence, the obstacle-only prerequisite is negative on the
same-seed comparison:

- baseline predictive success: `0.1304`
- obstacle-feature predictive success: `0.1014`
- hard-seed success: `0.0000` for both variants
- mean min distance: `2.1931` (baseline) vs `2.2105` (obstacle final eval)
- best planner-grid row: `risk_aware_adaptive` (baseline) vs `baseline_like` (obstacle)

The tracked note also states the obstacle-feature model improved forecast-loss / clearance signals,
so the most likely mechanism is **prediction-to-control coupling failure**, not "no forecast
improvement." Planner-grid interaction is a caveat, but it is not strong enough to rescue the row:
both variants still failed the same success gate and neither solved the hard seeds.

Important limitation: the git-tracked #1427 evidence preserves the paired 23-scenario / 92-seed
schedule, but it does **not** preserve exact per-row collision, near-miss, or low-progress/timeout
values, so those should not be invented in follow-up summaries.

Recommendation for #1490: keep the four-way matrix blocked until the planner-side coupling or
planner-aligned objective is revised and a bounded preflight shows closed-loop improvement, not only
forecast improvement.

Durable note: `docs/context/issue_1543_predictive_v2_negative_audit.md`
```

## Follow-up boundary

This note does not close #1490 or mutate GitHub state. It records why the current durable evidence
supports a pre-expansion design revision rather than immediate four-way execution.
