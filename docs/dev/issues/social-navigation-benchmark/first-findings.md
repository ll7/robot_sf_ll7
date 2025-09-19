# Social Navigation Benchmark – First Findings (Pilot)

Date: 2025-09-18

Purpose: Record the initial sanity check of the SocialForce (SF) pilot batch and the generated figures, and outline next steps.

## Pilot context
- Scenarios: 12 (from `docs/dev/issues/social-navigation-benchmark/scenario_matrix.yaml`)
- Episodes written to: `results/episodes_sf.jsonl`
- Figures generated to: `docs/figures/`
  - pareto.pdf/png, dist_collisions.pdf/png, dist_comfort_exposure.pdf/png, fig-force-field.pdf/png
- Baseline table: `docs/figures/baseline_table.md`
- Note: The pilot used a very short horizon (approx. 30 steps at dt=0.1), intended as a quick smoke.

## Quick metrics summary (n=12)
- collisions: mean/median/p95/min/max = 0.0
- comfort_exposure: 0.0 across the board
- success: 0.0 across the board (all episodes report `success=false`)
- Other kinematic metrics are populated (e.g., min_distance, jerk_mean, curvature_mean, energy), so simulation ran.

## Generated artifacts sanity
- All figures exist and are non-empty PDFs/PNGs.
- Baseline table renders correctly but shows zeros for both collisions and comfort_exposure for each scenario.

## Assessment vs. expectations
- We expected some non-zero comfort exposure for medium/high density or constrained scenarios.
- All-zero collisions and exposure across all scenarios + all failures likely stems from the short horizon (robot doesn’t reach goals, and there’s limited time for interactions to exceed thresholds).

## Next steps
1) Run a longer pilot (horizon ≈ 400, dt=0.1; repeats small) to induce measurable signal.
2) Regenerate the distributions/Pareto/table into a separate output folder for comparison.
3) If key metrics remain zero, investigate metric configuration or interaction flags (e.g., thresholds, ped–robot interaction settings) and validate scenario dynamics.

---
Related paths:
- Episodes (pilot): `results/episodes_sf.jsonl`
- Figures (pilot): `docs/figures/`
- Scenario matrix: `docs/dev/issues/social-navigation-benchmark/scenario_matrix.yaml`


## Post-fix longer batch (horizon≈400)

Date: 2025-09-18

Context:
- Fix: Pedestrian force buffer dtype set to float in `robot_sf/benchmark/runner.py` so force-derived metrics are not truncated.
- Episodes: `results/episodes_sf_long_fix1.jsonl` (12 scenarios × 10 repeats = 120 records)
- Figures/tables (canonical): `docs/figures/episodes_sf_long_fix1__a3953a1__v1/` (pareto, distributions for collisions/comfort_exposure/near_misses, baseline_table.md, fig-force-field)
- Latest alias: `docs/figures/_latest.txt` → `episodes_sf_long_fix1__a3953a1__v1`

Quick aggregate snapshot (n=120):
- collisions: mean 0.0, max 0.0
- near_misses: mean ≈ 1.7, max ≈ 8
- min_distance: mean ≈ 0.534, max ≈ 0.782
- comfort_exposure: mean ≈ 0.306, max ≈ 0.70
- force_exceed_events: mean ≈ 1.37e3, max ≈ 8.8e3
- avg_speed: mean ≈ 1.36, max ≈ 1.96

Observations:
- Comfort exposure and force exceed events are now clearly non-zero across scenarios, confirming the dtype fix.
- Collisions remain at 0.0; likely due to scenario design and/or conservative collision threshold (D_COLL=0.25). Will investigate.

Artifacts:
- Folder (canonical): `docs/figures/episodes_sf_long_fix1__a3953a1__v1/`
  - pareto.pdf/png
  - dist_collisions.pdf/png, dist_comfort_exposure.pdf/png, dist_near_misses.pdf/png
  - fig-force-field.pdf/png
  - baseline_table.md
  - meta.json (episodes path, git sha, schema version, CLI args)

Notes on reproducibility:
- The figures are now generated via the orchestrator with `--auto-out-dir --set-latest`, which stamps a canonical folder name and updates `_latest.txt` for stable includes in LaTeX/docs.

Next steps:
1) Analyze collision metric definition and thresholds vs. scenario scales; verify any episodes with min_distance < D_COLL.
2) If warranted, add a scenario or tweak parameters to induce occasional contact for validation.
3) Optionally compute baseline med/p95 for SNQI normalization and add to the reporting pipeline.
