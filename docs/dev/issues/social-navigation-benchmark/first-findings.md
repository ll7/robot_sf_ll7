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
