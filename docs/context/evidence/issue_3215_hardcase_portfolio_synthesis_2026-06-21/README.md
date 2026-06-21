# Issue #3215 Hard-Case Portfolio Synthesis

**Date:** 2026-06-21 · **Evidence tier:** `diagnostic` (NOT benchmark/paper-grade) · **Decision:** negative result with one narrow positive signal

## Source

The portfolio was executed on LiCCA (CPU) via #3306's tooling:
`SLURM/Licca/hardcase_portfolio_array.sl` over `configs/algos/hardcase_authority/*.yaml`
on `configs/scenarios/sets/predictive_hardcase_portfolio_v1.yaml`. 37 evaluation runs
(5 checkpoints × 9 authority variants × {hard, robusta}). Raw summaries/JSONL live in cluster
scratch `…/robot_sf/hardcase_portfolio/results/`; `portfolio_results.csv` (this dir) is the promoted
compact table. Nothing was running at synthesis time.

## Decision table (success rate)

**Hard scope** (`predictive_hard_seeds_v1`, n=7 episodes):

| checkpoint | baseline | combined_max | deep_seq | dense_lattice | high_angular | **nearfield_turn** |
|---|---|---|---|---|---|---|
| predictive_proxy_selected_v1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.143** |
| predictive_proxy_selected_v2_full | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **0.143** |

**Robusta scope** (n=60): baselines 0.067–0.083; `nearfield_turn`/`nf_speedcap_only` 0.100–0.117;
`nf_headings_only`/`nf_horizonboost_only` = baseline (no effect). Best: `sweep_h256_mp4_s42` +
`nearfield_turn` = 0.117. Full numbers in `portfolio_results.csv`. `mean_min_distance` 2.0–2.84 m,
`pass_min_distance=True` everywhere (no safety regression observed).

## Cross-bet classification

- **Authority bet (#3213): marginal — not a breakthrough.** The aggressive cruise-phase levers
  (`combined_max_authority`, `high_angular`, `dense_lattice`, `deep_sequence`) give **zero** hard-seed
  improvement. The **only** lever with a consistent positive effect is the **near-field turn budget**
  (`nearfield_turn`; its speed-cap component `nf_speedcap_only` also helps, while `nf_headings_only`
  and `nf_horizonboost_only` do not): hard 0 → 1/7, robusta +0.02–0.05, without min-distance
  regression. On the hard set this is a single solved episode (n=7) — within noise.
- **Selection bet (#3204): negative.** No checkpoint (proxy_selected_v1/v2, rgl, two sweeps) closes the
  hard gap; all ≈0 at baseline authority. Proxy selection did not surface a plateau-breaking checkpoint.
- **Model bet (#3214): negative.** The retrained/selected checkpoints do not beat the sweep checkpoints
  on hard seeds.
- **Portfolio decision:** **the hard-case plateau is NOT closed by selection, authority, or hard-case
  retraining alone** — the publishable negative result anticipated in #3215. Carryover signal:
  **near-field turn budget** is the single lever worth isolating in a larger-sample follow-up;
  aggressive cruise authority is not the binding constraint.

## Caveats (must travel with any reuse)

- **Tiny hard sample:** hard scope is n=7; the `nearfield_turn` 0.143 = one solved episode. No
  statistical claim; needs S20/S30 to test whether the signal survives.
- **Diagnostic only:** `experimental` algorithm tier, `tracked_agents_no_noise` (perfect perception).
  Not benchmark or paper-grade.
- **Collision metric absent** in the summaries (null) — the safety read here is `mean_min_distance`
  only, against a lenient diagnostic gate (`pass_success_rate=True` everywhere reflects a low gate,
  not a real success bar).
- **Durability:** raw results are in worktree-external cluster scratch, not git. Promote summaries/JSONL
  to a durable store (W&B / artifact manifest) via `scripts/tools/slurm_job_finalize.py --durable-uri`
  before any benchmark/paper use.

## Suggested next step

If the near-field-turn signal is worth chasing: a larger hard-seed sample (S20/S30) with the
`nearfield_turn` / `nf_speedcap_only` configs, collision-metric emission, and an observation-noise
slice — to decide whether it is a real (if small) effect or sampling noise. Otherwise record the
negative result in the forecast-lane limitation text (#2835) and route effort elsewhere.
