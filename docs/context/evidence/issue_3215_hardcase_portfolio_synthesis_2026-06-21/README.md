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

Issue #3342 added the missing collision-rate and uncertainty fields, configured S20/S30 hard-seed
manifests, and ran a local S20 diagnostic slice for `baseline`, `nearfield_turn`, and
`nf_speedcap_only` across clean/noisy observation slices. That S20 result did not reproduce the
near-field-turn signal and remains diagnostic-local, not benchmark or paper evidence. See
`docs/context/evidence/issue_3342_nearfield_turn_budget_2026-06-21/`.

## Closure audit (2026-07-06)

**Status:** epic scope complete at `diagnostic` tier; recommended close as a completed negative
result. All three child bets are now closed and the cross-bet portfolio decision is recorded above.

### Child bet state

| Bet | Issue | State | Outcome |
|---|---|---|---|
| Selection (proxy checkpoint) | #3204 | **CLOSED 2026-07-05** | negative — no checkpoint closes the hard gap |
| Authority (action-lattice / turn budget) | #3213 | **CLOSED 2026-07-01** | marginal — only near-field turn budget helps, within n=7 noise |
| Model (hard-case retraining) | #3214 | **CLOSED 2026-07-06** (PR #4621, negative result) | negative — retrained checkpoints do not beat sweeps on hard seeds |

### Acceptance criterion → evidence

Original body criteria:

- **All three bets executed under the shared protocol** — met at `diagnostic` tier: 37 runs on LiCCA
  CPU via #3306 tooling over `predictive_hardcase_portfolio_v1`; per-bet children #3204/#3213/#3214
  all executed and closed. The pre-authorized overnight *GPU* budget was never granted, but the
  diagnostic tournament already covered all three levers with a consistent negative, so the GPU
  upgrade is a paper-grade *confirmation* follow-up, not a blocker for the portfolio decision.
- **Per-bet classification with uncertainty** — met (`## Cross-bet classification` above): authority
  marginal, selection negative, model negative, each with explicit n and noise caveats.
- **Single portfolio decision** — met: the hard-case plateau is **not** closed by selection,
  authority, or hard-case retraining alone (publishable negative result).
- **Winning artifacts promoted durably; synthesis note + evidence summary recorded** — met for the
  compact decision table (`portfolio_results.csv`, registered in `docs/context/catalog.yaml`); raw
  cluster-scratch JSONL durability caveat retained above.
- **Result classified on evidence ladder; honest limitation text drafted** — met: `diagnostic` tier,
  limitations enumerated in `## Caveats`.

Agent-executable slice criteria (`agent-exec-spec`):

- **Shared protocol + per-bet caps + stop rules frozen and validated (dry-run)** — met:
  `predictive_hard_seeds_v1` fixture + closed-loop gate flags in
  `scripts/validation/run_predictive_success_campaign.py`; validated by
  `tests/validation/test_predictive_success_campaign_tags.py` (2026-07-06: 20 passed).
- **Synthesis harness produces the cross-bet decision table** — met: harness
  (`run_predictive_success_campaign.py` `--help` exit 0; `scripts/tools/campaign_result_store.py`)
  assembles the three bets into the decision table promoted here.
- **All-flat outcome recorded as a publishable negative result** — met (this note + PR #3337).

### Closure decision

All agent-executable and synthesis criteria are met and validated; the portfolio decision is
reached; all three children are closed (two explicitly as negative results). The only unmet item —
a bounded pre-authorized overnight GPU tournament — is maintainer-gated and now **moot** for the
epic's decision, because the diagnostic tournament already returned a consistent negative across all
three levers. Any GPU rerun would be an optional paper-grade *confirmation* of an answered question,
which belongs in the forecast-lane epic (#2835), not as a blocker on this epic. Recommended: close
issue #3215 as a completed diagnostic-tier negative result, mirroring the #3213/#3214 child closures.

**Residual (non-blocking):** if a paper-grade confirmation of the negative result is later required,
run the S20/S30 hard-seed tournament under a maintainer-set GPU budget and promote raw JSONL to a
durable store; carry the near-field-turn-budget carryover signal as the single lever worth a
larger-sample isolation. This is optional follow-up, not remaining epic scope.
