<!-- AI-GENERATED (robot_sf#4879, 2026-07-08) - NEEDS-REVIEW -->

# Non-Transfer Finding (Confirming-Eval Plateau)

## Decision

The closed-loop confirming-eval grid confirms the predictive-planner **non-transfer** finding from
the `#3254`-era work: across four populated prediction checkpoints and five planner-authority
variants, hard-case success stayed on a plateau far below the `0.30` success gate. No checkpoint
choice and no planner-authority knob closes the gap, so the binding constraint is model/data-side,
not planner-authority-side.

## Plateau vs Gate

Populated closed-loop success-rate cells (broad robust seeds 200-229, ~120 episodes/cell; see
`seed_provenance.yaml`):

- Baseline authority, across checkpoints `predictive_proxy_selected_v2_full`,
  `predictive_proxy_selected_v1`, `predictive_rgl_full_v1`, `sweep_h192_mp3_s7_wd5e5`:
  `0.0667`-`0.0833`.
- Near-field speed cap (`nf_speedcap_only`) and `nearfield_turn`: `0.0917`-`0.1` (the only active
  knobs; about `+0.03` absolute / `+33%` relative over baseline, consistent across checkpoints).
- Inert knobs (about baseline): `nf_headings_only`, `nf_horizonboost_only`.
- Required success gate: `0.30`. Observed populated range: `0.0667`-`0.1`.

The single-point `#3254` final eval (`0.08696`, bundle
`issue_3254_predictive_crossing_conflict_13042_2026-06-23`) sits inside this plateau, so the final
run is not an outlier — the plateau is the repeated confirming signal.

## What This Confirms (non-transfer)

1. **Checkpoint choice barely moves baseline.** Four independently selected prediction checkpoints
   all land within `0.0667`-`0.0833` baseline success. Selecting a different trained checkpoint does
   not rescue closed-loop success.
2. **Planner authority does not close the gap.** Only the near-field speed cap lifts success, and
   only to ~`0.1`; richer turn rate, denser action lattices, deeper sequence search, and heading /
   horizon-boost ablations are inert.
3. **The binding constraint is model/data-side.** Because neither checkpoint selection nor planner
   authority rescues success, the closed-loop failure is attributable to the learned prediction
   model / training data not transferring to crossing-conflict closed-loop behavior — the
   predictive-planner **non-transfer** result.

## What This Does NOT Support

This is `diagnostic_only` evidence. It does **not** support:

- benchmark-strength planner-ranking claims,
- a safety / collision-mitigation claim,
- model promotion of any predictive checkpoint,
- any paper-facing result, or
- a quantitative magnitude claim beyond the small `+0.03` near-field speed-cap effect.

## Recommended Next Action

Keep `predictive_near_field_speed_cap` as a minor safety-progress tuning knob, not a success
driver. Stop further planner-authority tuning as a plateau fix; prioritize model-side bets (issue
[#3214] retraining / richer hard-case data) and proxy-vs-ADE checkpoint selection (issue
[#3204]). Do not blind-resubmit the `#3254` config.

[#3204]: https://github.com/ll7/robot_sf_ll7/issues/3204
[#3214]: https://github.com/ll7/robot_sf_ll7/issues/3214

## Evidence Grade

- Evidence tier: `diagnostic`.
- Failure mode: `mechanism_failed` (authority knobs inert) combined with `non_transfer` (checkpoint
  choice inert).
- Paper-facing boundary: state only that the plateau holds across checkpoints and authority knobs
  and that the binding constraint is model/data-side; do not state a success magnitude or a
  planner-improvement claim.
