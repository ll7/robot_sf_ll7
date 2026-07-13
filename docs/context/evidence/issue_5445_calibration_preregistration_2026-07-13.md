<!-- AI-GENERATED (issue #5445 calibration preregistration) - NEEDS-REVIEW -->

# Issue #5445 — Matched collision-risk calibration: preregistration + fixture evidence

**Status:** evidence (API + fixture). **Evidence grade:** diagnostic / self-consistency; NOT
calibrated benchmark risk for the simulator distribution, never a real-world risk claim.

**Claim boundary.** The realized collision labels used here are drawn from an *explicit, declared*
constant-velocity Gaussian forecast model (optionally misspecified per scenario family), not from
full simulator rollouts. This packet therefore (a) validates the calibration machinery, (b)
demonstrates self-consistency of the constant-velocity Monte Carlo estimator against its own
declared model, and (c) proves the machinery *detects* miscalibration when the ground-truth
distribution is deliberately mismatched. A real-distribution calibration run requires eligible
simulator traces with action-conditioned labels and is **out of scope** until an approved compute
packet exists (see the stop rule below).

Depends on / builds atop the #5444 API merged in PR #5458
(`robot_sf/research/collision_risk/`). Integrates the comparison intent of #1472 (learned risk) and
#5307 (planner) without duplicating either; both are reported as `unavailable` estimator rows
because no learned/multimodal surface is merged in-repo yet.

## Preregistration (fixed before scoring)

Encoded in `configs/analysis/issue_5445_matched_calibration.yaml` and copied verbatim into the
report `provenance` block so the comparison cannot be tuned post hoc:

- **Target distribution:** the estimator's own declared constant-velocity Gaussian model for the
  in-model families; a deliberately misspecified model (noisier or biased) for the misspecified
  families.
- **Collision predicate:** robot/actor disc footprints touch (segment-minimum centre distance ≤
  summed radii) at any horizon step — the *identical* geometry the estimator uses, so labels and
  predictions live on the same predicate.
- **Horizons:** `H = 20` steps at `dt = 0.1 s` (2.0 s).
- **Planner rows:** one planner-agnostic constant-velocity drive-through action family.
- **Discovery/calibration split:** none — labels are i.i.d. from the declared model, unit weights,
  no case-control enrichment (the harness supports per-sample importance weights for enriched real
  traces, unused here).
- **Prevalence/weighting:** natural base rate, unit weights, no prevalence correction.
- **Sample count:** 880 matched scenarios (4 families × 220).
- **Compute cap:** CPU-only fixture generation + scoring; no campaign, no Slurm/GPU.
- **Stop rule:** stop before any real-distribution run until eligible simulator traces,
  target-distribution provenance, and action-conditioned labels exist; stop promotion of an
  estimator whose calibration CI includes the simple baseline while its runtime is worse.

## Reported metrics (per the acceptance criteria)

Brier score, log loss, reliability data with bin counts, expected calibration error (ECE) and its
bootstrap CI, area under the precision-recall curve (average precision), false-negative rate at
predeclared thresholds, time-to-warning, horizon monotonicity, and action sensitivity. Deterministic
warnings (TTC / velocity-obstacle / reachability) are graded **only** as rankings/warnings (AP, FNR,
time-to-warning) and are never placed on a probability reliability curve, per the comparison
contract.

## Fixture result (seed 20260713, this commit)

Reproduce with:

```bash
uv run python scripts/analysis/collision_risk_calibration_report.py \
  --config configs/analysis/issue_5445_matched_calibration.yaml --print-summary
```

Overall prevalence 0.633.

| Estimator | Kind | ECE (CI95) | Brier (baseline) | Log loss | AP | Latency p95 | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `constant_velocity_mc` | probabilistic | 0.070 (0.055–0.102) | 0.180 (0.232) | 0.55 | 0.872 | 5.2 ms → online | revise |
| `deterministic_ttc` | warning | n/a (ranking only) | n/a | n/a | 0.828 | online | revise |
| `multimodal_forecast_mc` | — | unavailable (no in-repo multimodal sampler) | | | | | unavailable |
| `learned_risk_1472` | — | unavailable (model not merged) | | | | | unavailable |

**Stratified calibration (constant_velocity_mc, ECE by family):**

| Family | n | prevalence | ECE |
| --- | --- | --- | --- |
| `in_model_low_density` | 220 | 0.464 | 0.074 |
| `in_model_high_density` | 220 | 0.791 | 0.049 |
| `misspecified_biased` | 220 | 0.673 | 0.075 |
| `misspecified_underconfident` | 220 | 0.605 | 0.166 |

## Reading of the fixture result (honest, bounded)

- **Self-consistency holds:** in-model families are close to calibrated (ECE ≈ 0.05–0.07), and the
  first-passage CDF is monotone in the horizon for 100% of samples.
- **Miscalibration is detected** where it is strong: the *underconfident* misspecification (ground
  truth noisier than assumed) drives ECE to 0.166, well above the in-model band.
- **A weak shift is only weakly detectable:** the *biased* misspecification at this magnitude
  (ECE 0.075) is comparable to the sampling noise on the moderate-prevalence in-model family
  (0.074). This is reported as-is, not tuned away — aggregate ECE is a blunt instrument for small
  distributional shifts on a skewed base rate.
- **Overall verdict `revise`, not `use online`:** on a blended packet that is 50% misspecified, the
  estimator's whole-dataset ECE (0.070) sits in the revise band and its Brier (0.180) beats the
  constant-prevalence baseline (0.232). Latency is comfortably online (p95 ≈ 5 ms ≪ 100 ms
  deadline). The deterministic warning ranks contacts usefully (AP 0.828) but at this prevalence
  cannot clear the AP/prevalence ratio bar, so it too reads `revise`.
- **Warning timeliness caveat:** mean lead time is slightly negative (≈ −0.17 s), i.e. warnings on
  average fire around/just after the realized first-contact step for these short 2 s horizons; a
  real deployment would need an earlier warning threshold. Recorded, not hidden.

## Out of scope (explicit)

No full benchmark campaign run, no Slurm/GPU submission, no paper/dissertation claim edits, and no
change to any benchmark metric semantics. Multimodal-forecast and learned-risk estimator rows remain
`unavailable` until their upstream models merge.
