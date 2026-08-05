<!-- AI-GENERATED (robot_sf#6592) - NEEDS-REVIEW -->
# Issue #6592 Retrospective Precision Derivation

This directory contains the deterministic achieved-precision and
minimum-resolvable risk-difference packet derived from the frozen
`0.0.3.post1` successor rows used by issue #5351.

> [!IMPORTANT]
> **Claim boundary:** This is a design-sensitivity diagnostic.
> It is NOT a post-hoc observed-data adequacy computation,
> NOT prospective sizing, and NOT a claim that 30 seeds were
> adequate for any target effect. Output remains
> `blocked_review_pending` and promotes no benchmark, paper, or
> dissertation claim automatically.

## What This Packet Reports

| Element | Value |
| --- | --- |
| Estimand | Paired risk difference in binary outcome rates between two planner arms on matched scenario-seed cells |
| Comparison unit | matched planner-scenario-seed cell |
| Outer resampling unit | one-stage scenario-family cluster bootstrap |
| Family count (n) | 35 |
| Cells per pair | 1440 |
| Confidence level | 0.95 |
| Interval method | equal-tailed percentile bootstrap |
| Bootstrap samples | 2000 |
| Family mapping SHA-256 | edd5dbed94bc4795255e7728e627fe8fb3282ab5efde8f64dfb92181758ef510 |

## Achieved Precision: Headline Collision Contrasts

| Planner Pair | Observed RD | CI Width | 95% CI | MRRD (observed direction) | Tail |
| --- | --- | --- | --- | --- | --- |
| goal vs orca | 0.3194 | 0.2364 | [0.2015, 0.4379] | 0.1376 | positive |
| guarded_ppo vs orca | 0.1764 | 0.2618 | [0.0355, 0.2973] | 0.1575 | positive |
| hybrid_rule_v3_fast_progress_static_escape vs orca | -0.0778 | 0.2127 | [-0.1806, 0.0321] | 0.1290 | negative |
| hybrid_rule_v3_fast_progress_static_escape_continuous vs orca | -0.1347 | 0.1663 | [-0.2191, -0.0527] | 0.1012 | negative |
| ppo vs orca | -0.0139 | 0.2496 | [-0.1455, 0.1041] | 0.1419 | negative |
| prediction_planner vs orca | 0.2146 | 0.1736 | [0.1271, 0.3007] | 0.1077 | positive |
| predictive_mppi vs orca | 0.4118 | 0.2370 | [0.2806, 0.5176] | 0.1467 | positive |
| risk_dwa vs orca | 0.5083 | 0.2054 | [0.3969, 0.6023] | 0.1286 | positive |
| sacadrl vs orca | 0.3986 | 0.2424 | [0.2726, 0.5150] | 0.1422 | positive |
| scenario_adaptive_hybrid_orca_v1 vs orca | -0.0681 | 0.2023 | [-0.1633, 0.0390] | 0.1261 | negative |
| scenario_adaptive_hybrid_orca_v2_collision_guard vs orca | -0.0722 | 0.2038 | [-0.1682, 0.0356] | 0.1268 | negative |
| social_force vs orca | 0.2104 | 0.3297 | [0.0540, 0.3837] | 0.1790 | positive |
| socnav_sampling vs orca | 0.4257 | 0.2407 | [0.3015, 0.5422] | 0.1413 | positive |

## Event-Rate Sensitivity

- The grid uses independent Bernoulli null-arm draws at each
  baseline event rate, so its expected paired risk difference is zero.
- It reuses the observed matched-cell to scenario-family assignment
  for the one-stage cluster bootstrap.
- These rows are synthetic design-sensitivity calibration, not
  observed evidence and not a claim about any unseen event rate.

## Key Distinction: Achieved Precision vs. Post-Hoc Adequacy

- **Achieved precision** (CI width) describes the resolution of the
  interval estimate actually obtained from the data and design.
- **Minimum resolvable risk difference (MRRD)** is the smallest true
  effect that the design could resolve as practically separable,
  derived from the shifted bootstrap percentile bounds. Positive
  effects use the lower tail; negative effects use the upper tail.
  The contrast table reports the tail matching the observed RD sign
  and the JSON retains both tails plus their worst case.
- **Statistical MRRD** is the zero-threshold normal approximation
  `z_(1-alpha/2) * bootstrap_se`; the practical normal approximation
  adds the declared threshold. Neither quantity is a CI width.
- Sensitivity rows have no observed effect direction, so their
  reported practical MRRD is the larger of the positive and negative
  shifted-tail values.
- This packet does NOT report any post-hoc observed-data adequacy
  metric. Such metrics are monotone transformations of the p-value
  and carry no information beyond what the p-value already provides.
- This packet does NOT claim that the 30-seed design was chosen
  via a prospective sizing calculation for any target effect.

## Frozen Input Provenance

- Release: `0.0.3.post1`
- Publication commit: `ded9027d2928512c14bc241397e0ab1d8f586654`
- Rows SHA-256: `c45c2ed8defdadaf47c001277e6bf9ca0c2238c101570d1d64be8015060febea`
- Total rows: 20160
- Arms: 14
- Rows per arm: 1440
- Families: 35

## Material Exclusions

- **rare_event**: The cluster bootstrap percentile interval is not validated for event rates near zero; contrasts where both arms have near-zero event rates produce degenerate intervals that cannot support precision claims
- **family_generalization**: The 35 scenario families are the resampling unit, not a random sample from a super-population; precision statements apply to these specific families, not to unseen scenario types
- **non_independent_interpretation**: Within-family correlation is preserved by the cluster bootstrap; treating individual cells as independent would artificially narrow intervals and overstate precision
- **prospective_sizing**: The 30 seeds per scenario were not chosen via a prospective sizing calculation for any specific target effect; the MRRD is a retrospective design-sensitivity measure, not a prospective adequacy claim

## Multiplicity Inference

- Method: `holm_step_down` over 39 exposed contrasts at alpha=0.05
- Test: exact two-sided McNemar test on matched binary outcomes
- Each contrast row in the JSON report records its raw p-value, Holm-adjusted p-value, and family-wise rejection decision.
- Holm is applied to inferential decisions only. The descriptive percentile precision intervals and MRRDs in the table above are not multiplicity-adjusted and are not changed by these decisions.


## Reproducibility

```bash
uv run python scripts/analysis/derive_retrospective_precision_issue_6592.py \
  --repo-root .
```

All artifacts are deterministic given the frozen rows and seeded RNG.
See `SHA256SUMS` for byte-level verification.
