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

## Achieved Precision: Headline Collision Contrasts

| Planner Pair | Observed RD | CI Width | 95% CI | MRRD (practical) |
| --- | --- | --- | --- | --- |
| goal vs orca | 0.3192 | 0.2364 | [0.2015, 0.4379] | 0.1380 |
| guarded_ppo vs orca | 0.1730 | 0.2618 | [0.0355, 0.2973] | 0.1580 |
| hybrid_rule_v3_fast_progress_static_escape vs orca | -0.0769 | 0.2127 | [-0.1806, 0.0321] | 0.1240 |
| hybrid_rule_v3_fast_progress_static_escape_continuous vs orca | -0.1339 | 0.1663 | [-0.2191, -0.0527] | 0.1060 |
| ppo vs orca | -0.0178 | 0.2496 | [-0.1455, 0.1041] | 0.1480 |
| prediction_planner vs orca | 0.2148 | 0.1736 | [0.1271, 0.3007] | 0.1080 |
| predictive_mppi vs orca | 0.4073 | 0.2370 | [0.2806, 0.5176] | 0.1470 |
| risk_dwa vs orca | 0.5055 | 0.2054 | [0.3969, 0.6023] | 0.1290 |
| sacadrl vs orca | 0.3948 | 0.2424 | [0.2726, 0.5150] | 0.1430 |
| scenario_adaptive_hybrid_orca_v1 vs orca | -0.0671 | 0.2023 | [-0.1633, 0.0390] | 0.1170 |
| scenario_adaptive_hybrid_orca_v2_collision_guard vs orca | -0.0712 | 0.2038 | [-0.1682, 0.0356] | 0.1180 |
| social_force vs orca | 0.2130 | 0.3297 | [0.0540, 0.3837] | 0.1800 |
| socnav_sampling vs orca | 0.4227 | 0.2407 | [0.3015, 0.5422] | 0.1420 |

## Key Distinction: Achieved Precision vs. Post-Hoc Adequacy

- **Achieved precision** (CI width) describes the resolution of the
  interval estimate actually obtained from the data and design.
- **Minimum resolvable risk difference (MRRD)** is the smallest true
  effect that the design could resolve as practically separable,
  derived from the bootstrap standard error. It is a property of the
  design (family count, cell count, event rate), not of the observed
  effect size.
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

## Reproducibility

```bash
uv run python scripts/analysis/derive_retrospective_precision_issue_6592.py \
  --repo-root .
```

All artifacts are deterministic given the frozen rows and seeded RNG.
See `SHA256SUMS` for byte-level verification.
