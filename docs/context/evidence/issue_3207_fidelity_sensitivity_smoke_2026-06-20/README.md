# Issue #3207 Fidelity Sensitivity Diagnostic Smoke 2026-06-20

- Status: `diagnostic_smoke`
- Git head: `40702550`
- Baseline variant: `dt_0_10_clean`
- Ranking metric: `min_distance`
- Claim boundary: diagnostic_smoke_not_benchmark_evidence: summarizes a small local same-scenario fidelity sensitivity smoke. It can show wiring, metric drift, and rank-stability calculation behavior, but it does not establish planner ranking, simulator realism, sim-to-real validity, or paper-facing benchmark evidence.

## Variant Summary

| Variant | Planner | Episodes | Seeds | Mean ranking metric |
|---|---|---:|---|---:|
| `dt_0_05_clean` | `simple_policy` | 3 | 101, 102, 103 | 6.17938 |
| `dt_0_05_clean` | `social_force` | 3 | 101, 102, 103 | 5.91157 |
| `dt_0_10_clean` | `simple_policy` | 3 | 101, 102, 103 | 5.25427 |
| `dt_0_10_clean` | `social_force` | 3 | 101, 102, 103 | 4.73614 |
| `dt_0_10_noise` | `simple_policy` | 3 | 101, 102, 103 | 5.25432 |
| `dt_0_10_noise` | `social_force` | 3 | 101, 102, 103 | 4.74053 |
| `dt_0_20_clean` | `simple_policy` | 3 | 101, 102, 103 | 2.89141 |
| `dt_0_20_clean` | `social_force` | 3 | 101, 102, 103 | 2.66883 |

## Rank Stability

| Variant | Kendall tau | Rank flips | Stable? |
|---|---:|---:|---|
| `dt_0_05_clean` | 1 | 0 | `True` |
| `dt_0_10_noise` | 1 | 0 | `True` |
| `dt_0_20_clean` | 1 | 0 | `True` |

This smoke is diagnostic only. It records sensitivity wiring, small-sample metric drift,
and rank-stability calculations; it is not benchmark-strength evidence.
