<!-- AI-GENERATED (#10101) - NEEDS-REVIEW -->
# PEDCONTACT Round 3 measurements
AI-GENERATED / NEEDS-REVIEW

Four arms; dev1001–1030; physical radius .28m; CALFIT default parameters. Seed-mean 95% Student intervals describe sampling uncertainty. V3 values from incomplete trials are completion-flow upper bounds, not point flows. V6 boundary onsets are right-censored lower bounds ≥3m, not exact onsets. Intervals printed for censored values describe the reported bounds: V6 [3,3] is a bound-estimator interval while true onset lies in [3,∞); incomplete V3 true completion flow lies in [0, its upper bound].

| Item | Off | Contact only | Wall only | Both |
|---|---|---|---|---|
| V1 native | 1.255 [1.183, 1.328] PASS | 1.255 [1.183, 1.328] PASS | 1.255 [1.183, 1.328] PASS | 1.255 [1.183, 1.328] PASS |
| V2 0.61 | unavailable MISSING | unavailable MISSING | 0.07944 [0.07572, 0.08316] PASS | 0.07944 [0.07572, 0.08316] PASS |
| V2 0.788 | unavailable MISSING | unavailable MISSING | 0.03379 [0.03075, 0.03684] FAIL | 0.03379 [0.03075, 0.03684] FAIL |
| V2 0.966 | unavailable MISSING | unavailable MISSING | 0.03114 [0.02745, 0.03483] FAIL | 0.03114 [0.02745, 0.03483] FAIL |
| V3 0.8 | unavailable; censored 30/30 MISSING | upper bound 0.4827 [0.4807, 0.4848]; censored 30/30 FAIL | upper bound 0.4758 [0.4757, 0.476]; censored 30/30 FAIL | upper bound 0.4762 [0.4759, 0.4766]; censored 30/30 FAIL |
| V3 0.9 | unavailable; censored 30/30 MISSING | upper bound 0.4283 [0.4265, 0.43]; censored 30/30 FAIL | upper bound 0.4228 [0.4226, 0.4229]; censored 30/30 FAIL | upper bound 0.4229 [0.4226, 0.4233]; censored 30/30 FAIL |
| V3 1.0 | upper bound 0.4766 [0.4677, 0.4855]; censored 30/30 FAIL | upper bound 0.3846 [0.3838, 0.3854]; censored 30/30 FAIL | upper bound 0.3803 [0.3802, 0.3804]; censored 30/30 FAIL | upper bound 0.3803 [0.3802, 0.3804]; censored 30/30 FAIL |
| V3 1.1 | upper bound 0.3631 [0.3289, 0.3974]; censored 29/30 FAIL | upper bound 0.3837 [0.3525, 0.4149]; censored 25/30 FAIL | upper bound 0.3456 [0.3455, 0.3457]; censored 30/30 FAIL | upper bound 0.3456 [0.3455, 0.3457]; censored 30/30 FAIL |
| V3 1.2 | upper bound 0.3342 [0.2998, 0.3686]; censored 29/30 FAIL | upper bound 0.3365 [0.3096, 0.3634]; censored 28/30 FAIL | upper bound 0.3167 [0.3167, 0.3168]; censored 30/30 FAIL | upper bound 0.3168 [0.3167, 0.3169]; censored 30/30 FAIL |
| V5 diagnostic | 0.8678 [0.8635, 0.8721] FAIL | 0.8678 [0.8635, 0.8721] FAIL | 0.3861 [0.3849, 0.3872] FAIL | 0.3861 [0.3849, 0.3872] FAIL |
| V6 1.15 | ≥3 [3, 3]; censored 30/30 PASS | ≥3 [3, 3]; censored 30/30 PASS | ≥3 [3, 3]; censored 30/30 PASS | ≥3 [3, 3]; censored 30/30 PASS |
| V6 1.42 | ≥3 [3, 3]; censored 30/30 PASS | ≥3 [3, 3]; censored 30/30 PASS | ≥3 [3, 3]; censored 30/30 PASS | ≥3 [3, 3]; censored 30/30 PASS |
| V6 1.78 | ≥3 [3, 3]; censored 30/30 PASS | ≥3 [3, 3]; censored 30/30 PASS | ≥3 [3, 3]; censored 30/30 PASS | ≥3 [3, 3]; censored 30/30 PASS |
| V4 all_data_flow_persons_s width slope | 1.732 [1.716, 1.749] FAIL | 1.722 [1.706, 1.738] FAIL | 1.634 [1.623, 1.645] FAIL | 1.613 [1.6, 1.625] FAIL |
| V4 steady_flow_persons_s width slope | 2.645 [2.619, 2.67] PASS | 2.569 [2.546, 2.593] PASS | 3.171 [3.135, 3.207] FAIL | 3.068 [3.037, 3.098] FAIL |

The denominator is scoreable checks (measured estimates or decisive bounds). An upper bound overlapping the acceptance range is CENSORED and cannot PASS; an unavailable estimate is MISSING. Neither enters the scoreable denominator. At the default both/wall settings, seven checks use point measurements and eight use decisive bounds (five V3 upper bounds plus three V6 lower bounds). Exact complete V3 flows and exact V6 onsets remain unmeasured.

| Arm | Passed / scoreable / potential | Overlap pair-steps | Wall penetration pedestrian-steps | ms/step |
|---|---|---|---|---|
| off | 5/10/15 | 91665533 | 0 | 7.378 |
| contact | 5/12/15 | 0 | 343667 | 11.17 |
| wall | 5/15/15 | 62026876 | 0 | 9.116 |
| on | 5/15/15 | 0 | 0 | 16.55 |

| Warm N | Off ms/step | Both ms/step | Ratio |
|---|---|---|---|
| 60 | 0.8665 | 1.593 | 1.838 |
| 150 | 4.344 | 6.319 | 1.455 |
| 350 | 21.84 | 31.17 | 1.427 |

Warm measurement uses the same real congested V4 frame after one warm-up step, then 20 timed steps. Full-suite runtime includes all measured scenarios and compilation/start-up effects; it must not be represented by the warm ratio.
