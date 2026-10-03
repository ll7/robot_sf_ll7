<!-- AI-GENERATED (#10101) - NEEDS-REVIEW -->
# PEDCONTACT Round 3 complete unpruned fit

AI-GENERATED / NEEDS-REVIEW

All 108 settings × dev1001–1003 ×18 cases (5,832 case trials); producer d52a1ffbb30acf390f2e78c45ffed2c5c7ea6b44. Complete before original deadline2026-10-07 18:32UTC. Radius .25/.28/.30, cap2/3m/s, wall A3/6/9, B.04/.08/.16, R.2/.5. Pair calibrated_v2 factor.003/offset.375 and positive stochastic desired N(1.29,.19) held fixed; V6 retains its prescribed source-speed controls1.15/1.42/1.78m/s. No setting or force-model class pruned. These are CALFIT's complete source-bound V1–V6 banks: V7 is an engineering diagnostic without an accepted range, V8 has unresolved distance origin, and V9 is context-only; they are outside the numerical denominator. V2 uses the inherited CALFIT feasible-width construction: small width2r+.05m, large width2.1×.46=.966m, middle their mean. Actual widths differ by radius and are labelled separately for the best settings. This bounded grid does not cover other pair-force families or establish a global optimum.

Rank by passed checks, then summed range residual, then id; residual mixes units and is only a diagnostic tie-break. Denominator: scoreable PASS/FAIL checks; an indeterminate censored bound or unavailable value is excluded. Exact V3 completion flows and exact V6 onsets can remain unmeasured even when their bounds prove FAIL/PASS. The printed V6 [3,3] interval is for the reported lower bound; true onset lies in [3,∞). V3 true completion flow lies in [0, its upper bound], not in the narrow seed-mean interval for that bound. All constituent values and Student95% seed-mean intervals are retained in [full108-point ranking](pedcontact_10101_round3_fit.json).

Numerical leader: 60884e2af9f0: 8/15 scoreable (15 potential), r=0.3, cap=3.0, wall A/B/R=3.0/0.16/0.5. Physical counters: 0 overlap pair-steps and 0 violating case trials. Physical exclusion holds for 108/108 settings. Best shipped radius: b24e1d438596: 7/15 scoreable (15 potential), r=0.28, cap=3.0, wall A/B/R=6.0/0.08/0.2. Fully accepted settings: 0. This conclusion uses corrected V3/V6 and contact-closure fallback; earlier rankings and the V6-pruned zero-survivor claim are withdrawn. The three V6 PASS checks establish only the author-ruled lower bounds, not exact onset timing or human speed dependence.

- 60884e2af9f0: 8/15 scoreable (15 potential), r=0.3, cap=3.0, wall A/B/R=3.0/0.16/0.5
- c1547cf0fec2: 8/15 scoreable (15 potential), r=0.3, cap=2.0, wall A/B/R=3.0/0.16/0.5
- 7538a2c3f966: 8/15 scoreable (15 potential), r=0.3, cap=2.0, wall A/B/R=3.0/0.16/0.2
- b94ced5b1318: 8/15 scoreable (15 potential), r=0.3, cap=3.0, wall A/B/R=6.0/0.16/0.2
- eb3d8e71761a: 8/15 scoreable (15 potential), r=0.3, cap=2.0, wall A/B/R=6.0/0.16/0.2
- d5fa8c3e511b: 8/15 scoreable (15 potential), r=0.3, cap=2.0, wall A/B/R=3.0/0.08/0.5
- b24e1d438596: 7/15 scoreable (15 potential), r=0.28, cap=3.0, wall A/B/R=6.0/0.08/0.2
- efe8107cdcf8: 7/15 scoreable (15 potential), r=0.3, cap=2.0, wall A/B/R=3.0/0.08/0.2
- 70be97719fab: 7/15 scoreable (15 potential), r=0.3, cap=3.0, wall A/B/R=6.0/0.08/0.2
- 01be2d9c3a72: 7/15 scoreable (15 potential), r=0.3, cap=2.0, wall A/B/R=6.0/0.08/0.2

| Item | Best overall | Best shipped radius |
|---|---|---|
| V1/native | 1.29111 [0.828121, 1.75411] PASS | 1.29111 [0.828121, 1.75411] PASS |
| V2/0.65 overall / 0.61 shipped | 0.223564 [0.145072, 0.302056] PASS | 0.203576 [0.123502, 0.28365] PASS |
| V2/0.808 overall / 0.788 shipped | 0.114371 [0.0779672, 0.150774] PASS | 0.0578908 [0.0419543, 0.0738274] FAIL |
| V2/0.966 | 0.0565885 [0.0377175, 0.0754595] FAIL | 0.0328837 [0.0247414, 0.041026] FAIL |
| V3/0.8 | upper bound 0.476283 [0.474086, 0.47848]; censored 3/3 FAIL | upper bound 0.475671 [0.474473, 0.47687]; censored 3/3 FAIL |
| V3/0.9 | upper bound 0.42267 [0.422092, 0.423248]; censored 3/3 FAIL | upper bound 0.42251 [0.421756, 0.423265]; censored 3/3 FAIL |
| V3/1.0 | upper bound 0.380283 [0.379369, 0.381198]; censored 3/3 FAIL | upper bound 0.380063 [0.379678, 0.380448]; censored 3/3 FAIL |
| V3/1.1 | upper bound 0.345627 [0.345016, 0.346237]; censored 3/3 FAIL | upper bound 0.345481 [0.345001, 0.345961]; censored 3/3 FAIL |
| V3/1.2 | upper bound 0.316637 [0.316002, 0.317271]; censored 3/3 FAIL | upper bound 0.316619 [0.316072, 0.317167]; censored 3/3 FAIL |
| V5/diagnostic | 0.530616 [0.486599, 0.574633] PASS | 0.455618 [0.443055, 0.468181] PASS |
| V6/1.15 | ≥3 [3, 3]; censored 3/3 PASS | ≥3 [3, 3]; censored 3/3 PASS |
| V6/1.42 | ≥3 [3, 3]; censored 3/3 PASS | ≥3 [3, 3]; censored 3/3 PASS |
| V6/1.78 | ≥3 [3, 3]; censored 3/3 PASS | ≥3 [3, 3]; censored 3/3 PASS |
| V4/all_data_flow_persons_s width slope | 1.83679 [1.74827, 1.9253] FAIL | 1.81663 [1.80319, 1.83006] FAIL |
| V4/steady_flow_persons_s width slope | 2.83097 [2.76791, 2.89402] PASS | 2.9011 [2.747, 3.0552] PASS |


Best overall: whole-bank 38.748 ms/step; N350 width5 54.431 ms/step. Best shipped radius: whole-bank 18.484 ms/step; N350 width5 47.158 ms/step. Best overall/shipped physical counters: overlap 0/0, wall penetration 0/0, violating case trials 0/0. [Every setting's measured integration cost and counters](pedcontact_10101_round3_fit_runtime.json). These three-seed costs include all integration calls and cold start/congestion; they are not paired off/on benchmarks at fitted parameters. No extrapolation of default warm ratios across the fit.
