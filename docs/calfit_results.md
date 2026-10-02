<!-- AI-GENERATED NEEDS-REVIEW -->
# CALFIT historical input-defect screen — 2026-10-02

**Superseded by the author protocol-repair ruling.** The V3 starts were inadmissible; this preserved screen is suite-defect evidence, not an accepted model safety/flow result. See [the repaired-protocol continuation](./calfit_protocol_repair.md).

Historical status (superseded): NO-PASS; AUTHOR_DECISION_REQUIRED. Draft [#10094](https://github.com/ll7/robot_sf_ll7/pull/10094), stacked on draft [#10075](https://github.com/ll7/robot_sf_ll7/pull/10075). The author ruling on feasible V2 and uniform tolerances is implemented; those decisions remain settled.

114 pre-recorded coarse settings plus exactly one 26-setting refinement completed (7,560 admitted case/seed acquisitions). Screen seeds were 1001–1003, from the permitted 1001–1030 dev set. None passed, so the time-box stops here; no additional tuning or 30-seed finalist run. A three-seed screen is not a full dev-bank pass claim.

The coarse grid is 3 radii (.25/.28/.30) × 2 independent caps (2/3 m/s) × 19 laws/settings: legacy factor10/offset−.57; calibrated factors .0003/.003/.03 × offsets .25/.375; gradient factors .0003/.001/.003, offset=radius; exponential A5/15/40 × B.02/.04/.08. Wall sigma0; positive N(1.29,.19) desired-speed assignment without upper clipping. Negative draws are redrawn; integration cap is separate. Refinement was recorded before submission, around four law-diverse censored boundary parents, factor×.5/1/2 and radius±.01, with coarse duplicates removed.

Numeric dense-wall/narrow-flow Pareto front: **empty**, because no setting has all 15 complete finite-N narrow-flow estimates. This does not mean a zero-flow front. Individual observations and conditional censor bounds remain diagnostics.

**Invalidated historical inference (inadmissible inputs; not a model conclusion):** The two empirical targets conflict on **every tested setting under this suite protocol**: dense-wall penetration0 m versus 1.2 m narrow specific flow within [1.576,2.364] persons/(m·s). 126/140 settings have positive dense-wall penetration; all remaining14 zero-wall settings have eventual finite-N cohort flow bounded above by .316733–.328129 persons/(m·s) at1.2 m, even if the censored people eventually exit. Thus every tested law fails at least one of these two targets without substituting partial flow for the estimator. This is a finite-grid result on the inherited starts, not impossibility outside the tested settings or a clean attribution to force law. Relaxing the numeric flow lower bound enough would require more than83.34% below the source mean1.97 and would still leave independent physical failures; recommend repairing the protocol equivalent instead, keeping safety and empirical flow targets.

| Law | settings | minimum dense-wall penetration m | zero-wall settings | complete narrow settings | largest observed narrow bank /15 |
|---|---:|---:|---:|---:|---:|
| calibrated_v2 | 44 | 0.000000 | 14 | 0 | 4 |
| exponential_edge | 59 | 0.050966 | 0 | 0 | 4 |
| gradient_v3 | 26 | 0.106565 | 0 | 0 | 12 |
| legacy_v1 | 11 | 0.249279 | 0 | 0 | 0 |

Every one of 2,100 V3 case/seed records starts with **54 overlapping pairs**, for every tested law and radius, before wall forces act. All also contain post-step overlaps. Thus the inherited 10×6 holding lattice and zero rigid-disc overlap cannot both be satisfied by any tested wall-law setting. This is a suite input-admissibility conflict, not proof that the empirical dense-wall and narrow-flow targets are intrinsically incompatible. No scalar flow/wall tolerance relaxation removes that independent failure.

Best passing candidate: none. Safety-first diagnostic (fewest physical-failure rows, smallest dense-wall penetration, most complete V3 observations, then fewer missing/numeric failures and ID): `5c1f68bcf0f0`, calibrated_v2 factor .003, offset .375 m, sigma0, radius .25 m, cap2 m/s, desired N(1.29,.19). It has 30 physical-failure rows, zero dense-wall penetration, 3,379,578 dense and 2,849,312 narrow pair-step overlaps. Initial V3 minimum centre distance .419327 m < diameter .50 m.

The table reports population means and signed estimate−target differences; a partial mean is explicitly missing for admission. Outside-distance is to the nearest acceptance boundary, not the signed target difference. All sources are those pinned in `robot_sf/research/pedestrian_acceptance.py`; every tolerance states **author-delegated engineering tolerance, 2026-10-02**.

| gate / variant | estimate | target | Δ | tolerance range | outside distance | status | observed/attempted |
|---|---:|---:|---:|---|---:|---|---|
| [V1](https://doi.org/10.1098/rspb.2009.0405) / native | 1.291114 | 1.290000 | 0.001114 | [1.100000, 1.480000] | 0.000000 | PASS | 3/3 |
| [V2](https://doi.org/10.1371/journal.pone.0124695) / 0.55 | unidentified | 0.207826 | unidentified | [0.096522, 0.333913] | unidentified | MISSING | 0/3 |
| [V2](https://doi.org/10.1371/journal.pone.0124695) / 0.758 | unidentified | 0.140000 | unidentified | [0.060000, 0.240000] | unidentified | MISSING | 0/3 |
| [V2](https://doi.org/10.1371/journal.pone.0124695) / 0.966 | unidentified | 0.140000 | unidentified | [0.060000, 0.240000] | unidentified | MISSING | 0/3 |
| [V3](https://arxiv.org/abs/physics/0702004) / 0.8 | unidentified | 1.610000 | unidentified | [1.288000, 1.932000] | unidentified | MISSING | 0/3 |
| [V3](https://arxiv.org/abs/physics/0702004) / 0.9 | unidentified | 1.860000 | unidentified | [1.488000, 2.232000] | unidentified | MISSING | 0/3 |
| [V3](https://arxiv.org/abs/physics/0702004) / 1.0 | unidentified | 1.900000 | unidentified | [1.520000, 2.280000] | unidentified | MISSING | 0/3 |
| [V3](https://arxiv.org/abs/physics/0702004) / 1.1 | 0.620272 | 1.930000 | -1.309728 | [1.544000, 2.316000] | 0.923728 | MISSING | 2/3 |
| [V3](https://arxiv.org/abs/physics/0702004) / 1.2 | unidentified | 1.970000 | unidentified | [1.576000, 2.364000] | unidentified | MISSING | 0/3 |
| [V5](https://doi.org/10.1016/j.gaitpost.2007.03.015) / diagnostic | 0.865624 | 0.500000 | 0.365624 | unspecified | unidentified | UNSPECIFIED | 3/3 |
| [V6](https://doi.org/10.1371/journal.pone.0089589) / 1.15 | 2.938622 | 2.100000 | 0.838622 | unspecified | unidentified | UNSPECIFIED | 3/3 |
| [V6](https://doi.org/10.1371/journal.pone.0089589) / 1.42 | 2.928919 | 2.400000 | 0.528919 | unspecified | unidentified | UNSPECIFIED | 3/3 |
| [V6](https://doi.org/10.1371/journal.pone.0089589) / 1.78 | 2.930237 | 2.700000 | 0.230237 | unspecified | unidentified | UNSPECIFIED | 3/3 |
| [V4](https://doi.org/10.1016/j.trpro.2014.09.005) / all_data_flow_persons_s width slope | 1.734490 | 2.300000 | -0.565510 | [1.840000, 2.760000] | 0.105510 | FAIL | 3/3 |
| [V4](https://doi.org/10.1016/j.trpro.2014.09.005) / steady_flow_persons_s width slope | 2.663643 | 2.500000 | 0.163643 | [2.000000, 3.000000] | 0.000000 | PASS | 3/3 |

Physical table: dense-wall maximum0 m / target0 / tolerance0 / PASS; initial V3 pairs54 / target0 / tolerance0 / FAIL; post-step dense/narrow pair overlaps3,379,578 / 2,849,312 / target0 / tolerance0 / FAIL. Overall exit3 (hard physical failure). V1 tau .448142 s versus source .54, Δ−.091858 s, informational because the ruling gates desired speed. V2 speed units m/s; V3/V4 persons/(m·s); V5/V6 m. The original .9-shoulder-width case is “not reproducible with rigid discs (shoulder rotation)”, recorded and excluded, not a failed gate.

At widths1.0/1.1/1.2, the diagnostic's eventual cohort mean finite-N flows are bounded above by .483127/.529090/.317437, below lower tolerances1.520/1.544/1.576. Bounds use complete observations where available and N/[width×(observed end−first crossing)] when censored. At .8/.9 no first crossing exists; eventual finite-N flow is unidentified, not estimated as zero. V5 .5 m and V6 head-on Table1 means2.1/2.4/2.7 m lack verified corresponding SDs/bands; pooled-angle Figure6 error bars are not substituted. No invented spreads.

## New evidence decision packet

AUTHOR_DECISION_REQUIRED; owner: author/domain reviewer; classification: geometry decision-ready, source spreads evidence-required. Already-ruled V2/tolerance choices are not reopened.

- What happened / why it matters: the suite places discs inside each other before measuring flow, so changing wall forces cannot produce a valid input. Calling this an intrinsic wall/flow incompatibility would overstate the experiment.
- Question / residual judgment: authorize source-equivalent nonoverlapping V3 initialization, or preserve literal initialization and park rigid-disc admission? This changes protocol equivalence, which measurement alone cannot authorize.
- Tested alternatives: an isolated static 60-disc, 3.3 persons/m², 20/20/20 holding-section layout gives minimum pair distance .561521 m and zero overlaps across dev1001–1030 for radii .25/.28; .30 fails that proposal (150 pairs/seed). This is geometry proof only, no model-performance claim. The literal layout yields54 pairs for all2,100 measured V3 records; preserving it cannot pass physical admission.
- Recommended answer `FEASIBLE_STARTS`: relax literal starting-lattice equivalence, keep zero penetration/overlap and ±20% empirical flow tolerances. A repaired start may still fail later contacts or flow; no passing promise.
- Strongest alternative `PARK_LITERAL`: appropriate if equivalent starts are scientifically unacceptable; preserve protocol, report NO-PASS, stop further calibration. Default if deferred: PARK_LITERAL; no new jobs.
- Automatic consequences: FEASIBLE_STARTS authorizes a separate opt-in initialization patch, geometry regression and a newly authorized bounded dev-seed campaign at supported radii after source-range resolution; PARK_LITERAL archives this result and leaves admission blocked. Both keep draft/defaults and existing releases unchanged; neither authorizes merge or release.
- Claim/evidence effect: future repaired-input runs would be documented equivalents, separate from this immutable literal-input screen. Exact patch: opt-in V3 holding initializer only, preserve 60 people, holding geometry/density/counts, feasible footprints; no rotating/compressible bodies. Downstream: issues10074/10061 admission remains blocked. Deferral cost: no calibrated rigid-disc claim. Valid tokens: FEASIBLE_STARTS, PARK_LITERAL. Reopen only on an author ruling or verified source evidence.
- Source-spread blocker: retain unspecified V5/V6 acceptance until matching primary-source SDs/bands are verified, or the author explicitly supplies engineering bands as a new exception. The implemented full-bank synthetic positive control passes with test-only spreads; removing them gives exit5. These are software reachability proofs, not scientific alternatives or published spreads.

## Verification and custody

Final code policy head `fd7d2cadb9d9f2327279481a813f6f8491e11b36`; immutable producer heads `1881243fe1d7a1ce1c9d5f3846fed3ceaf9aaf68` and exponential empty-wall fix `ff914568c20860eb30ca62eb5102d71f7d3bf576`. Final reanalysis applies the separately pinned policy without overwriting producer receipts. Lossless NPZ, full config/seeds, installed/source force hashes, Slurm receipts, raw manifests, failed attempts and logs are preserved under `/home/luttkule/calfit_evidence/ruling_2026_10_02/`; independent workstation verification covers all140 settings and7,560 raw traces.

41 focused tests pass; audited default suite10,970 passed,18 existing skips,7 xfails; protected-seed guard empty. Held-out stepping files excluded explicitly as recorded in the safe-selection receipt. Exact code-head coverage policy95.8/preflight98.3/search82.3/suite90.0%; lint/format, routing, broad exceptions, seed diff, evidence integrity, registry and PR contract pass. Strict curated Sphinx fails with inherited `Argument list too long`, reproduced on the stack base; no docs-build success.

Behavior probes: known-answer V1–V6 estimators all return expected finite values; full synthetic bank PASS0 with test-only spreads. Before author-policy fixes: ideal V1 exit5→0, missing+physics exit2→3, V2 seven original widths→three feasible, initial-overlap later-clear exit0→3, appended excluded original case exit3→0, policy fingerprint5→6 files. Empty-wall exponential V1 Numba type error→zero wall force. Real dev1001 desired draw1.486464→1.467141 m/s; independent .2 cap measured .200000 m/s without clipping the desired draw. Regression failure/pass logs retained. V6 controlled speeds and bottleneck goal-release suspicions were refuted from actual source/raw arrays; no speculative repair.

Slurm jobs16194/16243 (fixed-cause exponential retry),16318 (resource repair),16325 (single refinement); submitted from imech192,2CPUs/task. Operational exception: aggregate CALFIT allocation reached36CPUs for69s,12:17:30–12:18:39 CEST, from prematurely raising throttle with two original tasks active. Corrected immediately to14-task throttle; two cancelled partial tasks retained and retried identically after original-job dependency. Allocation stayed≤32 afterward; queued sealed0.0.8 priority checks found none. This breach is disclosed, not claimed compliant. No retired/sealed episodes; no protected artifacts, default changes, merge or ready transition.
