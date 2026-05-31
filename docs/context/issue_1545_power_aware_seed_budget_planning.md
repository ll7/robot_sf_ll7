# Issue #1545 Power-Aware Seed-Budget Planning

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1545>

## Goal

Define a conservative, citeable seed-budget methodology for future benchmark issues using only
existing durable campaign summaries. This note is for **planning** benchmark budgets and escalation
paths. It does **not** change benchmark pass/fail gates, and it does **not** promote insufficient
data into post-hoc significance claims.

Canonical benchmark contract references:

- [docs/benchmark_spec.md](../benchmark_spec.md)
- [docs/benchmark.md](../benchmark.md)
- [docs/benchmark_planner_family_coverage.md](../benchmark_planner_family_coverage.md)
- [docs/benchmark_camera_ready.md](../benchmark_camera_ready.md)
- [Issue #691 Benchmark Fallback Policy](issue_691_benchmark_fallback_policy.md)

## Claim Boundary

- Keep the fail-closed policy unchanged: fallback, degraded, partial-failure, failed, and
  `not_available` rows are not benchmark-success evidence.
- Treat this note as a **budgeting and uncertainty** reference, not as a promotion rule.
- Use the formulas below for **effect-size planning before a run**. Do not use them to retroactively
  claim "significance" from a small completed run.
- SNQI remains diagnostic-only on the cited h500 surface because the durable summary records
  `snqi_contract_status="fail"` for the S10/h500 candidate campaign and only `warn` for the Stage A
  fixed-h100 surface.
- The cited compact benchmark bundles predate explicit `benchmark_track` and
  `track_schema_version` metadata unless a row says otherwise. Treat them as
  `legacy_track_unknown` per
  [issue_1721_benchmark_track_metadata_audit.md](issue_1721_benchmark_track_metadata_audit.md), and
  do not aggregate them with track-aware result rows.

## Evidence Basis

| Mechanism | Source issue | Evidence tier | Config / surface | Seeds | Artifacts | Metrics | Verdict | Caveats |
|---|---:|---|---|---:|---|---|---|---|
| S3 nominal/stress primary-row sanity | #1344 | durable compact summary | paired nominal + stress primary rows | 3 | `docs/context/evidence/issue_1344_paired_amv_primary_2026-05-20/*campaign_summary.json` | success, collisions, near misses, SNQI | Useful only as a nominal-sanity / stress-smoke reference | `snqi_contract_status="fail"` and AMV coverage `warn`; not paper-facing |
| S3 broader AMV expansion | #1353 | durable compact summary | nominal, stress, cross-kinematics AMV preflight surfaces | mixed; core still S3-style | `docs/context/evidence/issue_1353_broader_amv_2026-05-26/*/campaign_summary.json` | success, collisions, near misses, SNQI | Useful as a small-slice escalation example only | nominal has 84 episodes, stress 1008, cross-kinematics only 9; not a seed-power study |
| S10 fixed-h100 sensitivity slice | #1454 | durable benchmark summary with fail-closed campaign status | fixed h100 broader robustness | 10 | `docs/context/evidence/issue_1454_stage_a_fixed_h100_2026-05-22/reports/{campaign_summary,statistical_sufficiency,seed_variability_by_scenario}.json` | success, collisions, near miss, `time_to_goal_norm`, SNQI | Useful for uncertainty illustrations on the seven successful rows | campaign-level `benchmark_success=false` because one row failed closed |
| S10 h500 compact benchmark | #1454 | durable benchmark-success summary | scenario-horizon h500 candidate comparison | 10 | `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/reports/{campaign_summary,statistical_sufficiency,seed_variability_by_scenario}.json` plus `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/reports/seed_episode_rows.csv` | success, collisions, near miss, `time_to_goal_norm` / artifact `time_to_goal` alias, SNQI | Best current durable basis for compact-benchmark seed planning | exploratory candidate surface, `paper_facing=false`, `snqi_contract_status="fail"` |
| Seed/scenario failure-mode synthesis | #1462 | durable derived summary | derived from S10/h500 compact benchmark | 10 | `docs/context/evidence/issue_1462_s10_h500_failure_modes_2026-05-24/{summary.json,seed_difficulty_table.csv,planner_scenario_seed_variability.csv}` | seed difficulty, scenario difficulty, timeout-or-unfinished summaries, per-cell seed variability | Good for escalation triggers and seed-instability examples | derived layer only; does not add statistical power beyond the underlying S10 run |

## Planning Approximations

For binary rates such as success, collision, or timeout:

- Planning half-width at 95% confidence:
  `1.96 * sqrt(p * (1 - p) / n)`
- Worst-case half-width at `p = 0.5`:
  `0.98 / sqrt(n)`

Use those values only as **transparent planning approximations**.

Two different `n` values matter:

1. **Per-scenario cell**: one planner on one scenario across `seed_count` seeds.
2. **Pooled planner row**: one planner over a fixed 48-scenario matrix and `seed_count` seeds,
   using `48 * seed_count` episode rows as an optimistic approximation.

The pooled-row approximation is useful for budgeting but is optimistic because scenario mix and
seed effects are not iid. The archived S10/h500 evidence still shows material aggregate seed
instability across the 10 fixed seeds:

- success range across planner-level seed means: median `0.1875`, max `0.2917`
- collision range across planner-level seed means: median `0.0833`, max `0.2708`
- near-miss range across planner-level seed means: median `10.8021`, max `21.4375`
- `time_to_goal_norm` range across planner-level seed means, recomputed from the artifact
  `time_to_goal` column: median `0.0806`, max `0.1182`

Those ranges mean a small apparent delta can still be seed-sensitive even when the pooled episode
count looks large.

## Recommended Tiers

Assume the usual 48-scenario stress matrix unless a narrower matrix is explicitly justified.

| Tier | Seeds per planner row | Approx. episodes per row on 48 scenarios | Expected 95% binary-rate uncertainty | Intended use | Escalate when |
|---|---:|---:|---|---|---|
| smoke | 1 | 48 | no meaningful per-scenario interval; optimistic pooled-row worst case `+-0.142` | CLI/preflight smoke, schema/path checks, fail-closed dependency checks | any benchmark claim, any ranking statement, any safety delta that matters |
| nominal sanity | 3 | 144 | per-scenario worst case `+-0.566`; optimistic pooled-row worst case `+-0.082` | confirm the surface is not obviously broken; large-effect nominal/stress sanity only | planner ordering changes with seed choice, or any target delta is below about `0.15` for rates |
| compact benchmark | 10 | 480 | per-scenario worst case `+-0.310`; optimistic pooled-row worst case `+-0.045` | bounded issue-level comparison, challenger screening, failure-mode mining | close rows, paper-facing language, or any safety/quality conclusion that depends on deltas below about `0.05` to `0.10` |
| paper-facing comparison | 20+ | 960+ | per-scenario worst case `+-0.219` at 20 seeds; optimistic pooled-row worst case `+-0.032` at 20 seeds (`+-0.026` at 30) | pre-specified manuscript or release-facing comparisons with declared primary metrics and effect sizes | if rankings still flip under seed resampling, expand to 30+ or narrow the claim |

### Tier Interpretation

- **Smoke** is only for proving the path runs and fails closed correctly.
- **Nominal sanity** is enough to catch large regressions or obviously broken challenger rows, but
  not enough for close comparisons.
- **Compact benchmark** matches the strongest current durable evidence base in issue #1454. It is
  good enough for bounded benchmark notes and failure-mode work, but the archived S10 seed ranges
  show that it is still too noisy for close paper-facing comparisons.
- **Paper-facing comparison** should be treated as a planning target, not an empirical guarantee
  derived from current bundles. The repository currently has durable S10 evidence, not durable S20
  or S30 comparison evidence.

## Metric-Specific Guidance

### Success rate and collision rate

These are the cleanest metrics for seed-budget planning because they are binary per episode.

Observed durable uncertainty from the S10 campaign summaries:

- S10/h500 per-scenario cell bootstrap half-widths:
  - success: median `0.0`, p90 `0.3`, max `0.3`
  - collision: median `0.0`, p90 `0.3`, max `0.3525`
- S10/fixed-h100 per-scenario cell bootstrap half-widths:
  - success: median `0.0`, p90 `0.2`, max `0.3`
  - collision: median `0.0`, p90 `0.3`, max `0.3525`

Interpretation: even at 10 seeds, single-scenario cells remain noisy enough that they are better
for mechanism-finding than for declaring narrow wins. Prefer campaign-row comparisons for bounded
claims, and escalate beyond S10 when a conclusion depends on small rate deltas.

### Near-miss counts

Near misses are much less stable than binary success/collision rates.

Observed durable uncertainty:

- S10/h500 per-scenario cell near-miss half-widths: median `3.2287`, p90 `22.92`, max `45.32`
- S10/fixed-h100 per-scenario cell near-miss half-widths: median `0.3`, p90 `5.605`, max `14.6`

Interpretation: near-miss counts are overdispersed and horizon-sensitive. Use them as a secondary
diagnostic unless the campaign archives per-seed bootstrap summaries and the effect is large.

### Min distance / clearance

The cited campaign summaries archive aggregate `mean_clearance`, `min_clearance`, and
`min_distance` fields, but the durable seed-variability and statistical-sufficiency summaries do
not provide matching per-seed uncertainty for those metrics.

Policy:

- treat aggregate minimum-distance differences as descriptive unless the archive includes per-seed
  bootstrap or quantile evidence,
- do not interpret tiny clearance differences as meaningful,
- prefer thresholded safety rates (collision, near miss, timeout/unfinished) for seed-budget
  planning when only current durable summaries are available.

### Low-progress / timeout / unfinished

The cited durable bundles do **not** provide planner-by-seed timeout or low-progress intervals.
They do provide scenario-level `raw_timeout_or_unfinished_rate` in the issue #1462 summary, which is
useful for identifying where longer horizons change failure composition.

Policy:

- if future campaigns archive timeout or low-progress as binary per-episode outcomes, plan them like
  success/collision rates,
- for the current archived evidence, use timeout-or-unfinished only as a qualitative escalation
  signal, not as a statistically bounded planner-comparison metric.

### `time_to_goal_norm` and comfort-exposure-style continuous metrics

These metrics are present in the durable summaries and seed-variability tables, but they are not
well approximated by simple binomial formulas.

The benchmark-spec metric is `time_to_goal_norm`. Some issue #1454 artifact summaries expose the
same normalized field as `time_to_goal`, while `seed_variability_by_scenario.json` uses
`time_to_goal_norm`; treat the `time_to_goal` values cited here as that artifact-level alias, not
as a separate raw-time metric.

Observed durable uncertainty:

- S10/h500 per-scenario cell `time_to_goal` artifact half-widths: median `0.0264`, p90 `0.1289`,
  max `0.2409`
- S10/fixed-h100 per-scenario cell `time_to_goal` artifact half-widths: median `0.0`, p90 `0.0213`,
  max `0.1116`

Policy:

- use bootstrap-over-seed-means, not normal-theory shortcuts,
- treat small shifts as unstable until they survive larger seed budgets,
- interpret them alongside success/collision, not in isolation.

### SNQI and SNQI components

Use conservative SNQI boundaries:

- S10/h500: `snqi_contract_status="fail"`, rank alignment `-0.2067`, outcome separation `0.2663`
- S10/fixed-h100: `snqi_contract_status="warn"`, rank alignment `0.3214`, outcome separation `0.2090`

Policy:

- do **not** use SNQI for ranking or significance claims on these archived surfaces,
- do **not** treat component-level changes as benchmark-strengthening unless a future campaign
  archives valid component-level uncertainty and passes the SNQI contract,
- SNQI may still be cited as a diagnostic aggregate when the note explicitly says it is diagnostic.

## When To Escalate

Escalate from a smaller tier to a larger one when any of these conditions holds:

1. The conclusion will be used in a paper-facing, release-facing, or baseline-promotion argument.
2. The observed win/loss depends on rate differences below about `0.05` to `0.10`.
3. Seed-level ranges are comparable to the reported delta.
4. Near-miss, timeout, or progress metrics disagree with success/collision direction.
5. Cross-kinematics or narrow scenario-smoke evidence is being reused as if it were full-matrix
   evidence.
6. SNQI is being invoked to break a tie.
7. A planner row is fallback, degraded, partial-failure, failed, or `not_available`.

## Effect-Size Planning vs Post-Hoc Significance

Use this policy in the following order:

1. **Before the run**, declare the primary metric, comparison surface, and the smallest effect size
   worth detecting.
2. Pick the cheapest tier whose projected uncertainty is smaller than that effect size.
3. **After the run**, report the observed delta together with the tier and the uncertainty caveat.
4. If the result is too close for the tier, escalate the seed budget instead of retrofitting a
   stronger claim.

Do **not** do the following:

- claim significance just because an S3 or S10 run looks directionally positive,
- mix smoke, nominal, and compact surfaces into one pooled significance statement,
- use SNQI contract-fail surfaces to rescue a weak direct-metric comparison,
- reinterpret a fail-closed campaign as benchmark success because some rows completed.

## Do Not Use This Policy For

- changing benchmark pass/fail gates,
- overriding `docs/context/issue_691_benchmark_fallback_policy.md`,
- turning cross-kinematics smoke or narrow AMV smoke into full benchmark evidence,
- making manuscript-strength SNQI claims from the current h500 evidence,
- inferring min-distance significance from aggregate-only clearance fields.

## Recommended Reuse Wording

Use wording close to:

> We used the issue #1545 seed-budget methodology note to choose a compact or paper-facing seed
> tier before running the campaign. The reported deltas should be interpreted as effect-size-planned
> results for that tier, not as post-hoc significance claims beyond the archived uncertainty
> evidence.
