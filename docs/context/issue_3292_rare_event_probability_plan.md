# Issue #3292 Rare-Event Probability Estimation Plan (2026-06-21)

Status: proposal / method decision only. This note does not run a benchmark campaign, does not
estimate any Robot SF failure probability, and does not promote adversarial stress cases to
benchmark or paper-facing evidence.

Related surfaces:

- GitHub issue: <https://github.com/ll7/robot_sf_ll7/issues/3292>
- Safety/falsification roadmap: [issue_1272_validation_falsification_strategy.md](issue_1272_validation_falsification_strategy.md)
- Stress/uncertainty schema: [issue_1434_stress_uncertainty_coverage_schema.md](issue_1434_stress_uncertainty_coverage_schema.md)
- Adversarial search design: [issue_1433_adversarial_edge_case_search_design.md](issue_1433_adversarial_edge_case_search_design.md)
- Adversarial campaign manifest: [issue_1500_adversarial_manifest.md](issue_1500_adversarial_manifest.md)
- Adversarial generation roadmap: [issue_2468_adversarial_generation_roadmap.md](issue_2468_adversarial_generation_roadmap.md)
- Current code surfaces: `robot_sf/adversarial/certification.py`,
  `robot_sf/adversarial/batch_certification.py`,
  `scripts/tools/run_adversarial_manifest_smoke.py`, and
  `robot_sf/benchmark/stress_uncertainty_coverage.py`

## Decision

Robot SF stress and falsification campaigns may report probability-like estimates only when the
campaign predeclares a target sampling distribution, an event predicate, denominators, exclusion
rules, and uncertainty interval before execution. A campaign that adaptively searches for failures
without a known sampling probability for each executed candidate may report found failure cases,
coverage/quality diagnostics, and replayable counterexamples, but not calibrated failure
probabilities.

For the first compact pilot, use a direct binomial Monte Carlo estimator over a deliberately narrow
and predeclared stress distribution. The estimate is:

```text
p_hat = k / n
```

where `k` is the number of executable episodes satisfying the predeclared failure predicate and
`n` is the number of executable, eligible episodes drawn from the frozen target distribution.
Report a two-sided 95% Wilson score interval as the primary interval and an exact Clopper-Pearson
interval as a conservative appendix when `k <= 5` or `k >= n - 5`.

This choice is not the most sample-efficient rare-event method. It is the lowest-risk estimator for
Robot SF right now because existing adversarial tooling already supports candidate generation,
pre-planner rejection, row classification, and fail-closed availability reporting, while it does
not yet preserve proposal densities or likelihood ratios required for calibrated importance
sampling.

## Probability Language Gate

Use probability language only when all of these conditions are true:

- **Target distribution**: the report names the scenario family, map/template, planner row,
  parameter ranges, seed policy, and sampling law. If the target is conditional on validity, say so
  explicitly, for example "probability under certified-valid crossing/TTC stress draws."
- **Frozen event predicate**: the failure event is a Boolean predicate over episode outputs before
  the run starts, such as `outcome.collision_event == true` within a fixed horizon.
- **Known denominator**: the report exposes proposed candidates, rejected invalid/duplicate
  candidates, simulation errors, fallback/degraded rows, executable eligible episodes, and failures.
- **No adaptive enrichment without correction**: any adaptive sampler, minimizer, cross-entropy
  updater, or adversarial policy must either provide valid sample weights for the target
  distribution or restrict itself to failure discovery language.
- **Fail-closed exclusions**: fallback, degraded, `not_available`, simulation-error, unknown
  metric-source, and post-hoc predicate rows are excluded from probability estimates and reported
  as caveats.
- **Uncertainty first**: every probability-like result reports numerator, denominator, confidence
  method, interval, sampling assumptions, and caveats next to the point estimate.

If any condition is missing, the campaign may say "found `k` failures in `m` searched candidates"
or "observed failure fraction under this diagnostic sample," but it must not say "the failure
probability is..." or imply operational risk.

## Method Comparison

| Method | Feasibility in Robot SF now | Strength | Main risk | Decision |
|---|---|---|---|---|
| Direct Monte Carlo over a frozen stress distribution | High. Existing manifest smoke and batch-certification paths can draw bounded candidates and classify executable rows. | Calibrated for the declared distribution with simple binomial intervals. Easy to audit and reproduce. | Inefficient for very rare events; a compact pilot may only produce a loose upper bound. | Use for the first pilot. |
| Importance sampling | Medium later. Requires a target distribution, proposal distribution, per-sample likelihood ratio, weight diagnostics, and effective sample size. | Can estimate rare-event probabilities with fewer target-distribution samples when the proposal is well chosen. | Biased or meaningless if proposal densities are unavailable, invalid-candidate conditioning is hidden, or high-variance weights dominate. | Defer until manifests record proposal density and likelihood-ratio fields. |
| Cross-entropy method | Medium for discovery, low for calibrated estimates today. Search/Optuna-style adaptation is plausible, but current artifacts are budget/quality diagnostics. | Efficiently finds failure regions and can train a good importance-sampling proposal. | Adaptive samples are not target-distribution draws; raw failure fraction is an optimizer yield, not a probability. | Use only as a proposal-training or failure-discovery stage until followed by weighted or held-out estimation. |
| Adaptive stress testing | Low for calibrated estimates now. The AST framing is useful for sequential decision falsification, but Robot SF does not yet have a verified AST MDP, adversary action model, or trajectory likelihood accounting. | Strong for finding high-likelihood failure trajectories when the disturbance model is explicit. | Without a disturbance model and trajectory probability, AST returns counterexamples rather than failure probabilities. | Defer to a separate design after compact Monte Carlo and likelihood-accounting gaps are closed. |

## Required Denominators and Fields

A probability-estimation run must emit at least these fields before any estimate is interpreted:

| Field | Meaning |
|---|---|
| `target_distribution_id` | Stable label for the scenario family, parameter ranges, planner row, horizon, and seed policy. |
| `target_distribution_hash` | Hash of resolved config/template/search-space inputs. |
| `estimator_id` | `direct_binomial_mc.v1` for the pilot; later estimators must version their weight semantics. |
| `candidate_draw_count` | Number of candidates drawn from the target/proposal before certification. |
| `invalid_candidate_count` | Count rejected by schema, geometry, route, degeneracy, or duplicate checks. |
| `certified_valid_count` | Count passing pre-planner certification for the conditional target. |
| `simulation_error_count` | Count that failed during reset or execution. |
| `fallback_degraded_count` | Count excluded because planner/runtime availability was fallback or degraded. |
| `eligible_episode_count` | The denominator `n`; executable rows eligible for the probability estimate. |
| `failure_count` | The numerator `k`; rows satisfying the frozen failure predicate. |
| `failure_predicate` | Machine-readable and prose definition of the event. |
| `seeds` | Exact generator and scenario seeds, or a deterministic seed-range expression plus expansion. |
| `ci_method` | Wilson 95% primary; Clopper-Pearson appendix for extreme counts. |
| `assumptions` | Conditional target, independence assumptions, invalid-row handling, and non-operational-risk caveat. |

For future importance-sampling work, add `proposal_distribution_id`, `log_target_density`,
`log_proposal_density`, `likelihood_ratio`, `effective_sample_size`, and weight-truncation policy.
Without those fields, an enriched sampler remains failure discovery only.

## Compact Pilot Plan

Pilot question:

> Under a certified-valid crossing/TTC stress distribution for the `goal` planner, what is the
> collision-event probability within a fixed short horizon, with a 95% confidence interval?

Pre-run scope:

- **Estimator**: `direct_binomial_mc.v1`.
- **Target distribution**: conditional on valid candidates generated from
  `configs/scenarios/templates/crossing_ttc.yaml` and
  `configs/adversarial/crossing_ttc_space.yaml`, with uniform independent draws over the declared
  scalar ranges and `generated_cases_are_benchmark_evidence: false`.
- **Planner row**: `goal` only. Do not mix planner rows in one estimate.
- **Failure predicate**: `outcome.collision_event == true` or metrics-equivalent collision count
  greater than zero within the predeclared horizon.
- **Sample count**: `n = 128` eligible executable episodes. Draw replacement candidates until
  either `eligible_episode_count == 128` or `candidate_draw_count == 192`, whichever comes first.
- **Seeds**: generator seed `329200`; scenario seeds `329200..329327` assigned in draw order for
  eligible episodes. If replacement draws are needed after invalid rows, continue scenario seeds
  monotonically and record the full seed list.
- **Intervals**: Wilson 95% for the primary estimate; Clopper-Pearson 95% when the failure count is
  in the first or last five counts. If zero failures occur in `n = 128`, the report should present
  the point estimate as `0/128` and emphasize the upper bound rather than "no risk."
- **Stop rule**: no adaptive early stopping except fail-closed abort if fallback/degraded rows are
  nonzero, if certification is unavailable in required mode, or if fewer than 96 eligible episodes
  are produced before the 192-candidate cap.
- **Evidence tier**: proposal until run; after execution, at most diagnostic probability estimate
  for this stress distribution, not operational safety, benchmark ranking, or paper-grade evidence.

The pilot intentionally estimates a narrow conditional stress-distribution probability, not the
real-world probability of collision and not the probability over the canonical benchmark suite.
That narrowness is useful: it makes the denominator auditable and lets later work decide whether
importance sampling is worth implementing.

## Reporting Template for a Future Run

A future run report should open in this order:

1. Claim boundary: "diagnostic probability estimate for the declared crossing/TTC stress
   distribution only."
2. Evidence status: diagnostic-only unless a later issue upgrades the contract.
3. Caveats and exclusions: invalid, fallback/degraded, simulation-error, and unavailable rows.
4. Numerator/denominator and interval: `k/n`, Wilson 95% interval, and optional exact interval.
5. Interpretation: what the estimate says about the stress distribution and what it does not say
   about benchmark robustness or real-world safety.

## Follow-Up Boundary

Do not create a follow-up run issue until the method and compute budget are clear. The next issue,
when approved, should name the exact target distribution, `n`, candidate cap, planner row, horizon,
failure predicate, seed list, expected runtime, and artifact location. It should also decide
whether the run is local-only or needs a larger budget.

Later method work should not start with cross-entropy or adaptive stress testing as probability
estimators. The safer sequence is:

1. Run the direct-binomial pilot and inspect denominator health.
2. If the event is too rare for useful direct bounds, add likelihood-accounting fields to
   adversarial manifests.
3. Use cross-entropy only to train/propose an importance distribution.
4. Estimate with held-out weighted importance sampling and effective-sample-size diagnostics.

## Validation

Docs-tier validation for this method note:

```bash
uv run python scripts/tools/run_adversarial_manifest_smoke.py --help
git diff --check
```

No benchmark campaign was run for this note.
