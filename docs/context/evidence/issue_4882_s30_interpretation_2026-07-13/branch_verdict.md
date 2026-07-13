<!-- AI-GENERATED (robot_sf#4882, 2026-07-13) — NEEDS-REVIEW -->
# S30 Hybrid-vs-ORCA Branch Verdict

**Tested pair:** leading hybrid `hybrid_rule_v3_fast_progress_static_escape_continuous` vs `orca`.
**Primary metric:** episode success rate. **Effect direction:** hybrid > ORCA (all point estimates).
**Budget:** S30, seeds 111–140, 48 scenarios, n = 1440 per arm.

## Pre-registration wording (verbatim)

> whether the targeted hybrid-vs-ORCA success lead survives the predeclared 30-seed schedule on the
> h600 surface

The pre-registration (`docs/context/issue_4365_h600_hybrid_vs_orca_s30_preregistration.md`) encodes
**no** bootstrap resampling unit and **no** CI-overlap-vs-paired-delta decision rule. Per the analysis
guardrail, the rule is treated as ambiguous, the wording is quoted verbatim, and the **most
conservative reading** drives the headline branch.

## Headline: `branch_b_boundary` (conservative reading)

Under the CI method the task/pre-registration specifies — a **scenario-clustered hierarchical
bootstrap** (resample 48 scenarios, then 30 seeds within each) — and the rule as literally worded
("the leading hybrid arm's CI separates from ORCA's on the primary metric"):

- Leading hybrid success 95% CI **`[0.688, 0.849]`** overlaps ORCA **`[0.591, 0.767]`** (overlap band
  0.688–0.767). The per-arm-CI separation rule is **not met** → boundary.

The per-arm CIs are wide because the scenario-clustered bootstrap treats the 48 scenarios as the
outer resampling unit and between-scenario heterogeneity is large; this is the intended, conservative
behaviour of the mandated hierarchical scheme (the task explicitly forbids naive row bootstrap).

## Alternative: `branch_a_separation` (paired and/or seed-block readings)

- **Paired delta (scenario-clustered):** success `+0.091 [+0.024, +0.162]`, collision
  `−0.135 [−0.217, −0.055]` — both exclude 0. The paired design cancels shared-scenario variance, so
  the delta is separated even though the marginal per-arm CIs overlap.
- **Seed-block (fixed-suite) bootstrap:** per-arm success CIs **do not** overlap
  (`[0.747, 0.794]` vs `[0.657, 0.703]`); paired success `+0.091 [+0.065, +0.117]`, collision
  `−0.135 [−0.159, −0.110]`. This reproduces the maintainer closeout / #5514 intervals exactly, and
  under it **all four hybrids** separate from ORCA on both success and collision.

## The maintainer decision fork

The verdict is **method-dependent** on one unspecified choice — the **inference target**:

- Treat the 48 curated scenarios as a **fixed benchmark suite** (quantify seed/stochastic uncertainty
  only) → **seed-block** → `branch_a_separation`. Appropriate if these scenarios *are* the benchmark.
- Treat them as a **sample from a scenario population** (generalize to unseen scenarios) →
  **scenario-clustered** → per-arm CIs overlap → `branch_b_boundary` under the literal rule (though the
  paired delta still excludes 0).

Recommendation for the maintainer's branch decision: ratify which inference target is canonical for
the fixed S30 evaluation suite, and whether the separation rule reads on per-arm CIs or the paired
delta. If the fixed-suite seed-block target is canonical (the standard choice for a curated benchmark),
the verdict is `branch_a_separation`; the conservative headline here does not contradict that, it
bounds it.

## Claim boundary

Diagnostic-only. This packet computes the pre-registered rule's output under explicit, stated methods;
it does not promote a paper, dissertation, leaderboard, record-breaking, or universal-planner claim,
and it does not itself make the maintainer's branch/promotion decision.
