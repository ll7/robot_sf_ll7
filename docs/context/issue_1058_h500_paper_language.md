# Issue 1058 H500 Paper Language

Date: 2026-05-07

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1058>

## Goal

Provide reusable paper/report language for h500 as a long-horizon sensitivity surface without
presenting it as a replacement for the fixed-horizon camera-ready benchmark or a single winner
table.

## Evidence Inputs

* [Issue #1044 H500 Follow-Up Benchmark Plan](issue_1044_h500_followup_benchmark_plan.md)
* [Issue #1045 H500 Solvability Mechanisms](issue_1045_h500_solvability_mechanisms.md)
* [Issue #1049 H500 Mechanism Pilot](issue_1049_h500_mechanism_pilot.md)
* [Issue #1056 H500 Failure Classification](issue_1056_h500_failure_classification.md)
* [Issue #1055 Exposure-Aware H500 Tables](issue_1055_exposure_aware_h500_tables.md)

## Reusable Language

### Short Version

> We report h500 as a long-horizon sensitivity analysis rather than as a replacement for the fixed
> camera-ready benchmark. Fixed horizons test strict-time-budget navigation; h500 asks what happens
> when planners are allowed to remain in the interaction long enough to finish. The longer horizon
> separates clean time-budget artifacts from persistent planner failures, but it also exposes
> additional safety and comfort costs.

### Results Paragraph

> In representative traces, h500 produces three qualitatively different outcomes. Some fixed-horizon
> timeouts become clean completions, supporting a time-budget-relief interpretation. Other h500
> completions require longer exposure to pedestrians and show higher force/comfort pressure, so
> success should be read together with exposure-normalized safety metrics. A third class of longer
> runs reveals collisions that fixed horizons would have reported only as unfinished episodes. For
> this reason, h500 results are reported as multi-table sensitivity evidence, not as a single
> aggregate winner ranking.

### SNQI Caveat

> Camera-ready SNQI v3 remains calibrated for the fixed-horizon paper benchmark. H500 SNQI values,
> if reported, should be labeled as sensitivity values unless a separately versioned h500 SNQI
> contract is calibrated. The preferred h500 presentation is therefore the underlying success,
> collision, near-miss, comfort, exposure, and duration terms.

### Trace-Backed Mechanism Caveat

> Mechanism labels are trace-backed only when retained per-step traces or videos show the behavior.
> Aggregate fixed-to-h500 deltas can identify candidate cells for inspection, but they do not prove
> waiting/yielding, recovery, replanning, or risk-taking mechanisms by themselves.

## Forbidden Or Unsafe Framing

Avoid these claims:

* "H500 replaces the fixed-horizon camera-ready benchmark."
* "H500 shows planner safety improves because success improves."
* "H500 identifies a single best planner."
* "H500 successes mostly come from waiting until pedestrians pass."
* "Near-miss changes are only a runtime artifact."
* "SNQI v3 is calibrated for h500."

Safer alternatives:

* "H500 is a long-horizon sensitivity/report surface."
* "Success gains must be interpreted with collision, near-miss, comfort, exposure, and duration."
* "Waiting/yielding remains a hypothesis unless trace/video evidence shows it."
* "Current h500 SNQI is a sensitivity value unless a separate h500 contract is calibrated."

## Table Requirements

Any paper-facing h500 table should include, or be accompanied by, these columns:

* fixed and h500 success,
* fixed and h500 collision,
* episode duration or steps,
* near misses per episode and per successful episode,
* per-step or per-second near-miss rates when raw traces exist,
* force/comfort exposure when traces expose those fields,
* planner execution mode, with fallback/degraded rows excluded or explicitly caveated.

Do not publish a standalone h500 winner table without these caveats.

## Validation

Reviewed against the h500 plan, aggregate mechanism analysis, trace pilot, classification note, and
exposure-aware table note. The required search surface is:

```bash
rtk rg -n "h500.*(winner|replace|waiting|artifact|calibrated)" docs/context docs/research_reporting.md -g '!docs/context/issue_1058_h500_paper_language.md'
```
