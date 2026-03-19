# Planner Zoo Context

This document gives agents the current integration and interpretation context for the repository's
planner-zoo work.

## Purpose

The planner-zoo effort is not a generic collection exercise. The repository wants planner entries
that are:

- realistically runnable in the current benchmark stack,
- attributable to original upstream code or checkpoints,
- compatible with explicit observation and action adapters,
- conservative enough for paper-facing interpretation.

## Current Planner Readiness Frame

Use `docs/benchmark_planner_family_coverage.md` as the source of truth for readiness categories and
benchmark-facing terminology.

Implemented and benchmarkable families today:

- `goal`
- `social_force`
- `orca`

Implemented but experimental families today:

- `ppo`

Conceptually adjacent only or testing-only entries:

- `guarded_ppo`
- `prediction_planner`
- `predictive_mppi`
- `risk_dwa`
- `mppi_social`
- `hybrid_portfolio`
- `stream_gap`
- `gap_prediction`
- legacy/model-sensitive adapters such as `sacadrl`, `socnav_sampling`, `socnav_bench`

## What Makes A Planner Integration Credible

Before claiming a planner family is supported, check:

1. upstream repo/license/checkpoint provenance is explicit,
2. the source harness or model-only path can actually run,
3. the observation adapter is thin and auditable,
4. the action/kinematics conversion is explicit,
5. benchmark output labels the planner as baseline-ready or experimental correctly.

If the local wrapper becomes the main implementation burden, the result is no longer a clean
original-code-backed benchmark entry.

## Current Research Anchors

Use these existing docs instead of restarting planner-family research from zero:

- `docs/context/issue_629_planner_zoo_research_prompt.md`
- `docs/context/issue_601_crowdnav_feasibility_note.md`
- `docs/context/issue_626_sonic_source_harness_probe.md`
- `docs/benchmark_planner_quality_audit.md`

## Questions Agents Should Answer Explicitly

When exploring or integrating a planner, answer:

- What exact observation contract does it need?
- Does it assume holonomic motion, waypoint actions, or unicycle commands?
- Is a source-harness validation run possible before wrapping?
- What benchmark claim can be made after integration?
- What would still block paper-facing use?

## Conservative Recommendation Pattern

Prefer these recommendation classes:

- integrate next
- prototype only
- assessment only
- inspiration only
- reject

That vocabulary already matches the planner-zoo research prompt and keeps scope disciplined.
