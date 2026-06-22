# Glossary

[Back to Documentation Index](./README.md)

Canonical definitions for acronyms and project-specific terms used across Robot SF. This file is the
single source of truth referenced by the `## Clarity` rule in
[`maintainer_values.md`](./maintainer_values.md#clarity): when a human-facing surface uses one of these
terms, expand it on first use or link here.

If you add a new acronym or domain term to the codebase or docs, add it here in the same change.
When the repository expands the same acronym several ways, treat the expansion below as canonical and
converge on it.

## Domain terms

| Term | Plain-language meaning |
| --- | --- |
| **Social navigation** | A robot moving through space shared with pedestrians, where success means reaching the goal *and* behaving acceptably around people (safe, legible, non-disruptive) — not just avoiding collisions. |
| **Planner** | The decision-making component that chooses the robot's next action (e.g. SocialForce, a trained PPO policy, or a rule-based baseline). Planners are the main thing the benchmark compares. |
| **Social Force / SocialForce** | A pedestrian-motion model where people are driven by virtual "forces" (attraction to their goal, repulsion from others and obstacles). Used here both to simulate pedestrians and as a baseline planner. |
| **VRU** | **Vulnerable Road User** — a non-vehicle participant such as a pedestrian or cyclist whose safety the robot must protect. "Cyclist-like VRU" means a faster, bike-shaped pedestrian variant. |
| **AMV** | **Autonomous Mobility Vehicle** — the autonomous robot/device under test in this simulator. *Note: the repo has also written this as "autonomous micromobility vehicle" and "autonomous mobile vehicle"; this is the canonical expansion — converge on it.* |
| **AMMV** | **Autonomous Micro-Mobility Vehicle** — the micromobility variant of the AMV (smaller, lighter, e.g. scooter/bike-scale). "AMMV/default" comparisons contrast this profile against the default robot profile. |
| **Occluder** | An object that blocks the robot's line of sight, creating sensing gaps. "Occluder timing perturbations" randomly vary *when* an occluder appears so planners are tested against realistic, time-varying visibility. |
| **Naturalistic prior** | An authored plausibility check on generated scenarios. `naturalistic_prior.passed: true` means a generated case looks realistic; `false` flags an intentionally unrealistic / stress-only case. |
| **Actuation variability** | Run-to-run differences in how the vehicle physically executes a commanded action (e.g. steering/braking response), sampled to test robustness to imperfect actuation. |
| **Perturbation family** | A named, bounded way to vary a scenario (timing offsets, occluder placement, etc.), registered so variations stay reproducible and within declared limits. See `robot_sf/scenario_certification/`. |

## Metrics and evaluation

| Term | Plain-language meaning |
| --- | --- |
| **SNQI** | **Social Navigation Quality Index** — a composite score combining several social-navigation metrics (with a per-component breakdown) into one comparable number. Weights are tunable; see [`docs/snqi-weight-tools/`](./snqi-weight-tools/README.md). |
| **Benchmark scenario** | A predeclared, versioned situation the robot is evaluated on, governed so results stay comparable across runs (see [`docs/benchmark_governance.md`](./benchmark_governance.md)). |
| **Claim boundary** | The exact scope of what a result does and does **not** support — stated first in any report so readers know how far to trust it. |
| **Fail-closed** | When a planner/dependency cannot meet the benchmark contract, the run reports an explicit `not available` / `failed` status instead of silently substituting something else and calling it success. |

## Evidence ladder

Robot SF grades every claim by how strong its evidence is. Always label results with one of these
(weakest to strongest). Full definitions live in [`docs/maintainer_values.md`](./maintainer_values.md).

| Tier | Plain-language meaning |
| --- | --- |
| **diagnostic-only** | Used for debugging or probing behavior. Makes no benchmark claim. |
| **smoke evidence** | Narrow proof that the wiring runs end-to-end. Not proof that the result is good. |
| **nominal benchmark evidence** | Results from the predeclared benchmark matrix. A real but not paper-final claim. |
| **paper-grade** | Fully reproducible and strong enough for manuscript-facing claims. |
| **fallback / degraded** | The run completed only by dropping to a weaker mode. Treated as a caveat or exclusion, **never** as success evidence. |

## Run modes

| Mode | Plain-language meaning |
| --- | --- |
| **native** | The planner ran through its own first-class integration. |
| **adapter** | The planner ran through a compatibility shim around a different interface. |
| **fallback** | A required input was missing, so the run dropped to a weaker substitute path (a caveat, not success). |
| **degraded** | The run completed but under reduced fidelity or partial inputs (a caveat, not success). |

---

**Last updated**: 2026-06-22
**Source of truth for**: acronyms and project-specific terms on human-facing surfaces (see the
`## Clarity` rule in [`maintainer_values.md`](./maintainer_values.md#clarity)).
