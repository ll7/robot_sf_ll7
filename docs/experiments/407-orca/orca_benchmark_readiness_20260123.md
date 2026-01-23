# ORCA Benchmark Readiness (2026-01-23)

## Summary judgment
- The current ORCA implementation is **good enough as a classical baseline** for a benchmark run.
- It is a **valid comparison to PPO** only when the comparison is strictly apples-to-apples (same robot model, limits, scenarios, seeds, and termination rules).

## Why it is "good enough" now
- Stable and reproducible: 102/129 successes, 27 collisions, 0 terminations on the latest run.
- Failures are localized and interpretable: narrow passages and crowd compression, which are expected weaknesses for reactive classical planners.
- Provides a meaningful lower-bound / classical baseline against a learned PPO policy.

## What makes it a valid PPO comparison
- Same robot model (bicycle vs diff-drive) and same speed/acceleration limits.
- Same scenario configs and seeds.
- Same observation scope (ORCA uses occupancy grid for static obstacles; PPO also uses the grid).
- Same termination/collision criteria and max steps.

## Key caveats to document
- ORCA is fundamentally **holonomic**; our implementation applies **non-holonomic constraints**, so the comparison is to a constrained ORCA baseline, not the idealized paper setting.
- The persistent failures in narrow layouts are likely geometry/turning-radius driven rather than algorithm instability.

## Suggested benchmark phrasing
“ORCA-style reactive baseline (non-holonomic constrained), grid-based static obstacle awareness.”
