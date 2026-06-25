# Issue #3573 — Reactive-vs-replay reactivity-ablation quantifier (increment)

**Status:** diagnostic / analysis. **Evidence grade:** idea-level quantification layer.

## What this is

`robot_sf/benchmark/reactivity_ablation.py` is the pure **quantification layer** for #3573. A
structural validity question for any pedestrian-interaction benchmark is whether the pedestrians
react to the robot. Non-reactive replay lets a planner intrude and still accrue good downstream
metrics (inflating apparent performance); fully reactive pedestrians may over-yield. This module
turns a paired reactive-vs-replay ablation (run over identical scenarios + seeds with common random
numbers) into the issue's deliverable.

It mirrors the accepted decision-layer pattern in `failure_cause.py` (#3484) and siblings.

## Quantifier (`reactivity_ablation.v1`)

- `reactivity_delta(contrast)` → per-metric `reactive − replay` deltas and a `replay_flatters` flag
  (replay shows fewer collisions/near-misses or more separation).
- `assess_reactivity_ablation(contrasts)` → per-planner deltas, the mean replay collision/near-miss
  inflation, the planners flattered by replay, and the **rank-reactivity-sensitive** planners
  (collision rank changes between conditions).

## Scope boundary

Pure and side-effect free. Running the paired reactive-vs-replay ablation (and the open-loop replay
pedestrian mode) needs benchmark runs and is the deliberate deferred follow-up.

## Tests

`tests/benchmark/test_reactivity_ablation.py` (7 tests): replay-flattering detection across metrics,
mean-inflation aggregation, rank-reactivity sensitivity (flip vs stable), and empty-input rejection.
