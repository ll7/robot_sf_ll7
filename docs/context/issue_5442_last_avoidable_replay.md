# Issue #5442 — Frozen-state counterfactual replay: last avoidable action

**Status:** implementation slice landed (controlled-fixture diagnostic evidence).
**Parent:** #5440 · **Depends on (report contract):** #5441 · **Sibling analysis:** #2924.
**Claim boundary:** controlled-fixture diagnostic evidence only. Not a real-episode
root-cause claim, not benchmark or paper-grade evidence, and assigns no legal or
moral fault (`normative_fault` is always `not_assessed`).

## What this slice delivers

A simulator-agnostic **frozen-state counterfactual replay engine** plus a
deterministic controlled fixture that validates it end to end on CPU:

- `robot_sf/benchmark/last_avoidable_replay.py` — the engine. It restores a
  decision-point snapshot, verifies the baseline replay is deterministic, then
  branches over the admissible robot action lattice at every step in
  `[t_danger, t_contact)` to decide whether — and how early — contact was
  avoidable. It reports `t_uca` (earliest avoidable unsafe control action) and
  `t_inevitable` (point of no return).
- `robot_sf/benchmark/last_avoidable_fixtures.py` — a 2D kinematic robot/pedestrian
  fixture implementing the engine's `CounterfactualModel` seam. It holds its own
  `numpy.random.Generator` and snapshots the RNG bit-generator state alongside
  actor state, so replays are bit-for-bit deterministic.
- `robot_sf/benchmark/schemas/last_avoidable_replay.v1.json` — the output contract.
- `scripts/analysis/run_last_avoidable_replay_issue_5442.py` — offline report CLI.
- `tests/benchmark/test_last_avoidable_replay_issue_5442.py` — acceptance tests.

## Determination vocabulary (fail-closed)

| Verdict | Meaning |
| --- | --- |
| `avoidable` | Deterministic baseline, full feasible-action coverage, and at least one admissible action prevents contact within the frozen horizon. `t_uca` and `t_inevitable` are reported. |
| `already_unavoidable` | Deterministic baseline, full feasible-action coverage, but **no** admissible action at any decision point prevents contact. `t_inevitable = t_danger`. |
| `unknown` | Baseline replay is not deterministic **or** feasible-action coverage over the window is incomplete. Per the issue contract this **never** collapses to `unavoidable`. |

`t_uca` is the earliest window step at which an admissible action prevents contact
(the earliest point the robot could have started avoiding). `t_inevitable` is one
past the latest step at which any admissible action still prevents contact (the
point of no return). Blame is placed on the earliest avoidable action, not on the
last command before contact.

## Why the fixture, and not the production simulator (scope decision)

The issue's allowed paths include "the smallest simulator snapshot/restore seam",
and the stop rule says to produce a diagnostic blocker "if snapshot support
requires broad simulator replacement". A code survey (robot_sf 2026-07 main) found
that a faithful mid-episode snapshot/restore of the production simulator **would**
require a broad change:

- Pedestrian goal/zone selection samples the **global** numpy RNG
  (`np.random.choice` / `np.random.uniform` in `robot_sf/ped_npc/ped_zone.py` and
  `ped_population.py`), not a per-object `Generator`. Deterministic branch replay
  would require capturing and restoring global RNG state around every branch.
- `robot_sf/sim/simulator.py` exposes only a reset-to-episode-start path
  (`_reset_social_force_state`), no general snapshot/restore API.
- `PedestrianBehavior` instances and `RouteNavigator` carry mutable state whose
  deep-copy safety is unproven.

Rather than replace the simulator (out of scope, and a determinism risk the issue
itself flags), the engine is decoupled behind the `CounterfactualModel` protocol —
the smallest seam — and validated against a fully deterministic controlled fixture.
A real-simulator adapter can implement the same protocol later; the required RNG
capture is documented in the fixture module.

## Acceptance-criteria mapping

| Acceptance criterion | Where satisfied |
| --- | --- |
| Snapshot/restore includes RNG + actor state | `KinematicCollisionModel.snapshot/restore`; `test_snapshot_includes_rng_and_actor_state`, `test_snapshot_without_rng_diverges` |
| Baseline branching reproduces the fixture within tolerance | `_verify_determinism` (20 replays); determinism check in each avoidable/unavoidable test |
| Action set, feasibility filter, horizon, collision predicate, pedestrian response versioned in output | `ReplayConfig.to_dict` → `config` block; schema `config` required fields |
| `t_inevitable` and `t_uca` computed for preventable late braking, already-unavoidable, two-action interaction | `test_preventable_late_braking_is_avoidable`, `test_already_unavoidable_contact`, `test_two_action_interaction_closed_loop_avoidable` |
| Missing feasible set or nondeterministic baseline → `unknown`, never `unavoidable` | `test_missing_feasible_action_returns_unknown`, `test_nondeterministic_baseline_returns_unknown` |
| Output conforms to a report contract and preserves every branch result | `last_avoidable_replay.v1.json`; `branches` preserved; `test_report_conforms_to_schema_and_records_provenance` |
| Runtime reported, no online gate | `runtime_s` recorded by the CLI/engine |

Note on the report contract: #5441's `collision_causal_report.v1` is not yet
merged. This slice emits a self-contained `last_avoidable_replay.v1` whose field
naming (`t_danger`/`t_uca`/`t_inevitable`/`t_contact` as available/unavailable,
`normative_fault: not_assessed`) is forward-compatible, so it can be embedded as
the counterfactual branch of that contract once #5441 lands — without re-running.

## Competing explanations carried from the issue

- Interactive pedestrian dynamics may make snapshot replay nondeterministic — the
  engine tests for this and abstains to `unknown` rather than guessing.
- The action lattice may omit the true avoidance action — a missing/empty feasible
  set drives `unknown`, never `unavoidable`.
- Emergency braking may prevent geometric contact while causing a different
  social-navigation failure — the fixture's collision predicate is geometric only;
  broader social-failure attribution is out of scope for this slice.
- The earliest divergent action may be a consequence of an earlier prediction or
  guard defect — this slice localizes the avoidable action; upstream defect
  attribution belongs to the #5441 causal-report join.

## Validation

```bash
uv run pytest -q tests/benchmark -k 'counterfactual or snapshot or avoidable'   # 38 passed
uv run python scripts/analysis/run_last_avoidable_replay_issue_5442.py          # 5 controlled fixtures
git diff --check
```

Observed CLI determinations: `preventable_late_braking` → avoidable (t_uca=0,
t_inevitable=7); `two_action_interaction` → avoidable (t_uca=0, t_inevitable=8);
`already_unavoidable` → already_unavoidable (t_inevitable=0); `nondeterministic_baseline`
→ unknown (nondeterministic_baseline); `missing_feasible_action` → unknown
(incomplete_feasible_action_coverage).

## Out of scope / remaining

- Real-simulator `CounterfactualModel` adapter (needs the global-RNG capture seam
  above; gated behind the determinism budget in the issue stop rule).
- Join into `collision_causal_report.v1` once #5441 merges.
- No benchmark campaign run, no Slurm/GPU submission, no metric/release semantics
  change, no paper/dissertation claim edits.
