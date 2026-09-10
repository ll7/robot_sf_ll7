# Issue #5442 — Frozen-state counterfactual replay: last avoidable action

**Status:** implementation slices landed (controlled-fixture and diagnostic
production-adapter evidence).
**Parent:** #5440 · **Depends on (report contract):** #5441 · **Sibling analysis:** #2924.
**Claim boundary:** controlled-fixture and diagnostic production-adapter evidence
only. Not a real-episode root-cause claim, not benchmark or paper-grade evidence,
and assigns no legal or moral fault (`normative_fault` is always `not_assessed`).

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

### Successor slice: production-simulator `CounterfactualModel` adapter (#5442, cheap-lane)

This slice adds the **production-simulator adapter** the issue thread named as
remaining (the global-RNG snapshot seam was the gated dependency):

- `robot_sf/benchmark/simulator_counterfactual_adapter.py` — `SimulatorCounterfactualModel`,
  a `CounterfactualModel` over the live `Simulator`. It captures the pedestrian PySF
  state buffer, per-pedestrian behavior runtimes (single-pedestrian waypoint/hold
  state and route-group navigator waypoint index), robot pose/velocity and route
  navigator progress, mutable group membership, per-behavior generators, and the
  stateful residual controller, plus the **global numpy RNG** via
  `numpy.random.get_state`/`set_state` (deep-copied so
  repeated restores stay independent). The `capture_rng` flag documents the seam:
  omitting it lets a mid-episode pedestrian respawn (`sample_zone` draws from the
  global RNG) diverge a replay by meters, which the engine's fail-closed `unknown`
  guard would catch. The default action lattice follows the native drivetrain
  control semantics and returns an empty set for unknown action contracts. Its
  `source_kind`, action-set ID (including native limits), and feasible-action
  filter are bound by the replay engine; caller-declared mismatches abstain to
  `unknown`.
- `tests/benchmark/test_simulator_counterfactual_adapter_issue_5442.py` — headless
  (no-display) `Simulator` construction, snapshot/restore determinism, the RNG-capture
  seam, and a full `locate_last_avoidable` run on a genuine production fixture
  (`classic_doorway.svg`, density 0.06, route seed 21, global seed 25) where the
  forward-acceleration baseline contacts after 39 applied ticks and native braking
  avoids contact (`avoidable`, deterministic baseline, full coverage of the declared finite
  action lattice). The underlying production command space remains continuous, so this is not
  exhaustive continuous-action coverage.

Scope correction versus the earlier doc's "broad simulator replacement" note: a code
re-survey on current `main` found pedestrian goal/zone resampling now draws from the
**global** numpy RNG (`sample_zone` / `ped_population` group respawn), not from the
broad per-object generators the earlier note assumed. The production adapter uses a
narrow snapshot seam for the global RNG, actor/behavior state, public grouping, and
the backend group list; it does not replace the simulator. The engine's determinism
check is the safeguard: if a replay diverges, it abstains to `unknown` rather than
guessing. Each baseline replay also records typed `NoOpStep` state receipts and
compares them with `compare_continuation_traces`; equal collision/contact ticks
alone are not determinism evidence.

## Determination vocabulary (fail-closed)

| Verdict | Meaning |
| --- | --- |
| `avoidable` | Deterministic baseline, full coverage of the declared action set, and at least one admissible action prevents contact within the frozen horizon. `t_uca` and `t_inevitable` are reported. |
| `already_unavoidable` | Deterministic baseline, full coverage of the declared action set, but **no** admissible action at any decision point prevents contact. `t_inevitable = t_danger`. |
| `unknown` | Baseline replay is not deterministic **or** feasible-action coverage over the window is incomplete. Per the issue contract this **never** collapses to `unavoidable`. |

`t_contact` and `observed_contact_steps` use the same state-tick convention: the
contact-producing action at index ``t_contact - 1`` yields contact state tick
``t_contact``. `t_uca` is the earliest window step at which an admissible action prevents contact
(the earliest point the robot could have started avoiding). `t_inevitable` is one
past the latest step at which any admissible action still prevents contact (the
point of no return). Blame is placed on the earliest avoidable action, not on the
last command before contact.

## Historical scope decision: controlled fixture before the production adapter

Before the production adapter was delivered, the issue's allowed paths included
"the smallest simulator snapshot/restore seam", and the stop rule said to produce
a diagnostic blocker "if snapshot support requires broad simulator replacement".
A code survey (robot_sf 2026-07 main) recorded that a *general* faithful
mid-episode snapshot/restore of the production simulator would require broader
work:

- Pedestrian goal/zone selection samples the **global** numpy RNG
  (`np.random.choice` / `np.random.uniform` in `robot_sf/ped_npc/ped_zone.py` and
  `ped_population.py`), not a per-object `Generator`. Deterministic branch replay
  would require capturing and restoring global RNG state around every branch.
- `robot_sf/sim/simulator.py` exposes only a reset-to-episode-start path
  (`_reset_social_force_state`), no general snapshot/restore API.
- `PedestrianBehavior` instances and `RouteNavigator` carry mutable state whose
  deep-copy safety is unproven.

That historical rationale led the engine to be decoupled behind the
`CounterfactualModel` protocol — the smallest seam — and initially validated
against a fully deterministic controlled fixture. It is superseded as a reason to
omit production coverage: PR #8612 implements a narrow diagnostic adapter over the
live simulator, including the required RNG and backend-group synchronization. The
adapter does not replace the simulator or turn replay output into a real-episode
causal claim; divergence still abstains to `unknown`.

## Acceptance-criteria mapping

| Acceptance criterion | Where satisfied |
| --- | --- |
| Snapshot/restore includes RNG + actor state | `KinematicCollisionModel.snapshot/restore`; `test_snapshot_includes_rng_and_actor_state`, `test_snapshot_without_rng_diverges` |
| Baseline branching reproduces the fixture within tolerance | `_verify_determinism` (20 replays) plus typed no-op trace comparison; determinism check in each avoidable/unavoidable test |
| Action set, declared action-set coverage, horizon, collision predicate, pedestrian response versioned in output | `ReplayConfig.to_dict` → `config` block; schema `config` required fields |
| `t_inevitable` and `t_uca` computed for preventable late braking, already-unavoidable, two-action interaction | `test_preventable_late_braking_is_avoidable`, `test_already_unavoidable_contact`, `test_two_action_interaction_closed_loop_avoidable` |
| Missing feasible set or nondeterministic baseline → `unknown`, never `unavoidable` | `test_missing_feasible_action_returns_unknown`, `test_nondeterministic_baseline_returns_unknown` |
| Output conforms to a report contract and preserves every branch result | `last_avoidable_replay.v1.json`; `branches` preserved; `test_report_conforms_to_schema_and_records_provenance` |
| Runtime reported, no online gate | `runtime_s` recorded by the CLI/engine |

Note on the report contract: #5441's `collision_causal_report.v1` is now merged,
and the join described below embeds the replay result into that contract.

## Join into `collision_causal_report.v1` (remaining item delivered)

`robot_sf/benchmark/collision/collision_causal_report.py` exposes
`collide_causal_report_from_last_avoidable`, which wraps a `last_avoidable_replay.v1`
result into the additive `collision_causal_report.v1` contract **without re-running**
the engine. The fail-closed failure semantics survive the join:

- `avoidable` replay → non-abstaining report, `verdict: avoidable`,
  `supported_actual_cause: true`, the replay's `t_uca`/`t_inevitable` carried through
  as available timestamps, and each minimal-sufficient preventing intervention recorded.
  Mechanism metadata remains caller-supplied evidence; this join does not infer it
  from replay coverage.
- `already_unavoidable` replay → non-abstaining report, `verdict: unavoidable`,
  `supported_actual_cause: false` (no planner action is the cause when contact was
  already unavoidable at `t_danger`), with mechanism and confidence fields set to
  explicit unknown values.
- `unknown` replay (nondeterministic baseline or incomplete feasible-action
  coverage), or an unsupported replay verdict → **fully abstaining** report,
  `verdict: unknown`, every planner-internal reconstruction element and unavailable
  timestamp declared in `missing_fields`. It **never** becomes `unavoidable`.

The replay summary has no per-element canonical trace, so planner-internal
reconstruction elements are unavailable for every joined verdict; only the
replay-derived critical timestamps can be marked available.

Native live-simulator replays are additionally rejected by the causal join until
the adapter carries a verified episode/map/seed/software provenance receipt. They
remain valid diagnostic replay evidence, but the join emits an explicit
`native_simulator_causal_join_unsupported` abstention rather than relabelling the
result as `synthetic_fixture`.

Replays whose `source_kind` remains `unspecified` are also rejected by the causal
join with `unspecified_replay_provenance`; controlled fixtures must declare
`synthetic_fixture` explicitly. This preserves diagnostic compatibility while
preventing provenance-free output from entering the causal report.

An explicit `synthetic_fixture` label is necessary but not sufficient: the causal
join also requires non-native replay provenance for `action_set_id`,
`feasibility_filter`, `collision_predicate`, and a known `pedestrian_response`
(`replayed` or `closed_loop`). Missing, `unspecified`, `unknown`, or malformed
values produce `incomplete_replay_provenance` and a fully abstaining report.
The replay engine may still emit its schema-safe omitted-response/unspecified
configuration for diagnostic inspection; that legacy behavior does not authorize a
causal join.

`normative_fault` is always `not_assessed`. The join is exercised by
`tests/benchmark/test_collision_causal_report_join_5442.py`.

## Competing explanations carried from the issue

- Interactive pedestrian dynamics may make snapshot replay nondeterministic — the
  engine tests for this and abstains to `unknown` rather than guessing.
- The action lattice may omit the true avoidance action — a missing/empty feasible
  set drives `unknown`, never `unavoidable`.
- Emergency braking may prevent geometric contact while causing a different
  social-navigation failure — the fixture's collision predicate is geometric only;
  broader social-failure attribution is out of scope for this slice.
- The earliest divergent action may be a consequence of an earlier prediction or
  guard defect — this slice identifies an intervention-supported avoidability result,
  but does not localize an upstream planner defect without a canonical trace.

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

- The `collision_causal_report.v1` join is delivered by PR #5713; broader
  planner-specific action lattices remain out of scope — subclasses can supply a
  planner action set.
- No benchmark campaign run, no Slurm/GPU submission, no metric/release semantics
  change, no paper/dissertation claim edits.
