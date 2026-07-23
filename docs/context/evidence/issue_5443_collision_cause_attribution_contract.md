<!-- AI-GENERATED: validation-contract evidence; NEEDS-REVIEW: maintainer verification before reuse. -->
# Issue #5443 Collision-Cause Attribution Validation Contract

Date: 2026-07-13
Issue: <https://github.com/ll7/robot_sf_ll7/issues/5443>
Parent: #5440 · Depends on: #5441, #5442

## Claim Boundary

This is a **validation-side contract slice** for "benchmark collision-cause attribution on injected
and ambiguous faults". It delivers the frozen ground-truth fixture manifest, the pure attribution
scoring/report machinery, and a fail-closed validation harness. It runs no simulation, trains no
classifier, and makes **no benchmark or paper-grade claim**. The accuracy RUN against a real analyser
is deliberately **not** performed here because the analyser under test (cause-report contract #5441,
last-avoidable-action counterfactual replay #5442) is not yet implemented — the run is blocked exactly
as a campaign RUN would be.

Evidence grade: `diagnostic-only` / contract. No metric semantics change.

## What This Slice Delivers

- `robot_sf/benchmark/collision_cause_attribution.py` — frozen `GroundTruthFixture` contract,
  manifest coverage check (`validate_fixture_manifest`), pure scoring (`score_attribution`) for class
  precision/recall, top-explanation accuracy, temporal-localization error, avoidability accuracy,
  abstention coverage, and confidence calibration, plus a fail-closed `build_validation_report`.
- `robot_sf/benchmark/schemas/collision_cause_attribution_fixture.v1.json` — fixture record schema;
  the `cause_class` enum is kept in sync with the module via a test.
- `tests/benchmark/fixtures/collision_cause_attribution_manifest_5443.json` — the frozen manifest
  covering the predeclared validation matrix: one activation per single cause (observation
  omission/delay, prediction miss, candidate omission, bad selection, guard omission, infeasible
  applied command, route trap, already-unavoidable contact, metric artifact), two interacting/ambiguous
  fixtures, and two negative controls.
- `scripts/analysis/validate_collision_cause_attribution_issue_5443.py` — deterministic report CLI;
  emits `analyser_unavailable` when no verdicts are supplied.
- `robot_sf/benchmark/collision_cause_analyser.py` — the deterministic **rule-based cause
  analyser under test**. It translates last-avoidable counterfactual replay results plus
  observable fault evidence into `AttributionVerdict` objects conforming to the 10-class cause
  schema. It never reads the manifest's ground-truth `cause_class`: it attributes from computed
  replay verdicts, computed counterfactual decisiveness (does repairing a fault remove contact?),
  and computed reported/physical-collision mismatches.
- `robot_sf/benchmark/last_avoidable_fixtures.py` — the 14 deterministic fault-injection fixture
  builders (`CollisionCauseFixture` / `InjectedFault`) over the existing kinematic pattern, keyed
  to the frozen manifest by `fixture_id` without carrying the answer key.
- `scripts/analysis/run_collision_cause_attribution_issue_5443.py` — runner that drives
  fixtures -> last-avoidable replay -> rule-based analyser -> verdicts, and optionally scores them
  against the frozen manifest.
- `tests/benchmark/test_collision_causal_attribution_issue_5443.py` — 18 tests, selectable by the
  issue's `pytest -k 'causal and attribution'` command.
- `tests/benchmark/test_collision_cause_analyser_issue_5443.py` — fixture determinism,
  decisive-pattern, and end-to-end attribution tests proving criteria 3-5 on the 14 injected
  fixtures (29 tests).

## Acceptance-Criteria Mapping

- [x] Ground-truth fixture manifest freezes cause class, activation window, allowed intervention, and
  ambiguity status before analysis — `GroundTruthFixture` + manifest.
- [x] Report metrics defined and computed (class P/R, top-explanation accuracy, temporal localization,
  avoidability accuracy, abstention coverage, calibration) — `score_attribution` / `AttributionReport`.
- [ ] Simple injected causes reach ≥0.90 top-explanation accuracy, median temporal error ≤1 step —
  **PASS on injected fixtures**: the rule-based analyser scores top-explanation accuracy 1.0 and
  median temporal-localization error 0.0 steps over the ten unambiguous fixtures. Injected-fixture
  validation only; a held-out real-trace study remains required for paper-facing claims.
- [x] Ambiguous fixtures never receive a high-confidence single-cause verdict — guard encoded and
  tested; **measured PASS** (both ambiguous fixtures abstain at confidence 0.3, below the 0.8
  high-confidence threshold).
- [x] Negative controls not promoted to actual cause — guard encoded and tested; **measured PASS**
  (both negative controls abstain to `none` below the high-confidence threshold).
- [ ] Two reviewers independently inspect a frozen sample — process step, remaining.
- [x] Compact evidence schema-validated and bound to commit hashes — this note + schema/enum sync test.

## Validation

- `pytest -q tests/benchmark -k 'causal and attribution'` → 47 passed (18 contract + 29 analyser
  tests).
- `python scripts/analysis/validate_collision_cause_attribution_issue_5443.py` (no verdicts) →
  `status: analyser_unavailable`, `covered_matrix: true`, `n_fixtures: 14` (fail-closed).
- `uv run python scripts/analysis/run_collision_cause_attribution_issue_5443.py --score` →
  `status: scored`, `verdict: pass`, `top_explanation_accuracy: 1.0`,
  `median_temporal_localization_error: 0.0`, `abstention_coverage: 1.0`, no ambiguity or
  negative-control violations.
- `git diff --check` → clean. `ruff check` / `ruff format --check` → clean.

Claim boundary: this is **controlled-fixture injected-fault validation only** (diagnostic tier).
The kinematic fixtures reuse one avoidable collision geometry and a small set of repair transforms;
they validate the analyser's rule logic and honest abstention, not planner realism. Per the issue stop
rule, a later held-out real-trace study is required before any paper-facing attribution claim.

## Remaining Work

- Two-reviewer independent inspection of a frozen sample (criterion 6) — process step, remaining.
- A held-out real-trace study for paper-facing attribution claims — out of scope for this slice.
- The `already_unavoidable_contact` fixture localizes activation to the replay `t_inevitable` (step
  0 for the kinematic primitive) rather than the manifest window `[18, 18]`; the median temporal
  error stays at 0.0, but a later long-approach unavoidable geometry could tighten this.
- The eight avoidable single-cause fixtures share one collision geometry with distinct fault
  signatures; richer per-cause kinematics would strengthen realism but are not needed for the
  rule-logic validation this slice delivers.
