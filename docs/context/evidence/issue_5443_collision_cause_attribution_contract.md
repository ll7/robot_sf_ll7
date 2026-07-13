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
- `tests/benchmark/test_collision_causal_attribution_issue_5443.py` — 18 tests, selectable by the
  issue's `pytest -k 'causal and attribution'` command.

## Acceptance-Criteria Mapping

- [x] Ground-truth fixture manifest freezes cause class, activation window, allowed intervention, and
  ambiguity status before analysis — `GroundTruthFixture` + manifest.
- [x] Report metrics defined and computed (class P/R, top-explanation accuracy, temporal localization,
  avoidability accuracy, abstention coverage, calibration) — `score_attribution` / `AttributionReport`.
- [ ] Simple injected causes reach ≥0.90 top-explanation accuracy, median temporal error ≤1 step —
  **blocked on the analyser (#5441/#5442)**; the stop rule is encoded and enforced by the harness.
- [ ] Ambiguous fixtures never receive a high-confidence single-cause verdict — guard encoded and
  tested; measurement blocked on the analyser.
- [ ] Negative controls not promoted to actual cause — guard encoded and tested; measurement blocked.
- [ ] Two reviewers independently inspect a frozen sample — process step, remaining.
- [x] Compact evidence schema-validated and bound to commit hashes — this note + schema/enum sync test.

## Validation

- `pytest -q tests/benchmark -k 'causal and attribution'` → 18 passed.
- `python scripts/analysis/validate_collision_cause_attribution_issue_5443.py` →
  `status: analyser_unavailable`, `covered_matrix: true`, `n_fixtures: 14` (fail-closed as expected).
- `git diff --check` → clean. `ruff check` / `ruff format --check` → clean.

## Remaining Work

The accuracy measurement (criteria 3–5) and the two-reviewer step (criterion 6) require the analyser
from #5441/#5442. When it can emit verdicts, feed them to the harness via `--verdicts`; the report
moves from `analyser_unavailable` to `scored` and applies the `revise`/`pass` stop rule automatically.
