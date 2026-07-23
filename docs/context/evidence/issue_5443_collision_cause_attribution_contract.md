<!-- AI-GENERATED: controlled-fixture validation evidence; independent review still required. -->
# Issue #5443 Collision-Cause Attribution Validation

Date: 2026-07-23
Issue: <https://github.com/ll7/robot_sf_ll7/issues/5443>
Parent: #5440 · Depends on: #5441, #5442

## Claim Boundary

This is controlled-fixture injected-fault validation only. It validates a
deterministic rule chain against a frozen answer key. It is not a held-out
real-trace study, campaign result, production root-cause claim, or assignment of
legal or moral fault.

## Separation Between Analyser and Scorer

- `tests/benchmark/fixtures/collision_cause_attribution_manifest_5443.json`
  is the scorer-only answer key. It freezes cause classes, activation windows,
  allowed interventions, ambiguity, and avoidability.
- `robot_sf/benchmark/last_avoidable_fixtures.py` supplies low-level
  `ObservableTraceEvent` records with expected and observed channel values. The
  analyser input contains neither `cause_class` nor an answer-key activation
  window.
- `robot_sf/benchmark/collision_cause_analyser.py` maps observable patterns to
  cause classes, derives onset from the first matching event, and verifies
  counterfactual decisiveness by replaying a mechanism-specific repair.
- The eight avoidable single-cause events occur before the shared baseline
  contact at step 23. Their repairs use distinct state transformations rather
  than one shared answer-key-bearing repair.
- Metric artifacts require a reported contact and no physical contact over the
  complete 120-step horizon. An earlier inflated-radius report followed by
  physical contact is explicitly rejected as an artifact.

## Evidence and Provenance Contract

`scripts/analysis/run_collision_cause_attribution_issue_5443.py` generates
`collision_cause_analyser_run.v1` and validates every payload against
`robot_sf/benchmark/schemas/collision_cause_analyser_run.v1.json` before
writing or scoring it.

Every payload records:

- the exact 40-character Git commit and whether the tracked tree was dirty;
- the frozen manifest path and SHA-256;
- the complete analyser rule configuration and its canonical SHA-256;
- the committed payload-schema SHA-256;
- per-fixture observable events, derived cause/onset, replay evidence, and
  counterfactual decisiveness.

A run with `git_tree_dirty: true` is development evidence only. Reusable evidence
must be generated from the final clean PR head; the exact head and hashes are
therefore carried by the payload rather than copied manually into this note.

## Acceptance-Criteria Mapping

- [x] Frozen cause/window/intervention/ambiguity manifest.
- [x] Precision/recall, top-explanation accuracy, temporal localization,
  avoidability, abstention, and calibration metrics.
- [x] Controlled simple fixtures: top-explanation accuracy 1.0 and median
  temporal-localization error 0.0 steps.
- [x] Ambiguous fixtures abstain below the high-confidence threshold.
- [x] Negative controls are not promoted.
- [ ] Two independent reviewers inspect the frozen sample.
- [x] Run payload is schema-validated and bound to exact commit, analyser
  configuration, manifest, and schema hashes.

## Validation

- `pytest -q tests/benchmark/test_collision_cause_analyser_issue_5443.py`
  → 33 passed. It exercises label blinding, scorer-key permutation,
  fixture-id permutation,
  counterfactual repairs, ambiguity, negative controls, whole-horizon metric
  handling, schema rejection, and provenance hashes.
- `pytest -q tests/benchmark -k 'causal and attribution'` covers the frozen
  scoring contract plus the analyser integration → 51 passed.
- `python scripts/analysis/validate_collision_cause_attribution_issue_5443.py`
  without analyser verdicts remains fail-closed with
  `status: analyser_unavailable`.
- `python scripts/analysis/run_collision_cause_attribution_issue_5443.py
  --score` produces a schema-validated `status: scored`, `verdict: pass` report.

## Remaining Work

- Independent two-reviewer inspection remains required by criterion 6.
- A held-out real-trace study remains required before any paper-facing or
  production attribution claim.
- `already_unavoidable_contact` localizes to replay `t_inevitable` (step 0 in
  this primitive), not the manifest’s nominal `[18, 18]`; this does not change
  the current median but remains a fixture-realism limitation.
