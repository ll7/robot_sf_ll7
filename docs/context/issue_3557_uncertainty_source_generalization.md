# Issue #3557 — Uncertainty-source generalization decision layer (increment)

**Status:** diagnostic / proxy (same caveats as #3471). **Evidence grade:** idea-level decision
layer.

## What this is

`robot_sf/representation/uncertainty_source_generalization.py` is the pure **decision layer** for
#3557. #3471 found the unsafe-dropping effect using a single uncertainty source
(existence-degradation); `ScenarioBelief` supports others (visibility/occlusion, covariance,
class-probability, tracking noise). This module turns a per-source retained-vs-dropped safety
contrast — run via the #3471 harness parameterized by source — into the issue's deliverable: a
per-source verdict plus whether the finding **generalizes**.

It mirrors the accepted decision-layer pattern in `failure_cause.py` (#3484) and
`stream_gap_gate_calibration.py` (#3558).

## Decision layer (`uncertainty_source_generalization.v1`)

- `classify_source(contrast, thresholds)` → `reproduces_unsafe_dropping` (dropping raises
  unsafe-commit or lowers separation beyond the detectable threshold), `no_unsafe_dropping_effect`,
  or `inconclusive` (sub-threshold difference / too-few episodes).
- `assess_source_generalization(contrasts, thresholds)` → per-source verdicts and a generalization
  verdict: `generalizes` (all measurable sources reproduce), `does_not_generalize` (none do),
  `source_specific` (mixed), or `inconclusive` (no measurable source). Inconclusive sources are
  excluded from the generalization call.

## Scope boundary

Pure and side-effect free. Parameterizing the #3471 episode harness by uncertainty source and
running it per source (reusing the #3450 condition builders) needs benchmark runs and is the
deliberate deferred follow-up.

## Tests

`tests/representation/test_uncertainty_source_generalization.py` (9 tests): reproduce / no-effect /
inconclusive (sub-threshold and thin-evidence) classification, and generalizes / source-specific /
does-not-generalize / inconclusive aggregation, plus empty-input rejection.

## Related

- Follows #3471 (PR #3553), #3450, #2546. Sibling: #3558 (gate-threshold calibration).
