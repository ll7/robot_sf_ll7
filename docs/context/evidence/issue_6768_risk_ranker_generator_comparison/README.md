<!-- AI-GENERATED (robot_sf#6768, 2026-08-06) - NEEDS-REVIEW -->

# Risk-ranker generator comparison (primitive vs RBF)

This offline diagnostic compares the deterministic primitive candidate generator
(`generate_primitive_candidates`) with the deterministic radial-basis-function (RBF)
candidate generator (`generate_rbf_candidates`, merged in #6676) on the same held-out
fixtures, with equal candidate budgets and identical risk-estimator, ranking-weight, and
hard-gate configuration. It never wires either generator into a planner loop.

## Claim boundary

- Evidence status: `diagnostic_only`.
- Baseline: `deterministic_primitive` from `generate_primitive_candidates`.
- Variant: opt-in deterministic RBF proposals from `generate_rbf_candidates`; this is not a trained policy.
- The report separately records candidate validity (including unique waypoint-sequence checks), `verify_trajectory` fallback-brake
  rejection, actuator-gate rejection, eligible/selected-candidate identity and whether
  generator choice changes selection, decomposed risk/time/jerk/path-length/clearance
  components, a model-risk reliability diagnostic where a fixture declares a known contact
  outcome, generation/ranking/total timing, and unavailable denominators with reasons.
- Risk scores are constant-velocity model scores, not calibrated real-world collision
  probabilities. Declared contact outcomes check deterministic geometry agreement; they are not
  independent calibration or reliability evidence. Only finite, in-range, complete-provenance,
  repeatable, non-degraded rows enter the declared-outcome denominator; otherwise that diagnostic
  is reported as unavailable or inconclusive. Timing is measured local offline wall time when
  requested, not online performance evidence; pinned deterministic reports omit it.
- Planner-loop wiring, online adaptation, nominal benchmark execution, planner improvement,
  safety, and real-world claims remain deferred.

## Split integrity

The config names a calibration fixture and a held-out evaluation fixture that are disjoint by
case id and split label:

- Evaluation (held out): `tests/fixtures/risk_ranker/issue_6768_risk_ranker_held_out_v1.yaml`
- Calibration: `tests/fixtures/risk_ranker/issue_6768_risk_ranker_calibration_v1.yaml`

The script fails closed on split overlap, a missing or malformed `diagnostic_only` fixture claim
boundary, missing or non-valid case status, missing fixture provenance, a known contact outcome
without an explicit candidate role, non-finite values, unequal candidate budgets, or a
generator/config hash mismatch.

## Reproduction

From the repository root, run:

```bash
uv run python scripts/analysis/compare_risk_ranker_generators_issue_6768.py \
  --config configs/analysis/issue_6768_risk_ranker_generator_comparison.yaml \
  --output <external-report>.json \
  --output-md <external-report>.md
```

Validate the committed config (including every fail-closed gate) with:

```bash
uv run python scripts/analysis/compare_risk_ranker_generators_issue_6768.py \
  --check-config configs/analysis/issue_6768_risk_ranker_generator_comparison.yaml
```

The JSON report records the command, config and fixture SHA-256 digests, calibration/evaluation
split integrity, seed, exact commit SHA, matched-comparison policy, generator/config hash,
fallback/degraded exclusions, and unavailable denominators. A valid pinned generation timestamp
(`--generated-at`) produces byte-stable JSON and Markdown; in that mode the timing section is
explicitly marked `not_measured_for_pinned_determinism`. Omit `--generated-at` when measured local
wall-clock timing is needed. Keep generated reports outside the worktree or promote only a small
provenance-bearing manifest; do not commit raw timing logs.

The report is diagnostic evidence for issue
[#6768](https://github.com/ll7/robot_sf_ll7/issues/6768), not nominal benchmark evidence.
