# Issue #3546 Signalized Failure-Predicate Contract

Evidence status: contract validation, not benchmark-result evidence.

Issue #3544 proved the live signalized smoke run can produce planner-observable signal metric rows,
but the issue #2754 failure-pack builder did not classify red-phase signal violations as
failure-pack cases. This bundle records the #3546 contract decision: planner-observable
signal-specific metrics are intentional failure-pack predicates when their thresholds are met.

## Decision

The signalized failure-pack builder now treats these metrics as signal-specific failure predicates:

- `signal_red_phase_violations >= 1`
- `signal_stop_line_crossings_under_red >= 1`

Generic social-navigation predicates still apply through the canonical failure extractor:
collisions, comfort exposure, and near misses. Unavailable, proxy-only, fallback, degraded,
synthetic, stale, or denominator-excluded rows still fail closed for figure eligibility through the
existing provenance and signal-denominator checks.

## Claim Boundary

This is a builder-contract update. It does not by itself produce a new live positive signalized
failure-pack case, planner-ranking comparison, traffic-light realism claim, or paper-facing figure.
A future positive pack still needs live or durable-replay trace and metric inputs with source-kind
metadata.

## Validation

```bash
uv run pytest tests/analysis/test_build_signalized_crossing_failure_pack_issue_2754.py -q
```

The focused test suite includes a signal-specific positive fixture where generic failure metrics are
zero and signal metrics trigger a case, plus a malformed/zero signal-metric fixture that remains a
negative control.

## Files

- `summary.json`: machine-readable contract decision and validation boundary.
