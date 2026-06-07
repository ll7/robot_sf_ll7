# Issue 2463 Mechanism Signal Checker

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2463>

## What Was Implemented

`scripts/validation/check_mechanism_signal.py` classifies a baseline/intervention trace pair before
the pair is routed as mechanism evidence.

Example:

```bash
uv run python scripts/validation/check_mechanism_signal.py \
  --baseline-trace <baseline-trace.json> \
  --intervention-trace <intervention-trace.json>
```

Output is JSON under `mechanism_signal`:

- `schema_version`
- `trajectory_delta_nonzero`
- `command_delta_nonzero`
- `mechanism_field_delta_nonzero`
- `activation_delta_nonzero`
- `outcome_delta_nonzero`
- `classification`

## Classification Boundary

- `rendering_sanity`: no trajectory, command, mechanism, activation, or outcome delta was found.
- `qualitative_illustration`: trajectory, command, or outcome changed, but no mechanism or
  activation field changed.
- `mechanism_difference_candidate`: at least one mechanism or activation field changed.

The checker is a routing guard only. A nonzero signal does not establish planner superiority,
transfer, benchmark success, or paper-grade mechanism proof.

Mechanism and activation detection uses conservative key-fragment matching across the trace payload.
False positives should route a pair to follow-up review, not to a stronger claim.

## Mechanism Panel Publication Rule

Before a trace pair is rendered, published, or routed as mechanism-panel evidence, panel work should
record one of the checker classifications above. The minimum acceptable record is the checker output
or an equivalent validation note that names the baseline trace, intervention trace, command, commit,
classification, and any missing fields.

Use the classifications as follows:

- `rendering_sanity`: the panel may be kept as a loader/rendering sanity fixture, not as a
  behavioral-difference or mechanism claim.
- `qualitative_illustration`: the panel may illustrate a scene or behavior change, but it should
  not be cited as mechanism evidence without a follow-up signal explanation.
- `mechanism_difference_candidate`: the pair can proceed to mechanism-panel review, still bounded
  by the trace-mechanism rubric and any benchmark row-status caveats.

An explicit null-result demonstration is allowed when the point of the panel is to show that a
candidate pair has no mechanism-relevant signal. In that case, label it as diagnostic/null evidence
and avoid mechanism-difference language.

## Proof Surface

`tests/validation/test_check_mechanism_signal.py` covers zero-delta, behavior-only delta,
mechanism/activation delta, and CLI JSON output cases using synthetic trace fixtures.
