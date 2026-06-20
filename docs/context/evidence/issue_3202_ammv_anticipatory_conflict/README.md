# Issue #3202 AMMV Anticipatory-Conflict Diagnostic

This evidence pack records a local diagnostic, not benchmark-strength or
paper-facing evidence. The benchmark probe ran the same single scenario
with default Social Force and AMMV-aware Social Force; the mechanism probe
checks whether the AMMV term activates in a controlled close-agent setup.

## Benchmark Probe

- Scenario: `issue_3202_ammv_anticipatory_conflict`.
- Seeds: `[3202]`.
- Default status counts: `{'failure': 1}`.
- AMMV status counts: `{'failure': 1}`.
- Default min clearance: `1.170978` m.
- AMMV min clearance: `1.170978` m.
- Default mean speed: `1.373952` m/s.
- AMMV mean speed: `1.373952` m/s.
- AMMV metadata surfaced in episode JSONL: `False`.

## Mechanism Probe

- Probe: `issue_3202_anticipatory_crossing_probe`, seed `3202`.
- AMMV max force magnitude: `2.642146`.
- AMMV max intrusion count: `1`.
- Verdict: `behavioral_delta_found`.
- Mean speed delta: `0.074366` m/s.
- Max lateral-velocity delta: `0.783477` m/s.
- Min robot-pedestrian clearance delta: `0.141140` m.
- Final robot lateral-offset delta: `-0.000000` m.

## Interpretation

The named benchmark slice is contextual diagnostic output. The direct
mechanism probe is the AMMV-specific same-seed comparison surface because
it runs `SocialForcePlanner` directly and exposes AMMV force diagnostics.
Pedestrian lateral deviation and speed adaptation remain unsupported by
this robot-planner-only probe. Classify this result as diagnostic only.

## Limitations

- One selected diagnostic slice only; no benchmark-strength or paper-facing claim.
- Benchmark runner episode records did not surface AMMV force/intrusion metadata.
- Both planner rows ran in adapter mode under differential-drive benchmark execution.
- Pedestrian lateral deviation and speed adaptation are unsupported by the direct robot-planner mechanism probe.
- `scripts/validation/run_multi_amv_smoke.py` is not used because it is a multi-robot goal-controller smoke, not a default-vs-AMMV Social Force harness.
- The direct probe resolves #3202 as diagnostic mechanism evidence; it does not promote AMMV to benchmark-valid comparison evidence.
