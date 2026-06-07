# Issue #2438 Static-Recenter Activation Closure Evidence

This directory contains compact diagnostic-only closure evidence for Issue #2438.

It does not contain new raw simulator traces. It reuses the tracked Issue #2306 instrumented rerun
and the Issue #2402 field-mapped synthesis because those artifacts already satisfy the requested
activation fields for the same held-out smoke.

Source artifacts:

- `docs/context/evidence/issue_2306_static_recenter_activation_trace_2026-06-05/summary.json`
- `docs/context/evidence/issue_2402_static_recenter_activation_2026-06-06/summary.json`

Decision:

- Overall Issue #2438 classification: `mechanism_inactive`
- Solved-row secondary classification: `comparator_already_solved_case`
- Recommendation: `stop_current_heldout_transfer_route`

This evidence is not benchmark-strength, transfer, planner-improvement, or paper-facing evidence.
