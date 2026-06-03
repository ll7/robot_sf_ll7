# Issue #2168 AMMV-Aware Social Force Pair Diagnostic 2026-06-03

This evidence pack records a local diagnostic, not benchmark-strength or
paper-facing evidence. The benchmark probe ran the same single scenario
with default Social Force and AMMV-aware Social Force; the mechanism probe
checks whether the AMMV term activates in a controlled close-agent setup.

## Benchmark Probe

- Scenario: `classic_head_on_corridor_low`.
- Seeds: `[111, 112, 113]`.
- Default status counts: `{'failure': 3}`.
- AMMV status counts: `{'failure': 3}`.
- Default min clearance: `0.484367` m.
- AMMV min clearance: `0.484367` m.
- Default mean speed: `1.215808` m/s.
- AMMV mean speed: `1.215808` m/s.
- AMMV metadata surfaced in episode JSONL: `False`.

## Mechanism Probe

- Probe: `issue_2168_close_front_agent_probe`, seed `42`.
- AMMV max force magnitude: `2.641802`.
- AMMV max intrusion count: `1`.
- Final robot lateral-offset delta: `0.201059` m.

## Interpretation

The named benchmark slice produced identical ordinary metrics for the two
planner configurations and timed out without collisions in all rows. The
episode records did not surface AMMV force metadata. The direct mechanism
probe shows the AMMV force term can activate, but pedestrian lateral
deviation and speed adaptation are unsupported by this robot-planner-only
probe. Classify this result as diagnostic only.
