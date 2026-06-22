# Issue #3372 Occluder-Timing Counterfactual Pilot

**Date:** 2026-06-22 · **Evidence tier:** `diagnostic-only` · **Parent:** #3369

## Source

This bundle records a compact local run of the Issue #3369 occluder-timing perturbation pilot on
durable, tracked inputs, synthesized as diagnostic counterfactual sensitivity. The run pairs the
`noop` baseline against a single `occluder_timing_offset` variant that delays the emerging
pedestrian release by `+0.5 s` while holding the source scenario, seed, map, and route geometry
fixed.

- Manifest: `configs/scenarios/perturbations/issue_3369_occluder_timing_pilot_v1.yaml`
- Scenario fixture: `configs/scenarios/single/issue_2756_occluded_emergence_live.yaml`
- Baseline commit: `877b603831309af9ddb93df49093096521209db4`
- Planner: `goal` · Seed: `111` · Horizon: `80` · dt: `0.1`

Preflight reported both variants `valid` (no exclusions); the perturbed variant carried the
`dynamic_interaction_present` eligibility reason. Materialized scenario matrix and route overrides
remain ignored local `output/` artifacts that are reproducible from the tracked manifest and the
commands in `summary.json`. Raw `trace_response.json` / `.md` are likewise ignored local outputs;
`summary.json` is the compact tracked representation.

## Result

Both rows are valid diagnostic results (fail-closed classification: `valid_diagnostic_result`).
Each variant ran the full horizon (`termination_reason: max_steps`); the pair completed.

| Row | Variant | Closest-approach step | Clearance (m) | Center distance (m) |
|---|---|---:|---:|---:|
| noop baseline | `..._noop` | 79 | 20.918 | 22.318 |
| occluder +0.5 s | `..._occluder_timing_h1_p050` | 79 | 25.518 | 26.918 |

Mean closest-approach deltas (perturbed − noop), `goal` planner, 1 pair completed:

- `clearance_m`: `+4.600`
- `center_distance_m`: `+4.600`
- `time_s`: `0.000`
- `goal_distance_m`: `0.000`
- `progress_m`: `0.000`

A `+0.5 s` occluder release delay increased the robot's closest-approach clearance to the emerging
pedestrian by ~`4.60 m`, with the closest approach still occurring at step 79 (positive clearance
delta = perturbed run was farther from the closest pedestrian at its closest approach). This is
consistent with the pedestrian emerging later, so the robot clears the conflict zone with more
margin under this single seed and planner.

## Claim Boundary

- Diagnostic-only counterfactual sensitivity input/result; **not** benchmark-strength, causal, or
  paper-facing evidence.
- Single seed (`111`), single planner (`goal`), single perturbation magnitude (`+0.5 s`): the
  reported delta is a per-pair diagnostic observation, not a distribution or a directional claim.
- No fallback or degraded rows are promoted as success; preflight eligibility is input validation
  only.
- Reproduce raw local output from the tracked manifest and the commands recorded in `summary.json`.
