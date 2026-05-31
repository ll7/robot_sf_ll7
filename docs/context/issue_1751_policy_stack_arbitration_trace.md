# Issue 1751 Policy Stack Arbitration Trace Packet

Related issue: [#1751](https://github.com/ll7/robot_sf_ll7/issues/1751)
Parent assessment PR: [#1749](https://github.com/ll7/robot_sf_ll7/pull/1749)
Contract predecessor: [issue_926_policy_stack_v1_contract.md](issue_926_policy_stack_v1_contract.md)
Runtime predecessor: [issue_1004_policy_stack_v1_runtime.md](issue_1004_policy_stack_v1_runtime.md)

## Decision

`policy_stack_v1` now exposes a diagnostic-only arbitration trace packet:

```text
policy_stack_v1.arbitration_trace_packet.v1
```

This packet is a data-contract preflight for future learned arbitration. It does not train, enable,
or ship a learned arbiter.

## Proposal And Command Contract

The packet records:

- configured `proposal_sources`;
- action space: `unicycle_vw`;
- command fields: `linear_velocity_m_s`, `angular_velocity_rad_s`;
- configured maximum linear and angular command bounds;
- source status vocabulary: `native`, `adapter`, `fallback`, `degraded`, `failed`,
  `not_available`, `rejected`;
- executable proposal statuses: `native`, `adapter`, `fallback`, `degraded`.

Per-step trace diagnostics include the selected proposal, proposal status counts, proposal commands,
rejection reasons, risk-score components, and `candidate_ranking` sorted by total score. The new
`last_decision()` accessor returns the most recent step decision for tooling that wants one packet
without parsing the whole episode diagnostics payload.

## Observation And Leakage Boundary

Inference-available features are limited to:

- `robot.position`
- `robot.heading`
- `goal.current`
- `goal.next`
- `pedestrians.positions`

Leakage exclusions are explicit:

- `future_trajectory`
- `simulator_collision_label`
- `episode_success_label`
- `route_outcome_label`
- `benchmark_metric_rollups`

Any future learned arbiter must keep those exclusions unless a later issue changes the observation
contract with its own proof.

## Switching And Dwell

The v1 trace records one arbitration decision per planner step. `min_dwell_steps=1` is declared as
the current trace-level minimum, and dwell enforcement is marked
`not_enforced_in_v1_trace`. A learned arbiter may not silently infer a stronger dwell guarantee from
this packet; any future dwell constraint must be implemented and recorded explicitly.

## Fallback Policy

Fallback and degraded proposals are caveat labels. `failed`, `not_available`, and `rejected`
proposals are non-executable diagnostic labels. None of these are successful training targets unless
a future data-preparation issue explicitly filters or labels them.

## Smoke Command

The compact shape check is:

```bash
uv run python scripts/validation/validate_policy_stack_trace_packet.py
```

To inspect the packet JSON:

```bash
uv run python scripts/validation/validate_policy_stack_trace_packet.py --json
```

The smoke uses an in-process fixture, not a benchmark run, so it proves packet shape without
changing benchmark metrics.
