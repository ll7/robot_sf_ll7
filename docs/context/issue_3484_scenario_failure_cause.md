# Issue #3484 — Universally-failing scenario failure-cause classifier (first increment)

**Status:** diagnostic / proxy. **Evidence grade:** idea-level; verdicts are versioned
modeling choices, diagnostic until validated. Answers the examiner challenge "no planner
succeeds in several scenarios — are they valid?" by separating scenario defects from
planner limits.

## What this is

`robot_sf/scenario_certification/failure_cause.py` is the **pure decision layer** that
reclassifies a universally-failing scenario family from per-family diagnostic outcomes.
A benchmark row no method can pass is useful for diagnosis but weak for comparative
ranking, so the cause must be disambiguated before the row is used.

## Classifier (`scenario_failure_cause.v1`)

`classify_failure_cause(FamilyDiagnostics)` consumes:
`any_planner_succeeded`, `route_feasible`, `actor_free_solved`, `extended_time_solved`,
`oracle_solved` (each may be `None` when not run), and returns a verdict with `cause`,
`comparable_for_ranking`, `evidence_complete`, `rationale`, and echoed `inputs`.

Decision order (first match wins):

1. a planner passed → `not_universally_failing`
2. `route_feasible is False` → `infeasible_route` (scenario defect)
3. `actor_free_solved is False` → `vehicle_infeasible` (footprint/adapter/limit defect)
4. `extended_time_solved is True` → `time_limited` (time limit binds)
5. `oracle_solved is True` → `planner_limited` (**only** rankable cause)
6. route + actor-free succeed but `oracle_solved is False` →
   `dynamic_blocking_or_deadlock`
7. otherwise → `indeterminate`

`comparable_for_ranking` is true **only** for `planner_limited`; every other cause is a
scenario/time/feasibility defect that must be excluded from ranking claims.

## Scope boundary

Pure and side-effect free — no runtime/benchmark behavior change. The sim-integrated
diagnostic **runners** (geometric clearance certification, oracle/scripted trajectory,
actor-free run, extended-time, difficulty ramp) and ledger emission are deliberate
follow-ups; this layer turns their outcomes into a reproducible verdict.

## Tests

`tests/scenario_certification/test_failure_cause.py`: each verdict branch, the
precedence order, the rankability rule, `indeterminate` on missing diagnostics, and the
versioned schema / input echo.

## Related

- Existing route-clearance certification: `configs/benchmarks/route_clearance_certifications_v1.yaml`,
  `robot_sf/scenario_certification/v1.py`.
- Safety-event ledger (emit target): #3482.
