# Issue #2474 Signalized Pedestrian Crossing Benchmark Scope (2026-06-06)

Status: scoped benchmark direction with one simulator-backed denominator smoke; not a
traffic-light compliance or planner-reasoning claim.

Related surfaces:

- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2474
- Parent roadmap issue: https://github.com/ll7/robot_sf_ll7/issues/2469
- Benchmark manifest: `configs/benchmarks/signalized_pedestrian_crossing_issue_2474.yaml`
- Existing crossing archetype: `configs/scenarios/archetypes/classic_urban_crossing.yaml`
- Existing proxy scenario: `configs/scenarios/single/francis2023_intersection_wait.yaml`
- Current signal-state proxy smoke:
  [issue_2564_signal_state_proxy_smoke.md](issue_2564_signal_state_proxy_smoke.md)
- Simulator-backed denominator smoke:
  [evidence/issue_2799_signalized_runtime/README.md](evidence/issue_2799_signalized_runtime/README.md)
- Existing phase-grid proxy:
  `configs/scenarios/perturbations/issue_1610_intersection_wait_phase_grid_v1.yaml`
- Fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`

## Result

Issue #2474 asks for a signalized pedestrian crossing benchmark direction. This pass defines the
scenario families, signal/phase semantics, metric requirements, trace fields, and first smoke path
without adding a traffic-signal simulator or claiming planner reasoning.

The current repository already has crossing-like scenarios, an intersection-wait phase-grid proxy,
the #2564 trace-only signal-state wrapper around the waiting-then-crossing fixture, and the #2799
runtime denominator smoke. The #2799 smoke proves that simulator-backed rows can now distinguish
planner-observable denominator rows from `unavailable` and `proxy_diagnostic` exclusions, including
red-required-stop and green-proceed row classes. It does not prove forced-waiting reasoning,
traffic-signal realism, or crossing-legality compliance.

## Claim Boundary

This is proposal, interface, and denominator-plumbing evidence only. It does not prove
forced-waiting reasoning, signal-legality compliance, traffic-signal realism, or planner
performance. Existing `wait_at`, timing perturbations, and the #2564 trace-only `signal_state`
wrapper remain proxy semantics; the #2799 smoke only promotes rows that explicitly declare the
planner-observable signal-state runtime contract.

## Required Future Proof

A benchmark-strength signalized crossing comparison still needs:

- explicit signal phase and right-of-way state in scenario/runtime data beyond trace-only proxy
  metadata;
- planner-observation policy for whether signal state is visible, hidden, or motion-only;
- trace fields for signal phase timeline, zone entry/exit, forced wait intervals, and legality
  events;
- fail-closed reporting under the benchmark fallback policy;
- a one-planner smoke before any planner comparison.

The #2799 smoke satisfies the one-planner denominator/exclusion smoke prerequisite, but not the
planner comparison, legality, or forced-waiting reasoning prerequisites.

## Validation

Targeted validation:

```bash
uv run pytest tests/benchmark/test_signalized_crossing_benchmark_manifest.py -q
uv run ruff check tests/benchmark/test_signalized_crossing_benchmark_manifest.py
uv run ruff format --check tests/benchmark/test_signalized_crossing_benchmark_manifest.py
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
git diff --check
```

## Follow-Up Boundary

The recommended next issue is an executable one-scenario spike that adds explicit signal phase
state, emits the required trace fields, and runs one baseline-safe planner smoke. Proxy phase-grid
rows should remain diagnostic-only until then.
