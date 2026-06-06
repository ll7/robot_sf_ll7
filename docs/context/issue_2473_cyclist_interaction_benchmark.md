# Issue #2473 Cyclist Interaction Benchmark Scope (2026-06-06)

Status: scoped benchmark direction, not benchmark evidence.

Related surfaces:

- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2473
- Parent roadmap issue: https://github.com/ll7/robot_sf_ll7/issues/2469
- Benchmark manifest: `configs/benchmarks/cyclist_interaction_issue_2473.yaml`
- Cross-kinematics benchmark: `configs/benchmarks/cross_kinematics_v1.yaml`
- Cross-kinematics compatibility contract:
  `configs/benchmarks/cross_kinematics_v1_compatibility.yaml`
- AMV comparability map: `configs/benchmarks/alyassi_comparability_map_v1.yaml`
- Overtaking proxy archetype: `configs/scenarios/archetypes/classic_overtaking.yaml`
- Fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`

## Result

Issue #2473 asks for a cyclist interaction benchmark direction. This pass defines the scenario
families, dynamics assumptions, required metrics, trace fields, and first executable smoke boundary
without adding a cyclist simulator backend or claiming planner performance.

The current repository has adjacent surfaces: robot bicycle-drive kinematics, pedestrian
overtaking/crossing scenarios, shared-space micromobility notes, and AMV actuation stress metadata.
Those are useful proxy surfaces for scoping, but they are not cyclist actor runtime support.

## Claim Boundary

This is proposal and interface evidence only. It does not prove cyclist behavior realism, cyclist
dynamics fidelity, local-planner performance, or benchmark coverage of cyclist interactions.
Existing pedestrian/overtaking and robot bicycle-drive surfaces remain proxy inputs until an
executable cyclist actor/scenario path exists.

## Required Future Proof

A benchmark-strength cyclist interaction row needs:

- explicit `actor_type` and cyclist state in scenario/runtime data;
- speed, acceleration, heading/yaw-rate, route/lane intent, and turn-radius assumptions;
- trace fields for relative closing speed, time-to-conflict, pass clearance, and forced braking;
- cyclist-specific metrics separate from generic pedestrian near-miss or collision counts;
- fail-closed reporting under the benchmark fallback policy;
- a one-planner smoke before any planner comparison.

## Validation

Targeted validation:

```bash
uv run pytest tests/benchmark/test_cyclist_interaction_benchmark_manifest.py -q
uv run ruff check tests/benchmark/test_cyclist_interaction_benchmark_manifest.py
uv run ruff format --check tests/benchmark/test_cyclist_interaction_benchmark_manifest.py
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
git diff --check
```

## Follow-Up Boundary

The recommended next issue is an executable one-scenario spike that adds one cyclist actor crossing
or overtaking scenario, emits the required trace fields, and runs one baseline-safe planner smoke.
Proxy surfaces should remain diagnostic-only until then.
