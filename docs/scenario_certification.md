# Scenario Certification

[← Back to Documentation Index](./README.md)

`scenario_cert.v1` is the first machine-readable certification surface for generated and curated
scenario manifests. It is intentionally conservative: malformed scenarios, missing inflated paths,
kinodynamic violations, and clearly blocked dynamic setups are excluded before they can support
benchmark claims.

For authored scenario intent before execution, use
[`scenario_contract.v1`](./scenario_contracts.md). A scenario contract records ODD assumptions,
actor models, invariants, observables, termination semantics, and provenance; it does not replace
the fail-closed feasibility and eligibility checks described here.

## Contract

The public schema lives at
[`robot_sf/benchmark/schemas/scenario_cert.v1.json`](../robot_sf/benchmark/schemas/scenario_cert.v1.json).
Each certificate includes:

- `schema_version`: always `scenario_cert.v1`.
- `scenario_id` and `source`: the scenario name/id and manifest or programmatic source.
- `classification`: one of `valid`, `invalid`, `geometrically_infeasible`,
  `kinodynamically_infeasible`, `dynamically_overconstrained`, `knife_edge`, or
  `hard_but_solvable`.
- `benchmark_eligibility`: `eligible`, `stress_only`, or `excluded`.
- `checks`: deterministic geometry, route, planner, kinodynamic, and dynamic checks.
- `route_certificates`: per-route evidence for every applicable robot route.
- `evidence`: optional scenario metadata and scenario-difficulty provenance.

Benchmark inclusion policy:

- `valid` and `hard_but_solvable` are benchmark-eligible.
- `knife_edge` is stress-only and should not be promoted as headline benchmark evidence without
  an explicit benchmark issue.
- `invalid`, `geometrically_infeasible`, `kinodynamically_infeasible`, and
  `dynamically_overconstrained` are excluded.

## Checks

The v1 certifier uses the repository scenario loader, so `map_id`, `map_file`, route overrides,
single-pedestrian overrides, and robot config parsing follow the same path as training and
benchmark tools.

Geometry checks:

- finite start and goal coordinates within map bounds,
- start/goal not inside static obstacles,
- inflated global path existence using the classic A* planner with no inflation fallback,
- shortest inflated path length,
- path length ratio against direct start-goal distance,
- authored route static clearance against obstacles.

Kinodynamic checks:

- differential-drive and holonomic robots are considered command-feasible when their existing
  settings validate, because they can rotate in place,
- bicycle-drive routes are excluded when the authored route contains a turn tighter than the
  configured `wheelbase / tan(max_steer)` limit.

Dynamic checks:

- moving single pedestrians are optional hardness evidence,
- static single pedestrians whose start position blocks the inflated robot route corridor classify
  the scenario as `dynamically_overconstrained`.

Infrastructure checks:

- map definitions may include optional `infrastructure_zones` metadata for public-space semantics
  such as `pedestrian_only`, `stairs`, `pedestrian_exit`, `signalized_crossing_zone`, or
  `shared_space_lane`,
- each infrastructure zone names its polygon vertices and `allowed_actor_types`,
- robot/AMV routes that intersect a zone whose allowed actors do not include `amv`, `robot`,
  `vehicle`, or `all` fail closed as `invalid`,
- these zones are certification-only metadata and do not change simulator physics, obstacle
  collision, route planning, or benchmark execution by themselves.

Scenario difficulty and planner residual analysis from
[`docs/context/issue_692_scenario_difficulty_analysis.md`](./context/issue_692_scenario_difficulty_analysis.md)
is linked as diagnostic evidence only. It is not treated as a replacement for validity or
feasibility checks.

## CLI

Generate a batch JSON document:

```bash
uv run python scripts/tools/certify_scenarios.py \
  configs/scenarios/sets/atomic_navigation_validation_fixtures_v1.yaml \
  --output output/scenario_cert/atomic_validation.json
```

Generate one JSON object per line:

```bash
uv run python scripts/tools/certify_scenarios.py \
  configs/scenarios/sets/atomic_navigation_validation_fixtures_v1.yaml \
  --jsonl
```

Use `--scenario-id <name>` to certify one scenario. Use `--fail-on-excluded` in gates that should
exit non-zero when any scenario is excluded.

## Library API

```python
from pathlib import Path

from robot_sf.scenario_certification import certify_scenario_file, certificate_to_dict

certificates = certify_scenario_file(
    Path("configs/scenarios/sets/atomic_navigation_validation_fixtures_v1.yaml")
)
payload = [certificate_to_dict(certificate) for certificate in certificates]
```

Programmatic tests can call `certify_map_definition(...)` directly with a `MapDefinition`.

## Limits

`scenario_cert.v1` is a first-pass fail-closed contract, not a high-budget oracle planner. It does
not generate adversarial scenarios, promote stress cases into headline benchmarks, or prove that a
planner will solve a valid scenario. It only certifies that the scenario geometry and currently
exposed robot/dynamic constraints are not malformed or impossible under the v1 checks.

`scenario_contract.v1` may explain what a scenario is meant to exercise, but a valid intent
contract does not make an excluded or uncertified scenario benchmark evidence.

For h500 interpretation, layer planner-failure classification on top of certification rather than
using h500 failures as certification evidence by themselves. Excluded or unresolved-certification
scenarios cannot support planner-failure attribution. Eligible and `hard_but_solvable` scenarios may
support planner follow-ups when the same mechanism recurs across seeds or planners. See
`docs/context/issue_1056_h500_failure_classification.md` for the current h500 classification
vocabulary.
