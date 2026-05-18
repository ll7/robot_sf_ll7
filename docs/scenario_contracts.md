# Scenario Contracts

[← Back to Documentation Index](./README.md)

`scenario_contract.v1` is the machine-readable governance surface for authored scenario intent.
It records what a scenario is meant to represent before execution: operating-design-domain
assumptions, actor models, invariants, observables, termination semantics, provenance, and the
eligibility hooks that must be satisfied before benchmark claims.

It is deliberately separate from `scenario_cert.v1`.

## Contract Boundary

Use the two scenario surfaces for different jobs:

- `scenario_contract.v1`: authored intent and reviewable assumptions.
- `scenario_cert.v1`: fail-closed feasibility and benchmark eligibility classification.
- benchmark episode records and reports: executed evidence for planner behavior, metrics, seeds,
  and outcomes.

A valid scenario contract is not proof that a scenario is feasible or benchmark-ready. Benchmark
claims still require certification plus run evidence.

## Public Files

- Schema:
  [`robot_sf/benchmark/schemas/scenario_contract.v1.json`](../robot_sf/benchmark/schemas/scenario_contract.v1.json)
- Loader:
  [`robot_sf/benchmark/scenario_contract.py`](../robot_sf/benchmark/scenario_contract.py)
- Fixture:
  [`configs/scenarios/contracts/station_platform_candidate_pack_issue736_contracts.yaml`](../configs/scenarios/contracts/station_platform_candidate_pack_issue736_contracts.yaml)

The fixture uses the station-platform candidate pack because that YAML already carried exploratory
intent metadata such as density, flow, coverage probes, and evaluation scope. The contract fixture
normalizes one representative entry without changing any scenario runner behavior.

Scenario contracts may also reference an ODD declaration instead of duplicating all operating
assumptions:

```yaml
odd_contract_ref:
  source: configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml
  contract_id: low_speed_public_space_v1
  required_for_benchmark_claim: true
```

See [ODD Contracts](./odd_contracts.md) for the ODD metadata boundary.

## Library API

```python
from pathlib import Path

from robot_sf.benchmark.scenario_contract import (
    load_scenario_contracts,
    validate_scenario_odd_contract_reference,
    validate_scenario_contract_references,
)

contracts = load_scenario_contracts(
    Path("configs/scenarios/contracts/station_platform_candidate_pack_issue736_contracts.yaml")
)
errors = validate_scenario_contract_references(contracts[0], repo_root=Path("."))
odd_errors = validate_scenario_odd_contract_reference(contracts[0], repo_root=Path("."))
```

`load_scenario_contracts(...)` validates JSON Schema first and raises
`ScenarioContractValidationError` with JSON-pointer-style paths for invalid actor, invariant,
observable, termination, or provenance fields. Reference validation is explicit so tools can decide
whether to require local scenario YAML availability.

## Authoring Guidance

Keep v1 contracts narrow:

- use closed enums for actor kind, termination reason, intended use, and ODD class,
- reference `odd_contract.v1` declarations for shared operating assumptions when benchmark claims
  depend on ODD metadata,
- put future or experimental vocabulary under `extensions` only when it cannot fit the v1 fields,
- point `scenario_ref.source` at a maintained scenario YAML file,
- set `certification.required_before_benchmark_claim: true` unless the file is a test fixture,
- keep `benchmark_eligibility.claim_boundary` conservative and specific.

Do not use a contract to promote a scenario into a paper-facing matrix. Promotion requires
`scenario_cert.v1` eligibility, explicit scenario-set inclusion, seed policy, and benchmark run
evidence.
