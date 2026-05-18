# ODD Contracts

[← Back to Documentation Index](./README.md)

`odd_contract.v1` is the machine-readable Operational Design Domain metadata surface for benchmark
and falsification evidence. It records the operating assumptions that bound a result: public-space
context, agent classes, speed envelope, pedestrian-density envelope, observation assumptions,
explicit exclusions, claim boundaries, and provenance.

An ODD contract is not a safety certificate, legal compliance claim, benchmark result, or scenario
feasibility certificate. It is metadata that helps benchmark claims and counterexamples state what
their evidence is allowed to cover.

## Contract Boundary

Use the related validation surfaces for different jobs:

- `odd_contract.v1`: operating assumptions and non-claim boundaries for evidence artifacts.
- `scenario_contract.v1`: authored scenario intent and scenario-specific assumptions.
- `scenario_cert.v1`: fail-closed feasibility and benchmark eligibility classification.
- benchmark episode records and reports: executed planner behavior, metrics, seeds, and outcomes.

ODD metadata can bound a claim, but it cannot promote a scenario, certify safety, or replace
execution evidence.

## Public Files

- Schema:
  [`robot_sf/benchmark/schemas/odd_contract.v1.json`](../robot_sf/benchmark/schemas/odd_contract.v1.json)
- Loader:
  [`robot_sf/benchmark/odd_contract.py`](../robot_sf/benchmark/odd_contract.py)
- Fixture:
  [`configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml`](../configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml)

The first fixture covers the current low-speed public-space micromobility lane used by benchmark
and falsification work. It is intentionally conservative and records exclusions such as
public-road autonomy, wet/icy dynamics, legal safety certification, and real-world deployment
readiness.

## Library API

```python
from pathlib import Path

from robot_sf.benchmark.odd_contract import (
    classify_odd_claim_boundary,
    load_odd_contracts,
    validate_odd_contract_references,
)

contracts = load_odd_contracts(
    Path("configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml")
)
errors = validate_odd_contract_references(
    source="configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml",
    contract_id=contracts[0].id,
    repo_root=Path("."),
)
status = classify_odd_claim_boundary(contracts[0], "safety_certification")
```

`load_odd_contracts(...)` validates JSON Schema first and raises `OddContractValidationError` with
JSON-pointer-style paths for invalid sections. Reference validation is explicit so callers can
choose when local config availability is required.
`classify_odd_claim_boundary(...)` returns `supported`, `excluded`, or `unknown` for compact claim
IDs against `supported_claims`, `non_claims`, and `exclusions`.

## Scenario Contract References

`scenario_contract.v1` supports an optional `odd_contract_ref` block:

```yaml
odd_contract_ref:
  source: configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml
  contract_id: low_speed_public_space_v1
  required_for_benchmark_claim: true
```

Use `validate_scenario_odd_contract_reference(...)` when a workflow needs to require that a
scenario contract's ODD reference resolves.

## Authoring Guidance

Keep v1 declarations narrow:

- set `claim_boundaries.evidence_status` to `metadata_only` until a consuming artifact has real
  execution evidence,
- include explicit `non_claims` for safety certification, legal compliance, and deployment
  readiness,
- keep `exclusions` concrete enough that report readers can distinguish covered assumptions from
  out-of-scope extrapolation,
- reference ODD contracts from scenario contracts or future BenchmarkClaim artifacts instead of
  duplicating all fields,
- update the schema version only for breaking shape or semantic changes.

Do not cite ODD metadata by itself as proof that a planner is safe, robust, socially acceptable, or
ready for real-world deployment.
