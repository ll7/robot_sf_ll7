# Hazard Traceability

[← Back to Documentation Index](./README.md)

`hazard_traceability.v1` maps benchmark scenario IDs or scenario families to the hazards they are
intended to exercise and to the metrics or evidence fields that can support claims about those
hazards.

This is traceability metadata only. It is not a safety case, legal certification, success metric,
or proof that a planner mitigates a hazard.

## Contract Boundary

Use the related surfaces for different jobs:

- `hazard_traceability.v1`: intended links from scenarios to hazards and supporting evidence fields.
- `scenario_contract.v1`: authored scenario intent and scenario-specific assumptions.
- `scenario_cert.v1`: fail-closed feasibility and benchmark eligibility classification.
- benchmark episode records and reports: executed planner behavior, metrics, seeds, and outcomes.

Traceability helps report coverage gaps and caveats. It does not replace benchmark execution.

## Public Files

- Schema:
  [`robot_sf/benchmark/schemas/hazard_traceability.v1.json`](../robot_sf/benchmark/schemas/hazard_traceability.v1.json)
- Loader:
  [`robot_sf/benchmark/hazard_traceability.py`](../robot_sf/benchmark/hazard_traceability.py)
- Fixture:
  [`configs/benchmarks/hazard_traceability/low_speed_public_space_v1.yaml`](../configs/benchmarks/hazard_traceability/low_speed_public_space_v1.yaml)

The first fixture covers low-speed public-space hazards such as collision, near miss, blind-corner
emergence, keep-clear violation, and pedestrian-flow disruption.

## Library API

```python
from pathlib import Path

from robot_sf.benchmark.hazard_traceability import (
    load_hazard_traceability,
    summarize_hazard_coverage,
)

mapping = load_hazard_traceability(
    Path("configs/benchmarks/hazard_traceability/low_speed_public_space_v1.yaml")
)
summary = summarize_hazard_coverage(
    mapping,
    scenario_families=["station_platform"],
)
```

`load_hazard_traceability(...)` validates JSON Schema first and raises
`HazardTraceabilityValidationError` with JSON-pointer-style paths for invalid sections.
`summarize_hazard_coverage(...)` emits a compact `hazard-traceability-coverage.v1` dictionary with
covered hazards, unmapped scenario IDs/families, and supporting metrics/evidence fields.

## Authoring Guidance

Keep v1 mappings conservative:

- use stable hazard IDs rather than prose-only labels,
- link every hazard to metrics or evidence fields that can actually appear in benchmark artifacts,
- report unsupported or unmapped scenario inputs as coverage gaps,
- state the claim boundary in the mapping,
- avoid using a hazard mapping as proof that a planner solved the hazard.

Future BenchmarkClaim artifacts can consume hazard summaries to state coverage caveats, but they
still need scenario certification and benchmark run evidence.
