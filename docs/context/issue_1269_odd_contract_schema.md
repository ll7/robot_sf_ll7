# Issue #1269 ODD Contract Schema

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1269>

## Summary

Issue #1269 adds `odd_contract.v1`, a small Operational Design Domain metadata contract for
benchmark and falsification evidence. The contract records operating assumptions and non-claim
boundaries without changing benchmark execution.

## Implemented Surfaces

- `robot_sf/benchmark/schemas/odd_contract.v1.json` defines the JSON Schema.
- `robot_sf/benchmark/odd_contract.py` provides typed loading, JSON Schema validation, semantic
  validation, and reference validation.
- `configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml` is the first tracked ODD
  declaration for low-speed public-space micromobility assumptions.
- `scenario_contract.v1` now supports an optional `odd_contract_ref` block.
- `configs/scenarios/contracts/station_platform_candidate_pack_issue736_contracts.yaml` references
  the tracked ODD declaration as an example.
- `docs/odd_contracts.md` documents the boundary and authoring guidance.

## Boundary

ODD metadata bounds evidence. It does not certify safety, establish legal compliance, promote a
scenario into a paper matrix, or replace `scenario_cert.v1` and benchmark run artifacts.

Scenario contracts and future BenchmarkClaim artifacts should reference ODD declarations by
`source` and `contract_id` instead of duplicating all fields. This keeps claim assumptions
auditable while preserving a single reviewable ODD source.

## Validation

Targeted validation for this issue should include:

```bash
uv run pytest tests/benchmark/test_odd_contract.py tests/benchmark/test_scenario_contract.py -q
uv run ruff check robot_sf/benchmark/odd_contract.py robot_sf/benchmark/scenario_contract.py tests/benchmark/test_odd_contract.py tests/benchmark/test_scenario_contract.py
uv run ruff format --check robot_sf/benchmark/odd_contract.py robot_sf/benchmark/scenario_contract.py tests/benchmark/test_odd_contract.py tests/benchmark/test_scenario_contract.py
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```

Run the repository PR readiness gate after syncing with latest `origin/main` before PR handoff.
