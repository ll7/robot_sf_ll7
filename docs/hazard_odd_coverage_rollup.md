# Hazard And ODD Coverage Rollup

`scripts/tools/hazard_odd_coverage_rollup.py` summarizes which hazard traceability and ODD
contract categories are represented in a benchmark campaign bundle. It is a coverage and provenance
audit, not safety proof and not a planner-success metric.

Example:

```bash
uv run python scripts/tools/hazard_odd_coverage_rollup.py \
  --campaign-root docs/context/evidence/issue_1023_scenario_horizons_local_full_2026-05-06 \
  --output output/hazard_odd_coverage/issue_1023 \
  --hazard-traceability configs/benchmarks/hazard_traceability/low_speed_public_space_v1.yaml \
  --odd-contract configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml \
  --scenario-contract configs/scenarios/contracts/station_platform_candidate_pack_issue736_contracts.yaml
```

Outputs:

- `hazard_odd_coverage_summary.json`
- `hazard_odd_coverage_summary.md`
- `hazard_coverage_table.csv`
- `odd_boundary_table.csv`
- `scenario_contract_table.csv`
- `coverage_status_summary.png`
- `checksums.sha256`

Status meanings:

- `covered`: at least one non-caveated executed campaign row represents the category.
- `partial`: campaign rows exist, but fallback, degraded, failed, or not-available caveats remain.
- `missing`: metadata maps the category, but no executed row represented it.
- `excluded`: ODD metadata or fail-closed scenario certification explicitly excludes the category.
- `unavailable`: an optional contract, mapping, coverage report, or campaign table was absent.

The JSON summary keeps `metadata_inputs` separate from `executed_evidence`. Scenario contracts, ODD
contracts, hazard traceability, and stress/uncertainty coverage metadata may explain intended
coverage, but they do not become benchmark evidence unless matched to executed campaign rows.
