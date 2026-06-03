# Issue #2156 Research-V1 Hazard/ODD Coverage Diagnostic

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2156>

This evidence pack runs the existing hazard/ODD coverage rollup on the tracked
cross-kinematics AMV evidence bundle from
`docs/context/evidence/issue_1484_broader_cross_kinematics_2026-05-28`.

Generation command:

```bash
uv run python scripts/tools/hazard_odd_coverage_rollup.py \
  --campaign-root docs/context/evidence/issue_1484_broader_cross_kinematics_2026-05-28 \
  --output docs/context/evidence/issue_2156_research_v1_hazard_odd_2026-06-03 \
  --report-id issue_2156_research_v1_hazard_odd_cross_kinematics \
  --hazard-traceability configs/benchmarks/hazard_traceability/low_speed_public_space_v1.yaml \
  --odd-contract configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml \
  --scenario-contract configs/scenarios/contracts/station_platform_candidate_pack_issue736_contracts.yaml \
  --stress-uncertainty-coverage docs/context/evidence/issue_1484_broader_cross_kinematics_2026-05-28/reports/amv_coverage_summary.json
```

Source commit at generation: `0b5a2efcdd738f5019e75a063218a99c9772d589`.

## Result

- Executed row-level records read: 21.
- Hazard statuses: `missing=5`.
- ODD boundary statuses: `partial=2`, `excluded=8`.
- Scenario contract statuses: `missing=1`.
- Stress/uncertainty input status: `unavailable` because the AMV coverage summary
  uses a different schema than the rollup expects.

## Interpretation

This is diagnostic traceability evidence, not benchmark-strength coverage proof.
The selected AMV evidence bundle has executed rows, but the current
hazard/ODD/scenario-contract metadata does not join those rows to the intended
hazard categories. The smallest next proof step is to add or select scenario
contracts and hazard mappings that name the executed AMV scenarios, then rerun
the same rollup.

## Files

- `hazard_odd_coverage_summary.json`: machine-readable summary and provenance.
- `hazard_odd_coverage_summary.md`: compact human-readable summary.
- `hazard_coverage_table.csv`: hazard status rows.
- `odd_boundary_table.csv`: ODD supported/excluded claim rows.
- `scenario_contract_table.csv`: scenario-contract status rows.
- `coverage_status_summary.png`: compact status count figure.
- `checksums.sha256`: checksums for generated artifacts.
