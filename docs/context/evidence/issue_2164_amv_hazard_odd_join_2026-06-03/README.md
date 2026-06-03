# Issue #2164 AMV Hazard/ODD Metadata Join

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2164>

This pack joins the tracked issue #1484 cross-kinematics AMV execution rows to
issue-specific hazard traceability and scenario-contract metadata for
`classic_cross_trap_low`.

## Generation Command

```bash
uv run python scripts/tools/hazard_odd_coverage_rollup.py \
  --campaign-root docs/context/evidence/issue_1484_broader_cross_kinematics_2026-05-28 \
  --output docs/context/evidence/issue_2164_amv_hazard_odd_join_2026-06-03 \
  --report-id issue_2164_amv_hazard_odd_join \
  --hazard-traceability configs/benchmarks/hazard_traceability/issue_2164_amv_cross_trap_v1.yaml \
  --odd-contract configs/benchmarks/odd_contracts/low_speed_public_space_v1.yaml \
  --scenario-contract configs/scenarios/contracts/issue_2164_amv_cross_trap_contracts.yaml \
  --stress-uncertainty-coverage docs/context/evidence/issue_1484_broader_cross_kinematics_2026-05-28/reports/amv_coverage_summary.json
```

## Result

- Executed row-level records read: 42.
- Hazard statuses: `covered=3`.
- ODD boundary statuses: `covered=2`, `excluded=8`.
- Scenario contract statuses: `covered=1`.
- Joined scenario: `classic_cross_trap_low`.
- Covered hazards: `robot_pedestrian_collision`, `near_miss`,
  `pedestrian_flow_disruption`.

## Interpretation

This closes the #2156 metadata-join gap for the selected
`classic_cross_trap_low` AMV slice. It does not claim full research-v1 matrix
coverage, CARLA transfer, safety certification, or paper-ready evidence. The
claim-map status may move to `candidate` only for this selected scenario slice;
broader AMV scenario criticality still requires additional scenario contracts,
hazard mappings, row-status accounting, and synthesis.

## Files

- `hazard_odd_coverage_summary.json`: machine-readable summary and provenance.
- `hazard_odd_coverage_summary.md`: compact human-readable summary.
- `hazard_coverage_table.csv`: hazard status rows.
- `odd_boundary_table.csv`: ODD supported/excluded claim rows.
- `scenario_contract_table.csv`: scenario-contract status rows.
- `coverage_status_summary.png`: compact status count figure.
- `SHA256SUMS`: checksums for promoted compact artifacts.
