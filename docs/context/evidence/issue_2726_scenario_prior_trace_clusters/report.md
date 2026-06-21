# Calibrate Scenario Priors from Simulation Trace Clusters

## Claim Boundary & Evidence Status

- **Claim Boundary**: `repository_trace_grounded_not_real_world_calibrated: this report and generated prior cards are derived entirely from deterministic simulation trace clusters. They do not claim real-world validity, representativeness, or generalizability to real-world pedestrian behavior. Refer to issues #3161 and #2918 for real-world staging and calibration requirements.`
- **Status**: `repository_trace_derived_proposal`
- **Context References**: Refer to issue #3161 for real-world staging contracts and #2918 for calibration context.
- **Axioms**: This analysis runs strictly on repository trace fixtures. No external dataset or real-world representativeness is claimed.

## Summary

- **Total simulation traces processed**: 20
- **Total unique clusters identified**: 9

## Cluster Table

| Cluster ID | Scenarios | Density | Signal | Outcome | Traces | Mean Min Dist (m) | Mean Stop Frac |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `bottleneck_sparse_unsignalized_nominal` | bottleneck | sparse | unsignalized | nominal | 2 | 1.11 | 0.50 |
| `crossing_sparse_signalized_nominal` | crossing | sparse | signalized | nominal | 4 | 1.17 | 1.00 |
| `crossing_sparse_unsignalized_collision` | crossing | sparse | unsignalized | collision | 1 | 0.25 | 1.00 |
| `general_dense_unsignalized_collision` | general | dense | unsignalized | collision | 1 | 0.16 | 0.00 |
| `general_sparse_signalized_nominal` | general | sparse | signalized | nominal | 2 | 1.97 | 1.00 |
| `general_sparse_unsignalized_near-miss` | general | sparse | unsignalized | near-miss | 1 | 0.47 | 1.00 |
| `general_sparse_unsignalized_nominal` | general | sparse | unsignalized | nominal | 2 | 1.46 | 0.67 |
| `occluded_sparse_unsignalized_near-miss` | occluded | sparse | unsignalized | near-miss | 6 | 0.53 | 0.05 |
| `occluded_sparse_unsignalized_nominal` | occluded | sparse | unsignalized | nominal | 1 | 0.98 | 0.00 |

## Cluster Lineage Details

For each cluster, the lineage pointers to source trace file paths, trace IDs, and episode IDs are detailed below.

### Cluster: `bottleneck_sparse_unsignalized_nominal`

- **Traces count**: 2
- **Source Trace Pointers**:
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/bottleneck_motion_rich_fixture.json` (trace_id: `issue_2937_bottleneck_motion_rich`, episode_id: `issue_2937_bottleneck_motion_rich_ep0000`)
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json` (trace_id: `fixture_trace_001`, episode_id: `fixture_episode_001`)

### Cluster: `crossing_sparse_signalized_nominal`

- **Traces count**: 4
- **Source Trace Pointers**:
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2868_goal_directed_crossing_fixture_0000.json` (trace_id: `issue_2868_goal_directed_crossing_0000`, episode_id: `issue_2868_goal_directed_crossing_episode_0000`)
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2868_signalized_crossing_fixture_0000.json` (trace_id: `issue_2868_signalized_crossing_0000`, episode_id: `issue_2868_signalized_crossing_episode_0000`)
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/issue_2868_goal_directed_crossing_fixture_extended.json` (trace_id: `issue_2868_goal_directed_crossing_0000_issue_2937_extended`, episode_id: `issue_2868_goal_directed_crossing_episode_0000`)
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/issue_2868_signalized_crossing_fixture_extended.json` (trace_id: `issue_2868_signalized_crossing_0000_issue_2937_extended`, episode_id: `issue_2868_signalized_crossing_episode_0000`)

### Cluster: `crossing_sparse_unsignalized_collision`

- **Traces count**: 1
- **Source Trace Pointers**:
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/crossing_proxy_motion_rich_fixture.json` (trace_id: `issue_2937_crossing_proxy_motion_rich`, episode_id: `issue_2937_crossing_proxy_motion_rich_ep0000`)

### Cluster: `general_dense_unsignalized_collision`

- **Traces count**: 1
- **Source Trace Pointers**:
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/dense_pedestrian_stress_episode_0000.json` (trace_id: `issue_2765_dense_pedestrian_stress_seed2765_ep0000`, episode_id: `dense_pedestrian_stress_ep0000`)

### Cluster: `general_sparse_signalized_nominal`

- **Traces count**: 2
- **Source Trace Pointers**:
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2868_waiting_intent_change_fixture_0000.json` (trace_id: `issue_2868_waiting_intent_change_0000`, episode_id: `issue_2868_waiting_intent_change_episode_0000`)
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/issue_2868_waiting_intent_change_fixture_extended.json` (trace_id: `issue_2868_waiting_intent_change_0000_issue_2937_extended`, episode_id: `issue_2868_waiting_intent_change_episode_0000`)

### Cluster: `general_sparse_unsignalized_near-miss`

- **Traces count**: 1
- **Source Trace Pointers**:
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/issue_2868_route_conflict_goal_fixture_extended.json` (trace_id: `issue_2868_route_conflict_goal_0000_issue_2937_extended`, episode_id: `issue_2868_route_conflict_goal_episode_0000`)

### Cluster: `general_sparse_unsignalized_nominal`

- **Traces count**: 2
- **Source Trace Pointers**:
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2868_route_conflict_goal_fixture_0000.json` (trace_id: `issue_2868_route_conflict_goal_0000`, episode_id: `issue_2868_route_conflict_goal_episode_0000`)
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/planner_sanity_open_episode_0000.json` (trace_id: `planner_sanity_open-ep0-seed7-source-aae87cf6`, episode_id: `0`)

### Cluster: `occluded_sparse_unsignalized_near-miss`

- **Traces count**: 6
- **Source Trace Pointers**:
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/occluded_emergence_episode_extended.json` (trace_id: `issue_2756_occluded_emergence_seed111_ep0000_issue_2937_extended`, episode_id: `issue_2756_occluded_emergence_ep0000`)
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/occluded_emergence_episode_0000.json` (trace_id: `issue_2756_occluded_emergence_seed111_ep0000`, episode_id: `issue_2756_occluded_emergence_ep0000`)
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/occluded_emergence_fast_pedestrian_episode_0000.json` (trace_id: `issue_2780_occluded_emergence_fast_pedestrian_seed111_ep0000`, episode_id: `issue_2780_occluded_emergence_fast_pedestrian_ep0000`)
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/occluded_emergence_late_visibility_episode_0000.json` (trace_id: `issue_2780_occluded_emergence_late_visibility_seed111_ep0000`, episode_id: `issue_2780_occluded_emergence_late_visibility_ep0000`)
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/occluded_emergence_left_close_episode_0000.json` (trace_id: `issue_2780_occluded_emergence_left_close_seed111_ep0000`, episode_id: `issue_2780_occluded_emergence_left_close_ep0000`)
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/occluded_emergence_slow_pedestrian_episode_0000.json` (trace_id: `issue_2780_occluded_emergence_slow_pedestrian_seed111_ep0000`, episode_id: `issue_2780_occluded_emergence_slow_pedestrian_ep0000`)

### Cluster: `occluded_sparse_unsignalized_nominal`

- **Traces count**: 1
- **Source Trace Pointers**:
  - `tests/fixtures/analysis_workbench/simulation_trace_export_v1/occluded_emergence_right_close_episode_0000.json` (trace_id: `issue_2780_occluded_emergence_right_close_seed111_ep0000`, episode_id: `issue_2780_occluded_emergence_right_close_ep0000`)

## Limitations

- These priors are repository-trace-grounded proposals, not real-world calibrated priors.
- The input set is limited to committed simulation trace fixtures and may overrepresent synthetic edge cases.
- Cluster assignment is deterministic and rule-based; it is intended for provenance and reviewability rather than statistical optimality.

## Follow-Up Data Requirements

- Stage real-world trajectory data under issue #3161 before making representativeness claims.
- Use issue #2918 calibration context before promoting these proposal cards into calibrated priors.
