<!-- AI-GENERATED (ll7/diss#2994, 2026-10-11) - NEEDS-REVIEW -->
# Issue #2994 Chapter 7 0.0.8 Worked-Example Traces

This handoff records the Chapter 7 worked-example trace reselection for the
published robot_sf_ll7 0.0.8 release.

## Provenance

- Published source commit: `373dbfde4f39667cf9e8732dabe7df5118bdeab1`
- Published rows: Zenodo DOI `10.5281/zenodo.23150472`
- Scenario config: `configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml`
- Slurm run host/path: `imech192:/home/luttkule/lanes/traces008/run`
- Slurm job: `31326`
- Development seed policy: `1001..1200` only; no sealed publication seeds were re-run.
- Preserved hub bundle:
  `imech192:/home/luttkule/external_data_hub/benchmark-results/robot_sf_ll7/issue_2994_ch7_trace_worked_examples_0_0_8_373dbfde4f_20261011`

## Selection Table

| figure cell | scenario | planner | published 0.0.8 outcomes | selected dev outcomes | match |
| --- | --- | --- | --- | --- | --- |
| head_on_orca_vs_socialforce | classic_head_on_corridor_medium | orca | success=29/30, collision=1/30 | success=29/30, collision=1/30 | exact |
| head_on_orca_vs_socialforce | classic_head_on_corridor_medium | social_force | success=28/30, timeout=2/30 | success=28/30, timeout=2/30 | exact |
| group_crossing_goal_vs_socialforce | classic_group_crossing_medium | goal | success=24/30, collision=6/30 | success=24/30, collision=6/30 | exact |
| group_crossing_goal_vs_socialforce | classic_group_crossing_medium | social_force | success=30/30 | success=30/30 | exact |
| speed_clearance_phase_portrait | classic_urban_crossing_medium | orca | success=29/30, collision=1/30 | success=29/30, collision=1/30 | exact |
| speed_clearance_phase_portrait | classic_urban_crossing_medium | social_force | success=23/30, timeout=7/30 | success=23/30, timeout=7/30 | exact |
| ego_occupancy | classic_doorway_medium | goal | success=13/30, collision=17/30 | success=13/30, collision=17/30 | exact |
| ego_occupancy | classic_doorway_medium | social_force | success=29/30, timeout=1/30 | success=29/30, timeout=1/30 | exact |
| dual_panel_interaction | francis2023_robot_crowding | orca | success=15/30, collision=14/30, timeout=1/30 | success=15/30, collision=14/30, timeout=1/30 | exact |
| dual_panel_interaction | francis2023_robot_crowding | social_force | collision=1/30, timeout=29/30 | collision=1/30, timeout=29/30 | exact |

Important interpretation change: the head-on corridor no longer supports the
old "social force stalls just short of the goal zone" contrast. In the published
0.0.8 rows, social force succeeds in most runs for this cell, so Figure 7.6 must
change the contrast rather than preserve the 0.0.7 story.

## Candidate Notes

- `classic_head_on_corridor_medium` is retained for continuity, with the changed
  interpretation above.
- `classic_group_crossing_medium` gives a clean goal-vs-social-force contrast:
  the goal baseline retains collisions while social force completes.
- `classic_urban_crossing_medium` is a stronger speed-clearance phase portrait
  candidate because ORCA is mostly successful and social force has multiple
  timeouts.
- `classic_doorway_medium` is a compact occupancy candidate with frequent goal
  collisions and mostly successful social-force runs.
- `francis2023_robot_crowding` is the hardest dual-panel interaction candidate:
  ORCA is mixed and social force mostly times out.

## Bundle Contents

The preserved hub bundle contains:

- `manifest.json`
- `selection_table.csv`
- `candidate_cells.csv`
- `trace_sha256_16.csv` with all 300 per-trace `sha256_16` values
- `proposed_evidence_pins.yaml` with the complete replacement YAML entry
- `SHA256SUMS`
- 300 trace directories, each with `metadata.json`, `trace_series.json`,
  `trace_timeseries.csv`, `min_distance_series.csv`, `README.md`, and local
  `SHA256SUMS`

## Validation Receipts

- Slurm campaign completed with 3/3 planner arms successful and 3000 total
  development episodes.
- Export selected 300 traces: five figure cells x two planners x 30 shared
  development seeds per cell.
- Local bundle checksum verification passed with `sha256sum -c SHA256SUMS`.
- Hub readback verification passed with `sha256sum -c SHA256SUMS`.
- Focused code checks:
  - `nice -n 15 scripts/dev/run_worktree_shared_venv.sh -- ruff format --check scripts/analysis/build_issue_2994_ch7_traces_008.py tests/analysis/test_build_issue_2994_ch7_traces_008.py`
  - `nice -n 15 scripts/dev/run_worktree_shared_venv.sh -- ruff check scripts/analysis/build_issue_2994_ch7_traces_008.py tests/analysis/test_build_issue_2994_ch7_traces_008.py`
  - `nice -n 15 scripts/dev/run_worktree_shared_venv.sh -- pytest tests/analysis/test_build_issue_2994_ch7_traces_008.py -q`

## Proposed Evidence Pins

The complete proposed entry is preserved in the hub bundle at
`proposed_evidence_pins.yaml`. A compact index suitable for dissertation-side
integration is:

```yaml
chapter7_worked_examples_0_0_8:
  bundle: issue_2994_ch7_trace_worked_examples_0_0_8_373dbfde4f_20261011
  source_commit: 373dbfde4f39667cf9e8732dabe7df5118bdeab1
  doi: 10.5281/zenodo.23150472
  claim_boundary: visualization-only development-seed trace reruns; published 0.0.8 rows remain the statistical source
  manifest: manifest.json
  selection_table: selection_table.csv
  trace_sha256_16: trace_sha256_16.csv
  figures:
    headon_orca_vs_socialforce:
      scenario_id: classic_head_on_corridor_medium
      traces_from: trace_sha256_16.csv
    group_crossing_goal_vs_socialforce:
      scenario_id: classic_group_crossing_medium
      traces_from: trace_sha256_16.csv
    speed_clearance_phase_portrait:
      scenario_id: classic_urban_crossing_medium
      traces_from: trace_sha256_16.csv
    ego_occupancy:
      scenario_id: classic_doorway_medium
      traces_from: trace_sha256_16.csv
    dual_panel_interaction:
      scenario_id: francis2023_robot_crowding
      traces_from: trace_sha256_16.csv
```
