# Issue #2428 Mechanism Trace Panels

Related issues: [#2428](https://github.com/ll7/robot_sf_ll7/issues/2428),
[#2227](https://github.com/ll7/robot_sf_ll7/issues/2227),
[#2405](https://github.com/ll7/robot_sf_ll7/issues/2405)

This bundle preserves the first durable diagnostic panel proof for the AMMV/default Social Force
mechanism trace lane. It contains one selected `classic_head_on_corridor_low` seed `111` trace
export for `default_social_force`, one matching trace export for `ammv_social_force`, and the
trajectory panels rendered from those traces.

Claim boundary: diagnostic-only. The files show that the AMMV single-row step-export path can feed
the existing trajectory-panel renderer and preserve `ammv.pedestrian_force_vectors` metadata in a
reviewable artifact bundle. They do not establish benchmark-strength, paper-facing, static-recenter,
or topology-guided mechanism evidence.

## Contents

- `summary.json`: compact result summary and validation record.
- `traces/default_social_force_trace_export.json`: loader-valid `simulation_trace_export.v1`
  export for the selected default Social Force row.
- `traces/ammv_social_force_trace_export.json`: loader-valid `simulation_trace_export.v1` export
  for the selected AMMV-aware Social Force row.
- `panels/representative_episode_selection.csv`: selected episode rows after promotion into this
  tracked bundle.
- `panels/trajectory_panel_manifest.json`: panel manifest with checksums for copied outputs and
  source trace exports.
- `panels/captions.md`: renderer-generated captions.
- `panels/trajectory_panels/*.png` and `*.pdf`: compact rendered trajectory-panel figures.

Raw benchmark JSONL and intermediate local output files were intentionally left out of git. The
bundle is reproducible from the tracked scenario/config inputs, seed `111`, the recorded commands
in the PR, and the generation commit named in the manifest.

## Result

Both promoted trace exports load with `simulation_trace_export.v1`, scenario
`classic_head_on_corridor_low`, seed `111`, and 20 frames. Both frame streams include planner
metadata with the `ammv` key; the AMMV-aware row preserves `ammv.pedestrian_force_vectors` for
trace-level review.

Issue #2227 is only partially advanced by this bundle:

- AMMV/default Social Force panel path: rendered and preserved.
- Static-recenter panel target: still missing a selected baseline/intervention trace pair.
- Topology-guided recovery panel target: still missing a selected baseline/intervention trace pair.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/summary.json
python -m json.tool docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/panels/trajectory_panel_manifest.json
scripts/dev/run_worktree_shared_venv.sh -- python - <<'PY'
from pathlib import Path
from robot_sf.analysis_workbench.simulation_trace_export import load_simulation_trace_export

for path in [
    Path("docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/default_social_force_trace_export.json"),
    Path("docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/ammv_social_force_trace_export.json"),
]:
    trace = load_simulation_trace_export(path)
    data = trace.to_dict()
    assert data["schema_version"] == "simulation_trace_export.v1"
    assert data["source"]["scenario_id"] == "classic_head_on_corridor_low"
    assert data["source"]["seed"] == 111
    assert len(data["frames"]) == 20
PY
```
