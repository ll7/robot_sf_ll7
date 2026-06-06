# Issue #2428 AMMV Mechanism Trace Panels

Date: 2026-06-06

Related issues: <https://github.com/ll7/robot_sf_ll7/issues/2428>,
<https://github.com/ll7/robot_sf_ll7/issues/2227>,
<https://github.com/ll7/robot_sf_ll7/issues/2405>

Status: diagnostic-only panel proof for the AMMV/default Social Force trace lane.

## Goal

Issue #2428 was split from the broader Issue #2227 mechanism-panel lane after Issue #2405 proved
that selected default/AMMV Social Force aggregate rows can be regenerated with step frames and exported as
loader-valid `simulation_trace_export.v1` files. The goal here is narrower than #2227: publish one
durable trace pair and panel bundle that demonstrates the AMMV path can feed the existing
trajectory-panel renderer.

## Result

The bundle at
`docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/` preserves:

- one `default_social_force` `classic_head_on_corridor_low` seed `111` trace export;
- one `ammv_social_force` trace export for the same scenario and seed;
- two PNG/PDF trajectory-panel figures rendered from those trace exports;
- captions, representative selection CSV, checksums, and a compact `summary.json`.

Both promoted traces loaded through `load_simulation_trace_export` with schema
`simulation_trace_export.v1`, 20 frames, scenario `classic_head_on_corridor_low`, and seed `111`.
The planner metadata includes the AMMV key in every frame, and the AMMV-aware trace preserves
`ammv.pedestrian_force_vectors` for review.

## Interpretation

This is useful diagnostic progress because it converts the Issue #2405 step-export unblocker into a
durable rendered artifact that later agents and reviewers can inspect without depending on
worktree-local `output/` files.

It is not benchmark-strength or paper-facing evidence. It covers one selected row per planner and
one AMMV/default Social Force panel target. It does not prove broad AMMV benefit, calibrated AMV
actuation, static-recenter attribution, or topology-guided recovery.

## Remaining #2227 Gap

Issue #2227 remains partially blocked:

- static-recenter panel target: still needs a selected baseline/intervention
  `simulation_trace_export.v1` pair;
- topology-guided recovery panel target: still needs a selected baseline/intervention
  `simulation_trace_export.v1` pair;
- AMMV/default Social Force target: now has a first durable diagnostic trace-panel bundle through
  Issue #2428.

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
scripts/validation/check_docs_proof_consistency.py --path docs/context/issue_2428_mechanism_trace_panels.md
git diff --check
```
