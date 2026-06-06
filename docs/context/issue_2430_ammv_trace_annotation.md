# Issue #2430 AMMV Trace Annotation Decision (2026-06-06)

Date: 2026-06-06

Related issues: <https://github.com/ll7/robot_sf_ll7/issues/2430>,
<https://github.com/ll7/robot_sf_ll7/issues/2159>,
<https://github.com/ll7/robot_sf_ll7/issues/2281>,
<https://github.com/ll7/robot_sf_ll7/issues/2428>

Status: diagnostic-only frame-level parity check; not behavioral-difference evidence.

## Goal

Issue #2430 asks whether the first durable AMMV/default Social Force trace pair from Issue #2428
can support a useful annotated Issue #2159 diagnostic case, or whether it should be classified as
insufficient for mechanism-difference review.

## Source Traces

- `docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/default_social_force_trace_export.json`
- `docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/ammv_social_force_trace_export.json`

Both traces load as `simulation_trace_export.v1` and preserve 20 frames for
`classic_head_on_corridor_low`, seed `111`. The recorded frame domain is steps `0..19` and
`time_s` `0.1..2.0`.

## Frame-Level Annotation

The traces are suitable for exact-frame annotation because each frame includes:

- `step` and `time_s`;
- robot `position`, `velocity`, and `heading`;
- pedestrian `id`, `position`, and `velocity`;
- planner `event`, `selected_action`, and `ammv.pedestrian_force_vectors`.

Useful anchors in this selected export window are:

| Anchor | Observation | Interpretation boundary |
| --- | --- | --- |
| Step `0`, `time_s=0.1` | Peak recorded total AMMV force-vector norm: `0.6419854506253375`; selected action is `linear_velocity=2.0`, `angular_velocity=0.2047055421214461`. | This is a telemetry anchor, not evidence that AMMV changed behavior. |
| Steps `16..17`, `time_s=1.7..1.8` | Selected angular velocity changes sign from `0.01631229185414007` to `-0.00537307638143103`. | The sign change is present in both traces, so it cannot explain a default-vs-AMMV difference. |
| Step `19`, `time_s=2.0` | Last promoted frame; total AMMV force-vector norm is `0.1106835289733311`; selected action is `linear_velocity=2.0`, `angular_velocity=-0.06093844961326944`. | End-of-window state only; no terminal outcome is included in the trace export. |

## Parity Finding

The selected `default_social_force` and `ammv_social_force` frame streams are numerically identical
over all recorded per-frame fields:

- `step`;
- `time_s`;
- `robot`;
- `pedestrians`;
- `planner.event`;
- `planner.selected_action`;
- `planner.ammv.pedestrian_force_vectors`.

A local comparison found `per_frame_max_abs_delta = 0.0`. The only observed differences are
top-level source metadata: `planner_id`, `episode_id`, and `generated_by`.

## Decision

This trace pair is sufficient as a telemetry-preservation and renderer-input proof for Issue #2159.
It is not sufficient as an AMMV behavioral-difference trace-review case.

The Issue #2159 lane should not spend more annotation effort on this exact selected row if the goal
is to explain a default-vs-AMMV mechanism difference. The next smallest useful AMMV step is to
select another row or seed with an observed planner-state, action, outcome, or force-pattern
difference, then annotate that row with the same exact-frame discipline.

## Missing Fields

Richer annotation would need at least one of:

- outcome or terminal-status metadata inside the trace export;
- reward, collision, or success metric fields inside the trace export;
- planner-force decomposition beyond aggregate per-pedestrian AMMV force vectors;
- command-source or arbitration timeline fields;
- a second selected AMMV row or seed with observed planner-state or action differences.

## Validation

```bash
python -m json.tool docs/context/evidence/issue_2430_ammv_trace_annotation_2026-06-06/summary.json
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
    assert len(trace.to_dict()["frames"]) == 20
PY
scripts/validation/check_docs_proof_consistency.py --path docs/context/issue_2430_ammv_trace_annotation.md
git diff --check
```
