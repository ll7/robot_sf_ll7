# Issue #2405 AMV Step-Export Decision

Date: 2026-06-06

Issue: [#2405](https://github.com/ll7/robot_sf_ll7/issues/2405)
Predecessor: [#2392](https://github.com/ll7/robot_sf_ll7/issues/2392)
Parent context: [#2227](https://github.com/ll7/robot_sf_ll7/issues/2227),
[#2159](https://github.com/ll7/robot_sf_ll7/issues/2159)
Evidence:
[summary.json](evidence/issue_2405_amv_step_export_2026-06-06/summary.json)

## Scope

Issue #2405 checks whether the AMV trace-review blocker from Issue #2392 is still active after PR
2396 added `--record-simulation-step-trace`. The result is a small implementation fix plus a
diagnostic proof: the map-batch worker now forwards the trace flag into each episode, and aggregate
trace frames can be converted into `simulation_trace_export.v1` after normalizing pedestrian IDs.

This is diagnostic trace-export evidence only. It does not promote raw local `output/` JSONL,
publish trace artifacts, or upgrade AMV/AMMV findings to benchmark-strength evidence.

## Decision

The old "no step frames" blocker is resolved for a selected one-row Issue #2168 Social Force slice:

| Input | Result | Interpretation |
| --- | --- | --- |
| Default Social Force with `--record-simulation-step-trace` | 3 aggregate rows, each with 20 step frames | End-to-end CLI now preserves step frames after the worker pass-through fix. |
| AMMV-aware Social Force with `--record-simulation-step-trace` | 3 aggregate rows, each with 20 step frames | Matched AMMV row also preserves step frames. |
| First default row converted with `build_simulation_trace_export.py` | `simulation_trace_export.v1`, 20 frames, loader-valid | One renderable baseline trace can be regenerated from the command path. |
| First AMMV row converted with `build_simulation_trace_export.py` | `simulation_trace_export.v1`, 20 frames, loader-valid | One renderable intervention trace can be regenerated from the command path. |

The proof is intentionally one selected row per side. The converter still treats a multi-row JSONL
as one timeline, so concatenating all three episode rows remains out of scope for this issue. Use a
single selected aggregate row, or add an explicit multi-episode export design in a separate issue.

## Mechanism Boundary

Issue #2227 is unblocked for a narrow AMMV renderable-trace smoke: the selected default/AMMV pair can
now be regenerated and converted into loader-valid `simulation_trace_export.v1` inputs. The trace
planner metadata includes `ammv.pedestrian_force_vectors` on all sampled frames.

Do not treat this as full mechanism-panel completion. The raw trace exports remain local and
ignored, only a compact proof summary is tracked here, and broader AMV panel work still needs a
durable trace-pair selection/annotation decision before any paper-facing claim.

## Validation

Executed from the Issue #2405 worktree:

```bash
scripts/dev/run_worktree_shared_venv.sh -- pytest tests/analysis_workbench/test_simulation_trace_export.py::test_build_trace_export_uses_aggregate_simulation_step_trace tests/benchmark/test_map_runner_utils.py::test_run_map_job_worker_forwards_metadata_params -q
scripts/dev/run_worktree_shared_venv.sh -- uv run robot_sf_bench validate-config --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml
scripts/dev/run_worktree_shared_venv.sh -- uv run robot_sf_bench preview-scenarios --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 scripts/dev/run_worktree_shared_venv.sh -- uv run robot_sf_bench run --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml --out output/issue_2405_after_fix/default_social_force_trace.jsonl --base-seed 111 --repeats 1 --horizon 20 --dt 0.1 --record-forces --record-simulation-step-trace --no-video --video-renderer none --algo social_force --workers 1 --no-resume --structured-output json
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 scripts/dev/run_worktree_shared_venv.sh -- uv run robot_sf_bench run --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml --out output/issue_2405_after_fix/ammv_social_force_trace.jsonl --base-seed 111 --repeats 1 --horizon 20 --dt 0.1 --record-forces --record-simulation-step-trace --no-video --video-renderer none --algo social_force --algo-config configs/baselines/social_force_ammv_aware.yaml --workers 1 --no-resume --structured-output json
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/build_simulation_trace_export.py --source output/issue_2405_after_fix/default_social_force_first.jsonl --output output/issue_2405_after_fix/default_social_force_trace_export.json --planner-id default_social_force --scenario-id classic_head_on_corridor_low
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/build_simulation_trace_export.py --source output/issue_2405_after_fix/ammv_social_force_first.jsonl --output output/issue_2405_after_fix/ammv_social_force_trace_export.json --planner-id ammv_social_force --scenario-id classic_head_on_corridor_low
scripts/dev/run_worktree_shared_venv.sh -- python - <<'PY'
from pathlib import Path
from robot_sf.analysis_workbench.simulation_trace_export import load_simulation_trace_export
for path in [Path("output/issue_2405_after_fix/default_social_force_trace_export.json"), Path("output/issue_2405_after_fix/ammv_social_force_trace_export.json")]:
    trace = load_simulation_trace_export(path)
    assert len(trace.to_dict()["frames"]) == 20
PY
```

Result: targeted tests passed, both benchmark runs wrote 3 rows with 20 step frames per row, and the
selected default/AMMV first-row exports loaded as `simulation_trace_export.v1`.

## Follow-Up Boundary

Future work should either select and publish a durable compact trace-pair/annotation package for the
AMMV mechanism panel, or design a multi-episode `simulation_trace_export.v1` bundle contract. Do not
reopen the old no-step-frame blocker unless the `--record-simulation-step-trace` CLI path regresses.
