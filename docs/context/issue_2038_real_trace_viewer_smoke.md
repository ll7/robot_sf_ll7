# Issue #2038 Real Trace Viewer Smoke

Status: diagnostic validation, June 1, 2026.

Related issue: [#2038](https://github.com/ll7/robot_sf_ll7/issues/2038)

## Summary

The static Three.js trace viewer was validated on a freshly generated RobotEnv JSONL recording,
converted to `simulation_trace_export.v1`, rendered through `robot_sf.render.trace_viewer`, and
checked with the browser pixel smoke. The viewer rendered a nonblank scene from the generated
real-environment trace:

- browser smoke: passed
- screenshot classification: 339670 non-background pixels, 590 distinct colors
- evidence screenshot:
  [issue_2038_real_env_trace_viewer.png](evidence/issue_2038_real_env_trace_viewer.png)

The linked screenshot is a promoted copy of
`output/validation/issue_2038_real_env_trace_viewer.png` from the generated real-environment trace
run, copied into `docs/context/evidence/` so the compact visual proof survives worktree cleanup.
Promoted screenshot checksum:
`sha256:2f496bc4ea3986e4b2620ff695049300cd199988473c96ff44d2340e88cdee8b`.

This is diagnostic visualization evidence only. It does not make benchmark, planner-quality,
paper-facing, or performance claims.

## Commands Run

Generate a short JSONL recording through the real environment recording path:

```bash
uv run python - <<'PY'
from pathlib import Path

from robot_sf.gym_env.environment_factory import JsonlRecordingOptions, make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig

out = Path("output/issue_2038_real_recording")
out.mkdir(parents=True, exist_ok=True)
config = RobotSimulationConfig(map_id="uni_campus_big")
jsonl_opts = JsonlRecordingOptions(
    enabled=True,
    recording_dir=str(out),
    suite_name="issue_2038",
    scenario_name="real_env_smoke",
    algorithm_name="random_policy",
    recording_seed=2038,
)
env = make_robot_env(
    config=config,
    seed=2038,
    jsonl_recording_options=jsonl_opts,
    recording_enabled=True,
    debug=False,
)
try:
    env.action_space.seed(2038)
    obs, info = env.reset(seed=2038)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
finally:
    env.close()
print("\n".join(str(path) for path in sorted(out.glob("*.jsonl"))))
PY
```

Convert the generated JSONL recording into the analysis-workbench trace schema:

```bash
uv run python scripts/tools/build_simulation_trace_export.py \
  --source output/issue_2038_real_recording/issue_2038_real_env_smoke_random_policy_2038_ep0000.jsonl \
  --output output/issue_2038_real_trace/issue_2038_real_env_smoke_trace.json \
  --planner-id random_policy \
  --scenario-id real_env_smoke
```

Render the static Three.js trace viewer:

```bash
uv run python -m robot_sf.render.trace_viewer \
  output/issue_2038_real_trace/issue_2038_real_env_smoke_trace.json \
  --output-dir output/issue_2038_real_env_trace_viewer
```

Run the browser pixel smoke:

```bash
uv run python scripts/validation/smoke_threejs_viewer_browser.py \
  --viewer-dir output/issue_2038_real_env_trace_viewer \
  --screenshot output/validation/issue_2038_real_env_trace_viewer.png \
  --timeout-ms 15000 \
  --min-non-background-pixels 100
```

Observed result:

```text
Three.js browser smoke passed: 339670 non-background pixels, 590 distinct colors.
```

Promote the compact screenshot used by this note:

```bash
cp output/validation/issue_2038_real_env_trace_viewer.png \
  docs/context/evidence/issue_2038_real_env_trace_viewer.png
sha256sum docs/context/evidence/issue_2038_real_env_trace_viewer.png
```

## Fixture Cross-Check

The tracked `planner_sanity_open_episode_0000.json` trace still works with the annotated fixture
path:

```bash
uv run python examples/advanced/34_trace_threejs_viewer.py \
  tests/fixtures/analysis_workbench/simulation_trace_export_v1/planner_sanity_open_episode_0000.json \
  --annotations tests/fixtures/analysis_workbench/trace_annotation_set_v1/issue_1962_planner_sanity_open_annotations.json \
  --output-dir output/issue_2038_real_trace_viewer
```

Its browser smoke also passed:

```text
Three.js browser smoke passed: 153080 non-background pixels, 322 distinct colors.
```

That trace is fixture-derived (`trace_fixture_gen`) and should be treated as a contract check, not
the main #2038 real-environment proof.

## Observations

- The real-environment trace exercised a nontrivial RobotEnv recording with many pedestrian
  positions in `simulation_trace_export.v1`.
- The generated viewer rendered a visible scene in headless Chromium and wrote a reviewable
  screenshot.
- The scene still uses trace-viewer auto bounds with no SVG map geometry, obstacles, or zones; this
  reflects the current viewer/export path used for this smoke, not evidence that the source
  environment lacks those concepts.
- The raw JSONL recording, converted trace, viewer directory, and validation screenshot under
  `output/` are disposable local artifacts. The compact screenshot above is the durable review
  evidence.

## Validation

```bash
uv run pytest tests/render/test_trace_viewer.py tests/validation/test_smoke_threejs_viewer_browser.py -q
```

Result: 21 passed.

## Evidence Boundary

This validation proves that the current static viewer path can load, render, and pass a browser
pixel smoke for one generated real-environment `simulation_trace_export.v1` trace. It does not
prove metric correctness, benchmark success, planner quality, map-geometry parity, or paper-facing
readiness.
