# Issue 632 Python-RVO2 Prototype Note

Date: 2026-03-19
Related issues:
- `robot_sf_ll7#632` Python-RVO2 benchmark integration prototype
- `robot_sf_ll7#629` planner-zoo research intake

## Goal

Validate `Python-RVO2` as an upstream-backed ORCA benchmark prototype without hiding the kinematics
translation behind generic benchmark plumbing.

## Upstream reference

- upstream repo: <https://github.com/mit-acl/Python-RVO2>
- vendored path: `third_party/python-rvo2`
- vendored commit: `56b245132ea104ee8a621ddf65b8a3dd85028ed2`
- local vendoring note: `third_party/python-rvo2/UPSTREAM.md`

## Adapter boundary

The Robot SF ORCA path does not reimplement reciprocal-avoidance selection from scratch.

Boundary:

- upstream `Python-RVO2` selects a collision-avoiding velocity in world coordinates
- `robot_sf/planner/socnav.py` converts that selected velocity into Robot SF
  `unicycle_vw` commands through the explicit projection step in
  `ORCAPlannerAdapter._velocity_to_command(...)`

This is the benchmark-visible contract:

- upstream command space: `velocity_vector_xy`
- benchmark command space: `unicycle_vw`
- projection policy: `heading_safe_velocity_to_unicycle_vw`

## Canonical validation commands

Validate the vendored upstream example:

```bash
uv run python third_party/python-rvo2/example.py
```

Run the dedicated integration probe:

```bash
uv run python scripts/tools/probe_python_rvo2_integration.py \
  --rvo2-root third_party/python-rvo2 \
  --output-json output/benchmarks/external/python_rvo2_probe/report.json \
  --output-md output/benchmarks/external/python_rvo2_probe/report.md
```

Run one representative Robot SF benchmark scenario:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/python_rvo2_orca_probe.yaml \
  --label issue632_python_rvo2 \
  --log-level WARNING
```

## Current result

Observed probe verdict:

- `output/benchmarks/external/python_rvo2_probe/report.md`
- verdict: `viable benchmark prototype`

Observed benchmark run:

- campaign root:
  `output/benchmarks/camera_ready/python_rvo2_orca_probe_issue632_python_rvo2_20260319_202233`
- report:
  `output/benchmarks/camera_ready/python_rvo2_orca_probe_issue632_python_rvo2_20260319_202233/reports/campaign_report.md`

Key benchmark result:

- planner: `orca`
- episodes: `3`
- success: `1.0000`
- collisions: `0.0000`
- execution mode: `adapter`
- prereq policy: `fail-fast`
- projection rate: `0.8170`
- infeasible rate: `0.8170`

Interpretation:

- The upstream-backed ORCA path executes successfully on a real Robot SF benchmark scenario.
- The projection from velocity-vector output into `unicycle_vw` commands is active often enough that
  it must stay explicit in benchmark-facing interpretation.
- The probe campaign is too small and degenerate for SNQI interpretation; use it as an integration
  proof, not as a ranking benchmark.

## Canonical paper-surface comparison

To answer the real performance question, the explicit `Python-RVO2` ORCA path was rerun on the
same scenario surface used by the canonical paper matrix.

Compared campaigns:

- frozen canonical ORCA:
  `output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue579_snqi_v3_regen_20260318_163407`
- explicit `Python-RVO2` ORCA on the same surface:
  `output/benchmarks/camera_ready/paper_experiment_matrix_v1_orca_python_rvo2_only_issue632_orca_only_paper_surface_20260319_211426`

Key result:

- frozen ORCA:
  - episodes: `141`
  - success: `0.2340`
  - collisions: `0.0426`
  - runtime: `98.2864s`
  - projection rate: `0.8195`
- explicit `Python-RVO2` ORCA:
  - episodes: `141`
  - success: `0.2199`
  - collisions: `0.0355`
  - runtime: `122.8279s`
  - projection rate: `0.8122`

Termination breakdown:

- frozen ORCA:
  - success: `33`
  - collision: `6`
  - max_steps: `102`
- explicit `Python-RVO2` ORCA:
  - success: `31`
  - collision: `5`
  - max_steps: `105`

Interpretation:

- Collision rate improves slightly.
- Success rate gets worse.
- Runtime gets worse materially.
- Therefore this prototype is **not** a performance upgrade for the canonical paper matrix.

Claim boundary for this comparison:

- The explicit `Python-RVO2` path is a better-documented and more defensible upstream-backed ORCA
  entry.
- It should not replace the frozen paper ORCA numbers based on current evidence.
- The value of this issue is provenance clarity and explicit command-space translation, not a
  superior headline benchmark result.

## Verdict rule

This issue should only claim `viable benchmark prototype` when all of the following hold:

- the vendored upstream example runs successfully
- the ORCA adapter executes with `allow_fallback: false`
- the projection from `velocity_vector_xy` to `unicycle_vw` is documented explicitly
- the dedicated ORCA probe scenario completes as a real Robot SF benchmark run

## Claim boundary

What this issue proves:

- Robot SF can execute an upstream-backed ORCA family baseline with explicit provenance
- the holonomic-to-unicycle projection is visible and reviewable

What this issue does not prove:

- paper-level ORCA parity against every external benchmark harness
- that holonomic assumptions disappear just because the final command space is `unicycle_vw`
- that the explicit `Python-RVO2` prototype improves ORCA performance on the paper matrix
