# Issue #1484 Broader Cross-Kinematics Launch Packet - 2026-05-28

Date: 2026-05-28

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1484>
- <https://github.com/ll7/robot_sf_ll7/issues/1353>
- <https://github.com/ll7/robot_sf_ll7/issues/1354>

## Scope

Issue #1484 follows the closed #1353 and #1354 evidence bundles by probing the broader #1353
runnable planner rows over the compact cross-kinematics scenario surface.

Config:

- `configs/benchmarks/issue_1484_broader_cross_kinematics.yaml`
- `configs/benchmarks/issue_1484_broader_cross_kinematics_compatibility.yaml`

The scenario and seed surface stays intentionally compact:

- scenario matrix: `configs/scenarios/sets/cross_kinematics_v1.yaml`
- seed: `111`
- kinematics: `differential_drive`, `bicycle_drive`, `holonomic`
- horizon: `80`

## Rows

Executed rows:

- core: `goal`, `social_force`, `orca`
- experimental: `ppo`, `prediction_planner`, `sacadrl`, `socnav_sampling`

Excluded rows:

- `socnav_bench`: `not_available`, because #1353 preserved it as missing SocNavBench
  control-pipeline assets.
- `rvo` and `dwa`: `unsupported` placeholder adapter rows.

Experimental rows are runnable probes, not planner-invariant cross-kinematics claims.

## Submission Command

```bash
CAMERA_READY_BENCHMARK_CONFIG=configs/benchmarks/issue_1484_broader_cross_kinematics.yaml \
CAMERA_READY_BENCHMARK_OUTPUT_ROOT=output/benchmarks/issue_1484 \
CAMERA_READY_BENCHMARK_LABEL=issue1484-broader-cross-kinematics-20260528 \
CAMERA_READY_SKIP_PUBLICATION_BUNDLE=true \
scripts/dev/sbatch_use_max_time.sh --time 04:00:00 --partition a30 --qos a30-gpu \
  SLURM/Auxme/camera_ready_benchmark.sl
```

## Interpretation Boundary

This run is a broader compatibility smoke/probe. It is not paper-facing evidence by itself and is
not campaign-sized #1484 evidence. Unsupported, unavailable, fallback, degraded, or failed
experimental rows must remain separated from successful core benchmark evidence.

## Completed Smoke Run 2026-05-28

SLURM job `12658` completed successfully:

- result: `COMPLETED 0:0`
- partition: `a30`
- elapsed: `00:02:04`
- campaign id:
  `issue_1484_broader_cross_kinematics_issue1484-broader-cross-kinematics-20260528-gf35fb3b5_20260528_172441`
- total runs: `21`
- successful runs: `21`
- total episodes: `21`
- benchmark success: `true`, basis `core`
- core rows: `9/9`
- AMV coverage: `pass`
- SNQI contract: `warn`

This completed in about two minutes because the config intentionally used one scenario, one seed,
three kinematics modes, seven planner rows, and horizon `80`, yielding 21 short episodes. Treat it
as proof that the matrix executes and reports cleanly. Do not use it to close #1484's original
broader-campaign objective without a larger scenario/seed/runtime surface or an explicit maintainer
scope decision.

`scripts/tools/analyze_camera_ready_campaign.py` reported no consistency findings. Compact evidence
is preserved under `docs/context/evidence/issue_1484_broader_cross_kinematics_2026-05-28/`; raw
episode/run output remains under ignored `output/`.
