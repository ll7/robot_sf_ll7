# Issue #1126 Real SDD Import Smoke

Date: 2026-07-06
Issue: <https://github.com/ll7/robot_sf_ll7/issues/1126>

## Claim Boundary

This is a CPU-only real-data import and smoke-validation slice for the first Stanford Drone Dataset
(SDD) derived scenario candidate. It is not a full benchmark campaign and does not make a
paper-facing claim. The generated SDD-derived map/scenario/provenance files remain in ignored
`output/` because they are derived from local BYO data.

Classification: `exploratory_only`. The scenario imported and executed, but both smoke horizons
timed out, so this is not yet `benchmark_ready`.

## Source And Selection

- Staged tree pin:
  `66dec2c82b0a01b23bf9fa9acef352af86549e7ea749811ea4ef9c47003d4acf`
- Matched tree: 60 `annotations.txt` files, 444959624 bytes
- Selected annotation: `annotations/bookstore/video0/annotations.txt`
- Selected annotation SHA-256:
  `58ec509091faa3e81f5d77c5ab43388ec3a8e7988f9a69952ac523c09799a2b4`
- Source URL: <https://cvgl.stanford.edu/projects/uav_data/>
- License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0
- Scale assumption used for smoke: `--meters-per-pixel 0.0417`

The annotation probe found 200021 usable `Pedestrian` points and 116 tracks meeting
`--min-track-points 8`.

## Commands

```bash
ROBOT_SF_EXTERNAL_DATA_ROOT=/home/luttkule/git/robot_sf_ll7/output/external_data \
  scripts/dev/run_worktree_shared_venv.sh -- \
  python scripts/tools/sdd_curation_preflight.py \
  --annotation /home/luttkule/git/robot_sf_ll7/output/external_data/sdd/annotations/bookstore/video0/annotations.txt \
  --min-track-points 8 \
  --require-benchmark-ready \
  --json

scripts/dev/run_worktree_shared_venv.sh -- \
  python scripts/tools/import_sdd_scenarios.py \
  --annotations /home/luttkule/git/robot_sf_ll7/output/external_data/sdd/annotations/bookstore/video0/annotations.txt \
  --out-dir output/sdd_curation/issue_1126_real_smoke \
  --dataset-id sdd_bookstore_video0_first_real \
  --label Pedestrian \
  --meters-per-pixel 0.0417 \
  --min-track-points 8 \
  --max-pedestrians 4 \
  --stride 12 \
  --max-waypoints 32
```

Generated ignored files:

- `output/sdd_curation/issue_1126_real_smoke/sdd_bookstore_video0_first_real.map.yaml`
- `output/sdd_curation/issue_1126_real_smoke/sdd_bookstore_video0_first_real.scenario.yaml`
- `output/sdd_curation/issue_1126_real_smoke/sdd_bookstore_video0_first_real.provenance.json`

## Smoke Results

Structured artifact load passed:

- scenario: `sdd_bookstore_video0_first_real`
- map pedestrians: 4
- provenance pedestrians: 4
- `meters_per_pixel`: 0.0417

`simple_policy` smoke runs:

| Horizon | Jobs | Outcome | Metrics |
| ---: | --- | --- | --- |
| 80 | `successful_jobs=1`, `failed_jobs=0` | timeout, no collision | `success=false`, `collisions=0`, `near_misses=0` |
| 384 | `successful_jobs=1`, `failed_jobs=0` | timeout, no collision | `success=false`, `collisions=0`, `near_misses=0` |

## Remaining Before Closure

- Decide whether to tune the selected candidate, choose a different scene/video, or accept this one
  as an exploratory-only fixture.
- If aiming for `benchmark_ready`, record a calibrated scene scale and produce a smoke result that
  satisfies the benchmark-ready acceptance boundary.
