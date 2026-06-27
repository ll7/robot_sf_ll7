# SocNavBench Asset Setup (License-Safe)

This repository vendors only the SocNavBench planner code subset in `third_party/socnavbench`.
It does **not** vendor external dataset assets due to licensing constraints.

Use this runbook to populate local assets in `third_party/socnavbench` without committing them.

Primary upstream references used:

- `output/SocNavBench/docs/install.md`
- `output/SocNavBench/sd3dis/README.md`
- `output/SocNavBench/sd3dis/preprocess_meshes.sh`
- `output/SocNavBench/surreal/README.md`
- `output/SocNavBench/wayptnav_data/README.md`

## Required Layout in `third_party/socnavbench`

For `schematic` mode (`render_mode=schematic`), required:

- `third_party/socnavbench/wayptnav_data`
- `third_party/socnavbench/sd3dis/stanford_building_parser_dataset`
- `third_party/socnavbench/sd3dis/stanford_building_parser_dataset/traversibles`

For `full-render`, additionally required:

- `third_party/socnavbench/surreal/code/human_meshes`
- `third_party/socnavbench/surreal/code/human_textures`

## 1. Dependencies

Install SocNav Python prerequisites in this repo environment:

```bash
uv sync --extra socnav
```

## 2. Obtain Data From Official Sources

### 2.1 Curated SD3DIS maps and `wayptnav_data` (recommended path)

Per upstream `docs/install.md`, download the curated package from the official SocNavBench drive and place:

- `stanford_building_parser_dataset` under `third_party/socnavbench/sd3dis/`
- `wayptnav_data` under `third_party/socnavbench/`

### 2.2 Raw SD3DIS meshes (alternative path)

Per upstream `sd3dis/README.md`:

1. Request and download raw meshes from the Stanford S3DIS dataset website.
2. Place raw tar files in `sd3dis/stanford_building_parser_dataset_raw/`.
3. Run preprocessing from `sd3dis`:

```bash
bash preprocess_meshes.sh
```

Then download traversibles as instructed upstream (`gdown` link in `sd3dis/README.md`) and extract:

```bash
tar -zxf traversibles.tar.gz
```

Target after extraction:

- `third_party/socnavbench/sd3dis/stanford_building_parser_dataset/traversibles`

### 2.3 SURREAL meshes/textures (only for full-render)

Follow upstream `surreal/README.md` license and generation steps.
This requires accepting SURREAL/SMPL licenses and running the mesh generation flow.

## 3. Optional: Stage From Existing Local Clone

If you already downloaded data in `output/SocNavBench`, copy available assets into `third_party/socnavbench`:

```bash
uv run python scripts/tools/prepare_socnav_assets.py --copy-from-source
```

Full-render staging check:

```bash
uv run python scripts/tools/prepare_socnav_assets.py \
  --copy-from-source \
  --render-mode full-render
```

## 4. Validate Asset Presence

Schematic validation:

```bash
uv run python scripts/tools/prepare_socnav_assets.py
```

If you need JSON output:

```bash
uv run python scripts/tools/prepare_socnav_assets.py \
  --report-json output/tmp/socnav_asset_report.json
```

Readiness reporting is fail-closed and placeholder-aware. Each required asset is reported
with a `status`:

- `available`: directory exists and contains real files (restored).
- `placeholder`: directory exists but is empty — a shell, **not** counted as restored.
- `missing`: directory does not exist.
- `excluded`: not required for the selected `render_mode` (for example SURREAL assets in
  `schematic`).

The command exits non-zero unless every required asset is `available`; both `placeholder`
and `missing` required assets appear under `missing_required` so an empty directory shell
can never pass as a restored asset (issue #1456).

## 5. Benchmark Sanity Test (No Fallback)

Run strict preflight smoke benchmark:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/camera_ready_smoke_all_planners.yaml \
  --label socnav_asset_sanity
```

Expected behavior:

- With complete assets, `socnav_bench` preflight does not fail due to missing data.
- Without required assets, run fails fast with explicit missing path errors.

## 6. SocNavBench Map Import Batches

Issue #334 uses a staged map-import manifest instead of bulk asset ingestion:

```bash
uv run python scripts/tools/validate_socnav_map_batch.py --batch-id eth_first
```

The first batch is `ETH` and is defined in
`configs/maps/socnavbench_import_batches.yaml`. This command validates that the exact source
mesh/traversible inputs are staged locally before any converted Robot SF SVG or scenario wiring is
accepted. Missing required source assets fail closed.

## Licensing and Repository Hygiene

- Do not commit downloaded dataset assets.
- Keep all third-party data local only.
- The repository `.gitignore` explicitly ignores SocNav data directories under `third_party/socnavbench`.
