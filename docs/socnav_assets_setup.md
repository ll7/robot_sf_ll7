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

## 7. Generate a Custom-Map Traversible (ETH and friends)

The official SocNavBench asset package ships the S3DIS *area* traversibles (areas 1-6) but **not**
the traversibles for the custom maps (`ETH`, `Hotel`, `Univ`, `Zara`, `DoubleHotel`). SocNavBench
derives those from each map's curated mesh. Because the derived `data.pkl` is *generated data*, it is
never committed: it lives in the data root next to the mesh, and external users regenerate it the
same way.

`scripts/tools/generate_socnavbench_traversible.py` wraps that generation with fail-closed input
validation so it is safe to run (and test) without the heavy SocNavBench mesh dependencies staged.

Validate inputs first (no SocNavBench dependencies required; safe in CI/local):

```bash
uv run python scripts/tools/generate_socnavbench_traversible.py --map ETH --dry-run
```

- Exit `0` means the mesh is staged and a build would run (or the traversible already exists).
- Exit `2` means the mesh is not staged; the printed `next_action` names the exact expected path.

Run the actual generation (maintainer step, in the SocNavBench environment with the mesh staged):

```bash
uv run python scripts/tools/generate_socnavbench_traversible.py --map ETH
```

- **Input:** `sd3dis/stanford_building_parser_dataset/mesh/ETH/` (curated mesh, staged locally).
- **Output (derived, never committed):**
  `sd3dis/stanford_building_parser_dataset/traversibles/ETH/data.pkl`, written **into the data root**
  (honors `ROBOT_SF_EXTERNAL_DATA_ROOT`, otherwise `third_party/socnavbench`).
- The command prints the SHA-256 of the produced `data.pkl` so the external-data registry pin can be
  updated after the maintainer re-seeds the internal store.

The build is idempotent: it skips when `data.pkl` already exists unless `--force` is passed. This
step produces the exact `eth_traversible_pickle` input that
`scripts/tools/validate_socnav_map_batch.py --batch-id eth_first --preflight` reports as missing,
so generating it unblocks the ETH map conversion (issue #1134).

## 8. Convert ETH Traversible to Robot SF SVG

After `eth_first` source assets pass preflight, convert the staged ETH traversible into the planned
Robot SF SVG map with:

```bash
uv run python scripts/tools/convert_socnavbench_traversible_to_svg.py \
  --socnav-root /path/to/socnavbench \
  --output-svg maps/svg_maps/socnavbench/socnavbench_eth.svg \
  --report-json output/maps/issue_1134_socnavbench_eth_conversion.json
```

Use `--dry-run` to validate the source pickle and conversion shape without writing the SVG. If the
official `traversibles/ETH/data.pkl` is absent or malformed, the converter exits `2` and writes no
placeholder map. This is conversion tooling only; it is not benchmark evidence until the generated
SVG passes parser and smoke validation on staged source assets.

For the full local issue #4546 stage/generate/convert/smoke path:

```bash
uv run python scripts/tools/stage_socnavbench_eth_traversible_svg.py \
  --socnav-root /path/to/socnavbench \
  --output-svg maps/svg_maps/socnavbench/socnavbench_eth.svg \
  --report-json output/maps/issue_4546_socnavbench_eth_stage_svg_smoke.json
```

The wrapper composes the same license-safe asset registry, ETH traversible generator, SVG converter,
and Robot SF SVG parser smoke check. Missing licensed source assets fail closed with exit `2` and a
JSON report; the command never downloads data or writes placeholder maps.

Derived SVG artifact policy for issue #4546:

- Commit `maps/svg_maps/socnavbench/socnavbench_eth.svg` only after the wrapper succeeds on the
  official/user-supplied ETH source data and the report records both input `data.pkl` SHA-256 and
  output SVG SHA-256.
- Record the conversion report or a compact evidence note under `docs/context/evidence/` so reviewers
  can verify the source hash, output hash, parser smoke result, and command used to regenerate the
  SVG.
- Do not commit placeholder, dry-run, or fixture-generated SVG output. Until official source data is
  staged and smoke validation is green, the derived SVG remains absent and issue #4546 stays blocked.
- Regenerate and review the committed SVG whenever the official traversible hash, converter settings,
  or SVG parser contract changes.

## Licensing and Repository Hygiene

- Do not commit downloaded dataset assets.
- Do not commit generated traversibles (`traversibles/<MAP>/data.pkl`); they are derived data that
  stays in the data root and is regenerated with the command in section 7.
- Keep all third-party data local only.
- The repository `.gitignore` explicitly ignores SocNav data directories under `third_party/socnavbench`.
