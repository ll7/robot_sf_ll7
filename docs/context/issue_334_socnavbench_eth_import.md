# Issue #334 SocNavBench ETH Import Batch

Date: 2026-05-10

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/334>
Follow-up issue: <https://github.com/ll7/robot_sf_ll7/issues/1134>

## Decision

The first SocNavBench map-import batch is `ETH`, matching the maintainer decision on issue #334.
The batch is now represented in `configs/maps/socnavbench_import_batches.yaml`.

## Current Status

The repository still cannot commit a converted ETH map safely because the actual SocNavBench ETH
map geometry and traversible files are not tracked in the upstream Git repository. Upstream Git
contains ETH episode parameters and pedestrian trajectory CSVs, but the map geometry comes from the
official SocNavBench/S3DIS asset distribution described by upstream `docs/install.md`.

Required local source assets before conversion:

- `third_party/socnavbench/sd3dis/stanford_building_parser_dataset/mesh/ETH/`
- `third_party/socnavbench/sd3dis/stanford_building_parser_dataset/traversibles/ETH/data.pkl`

The validator is deliberately fail-closed:

```bash
uv run python scripts/tools/validate_socnav_map_batch.py --batch-id eth_first
```

On a machine without the official ETH source assets staged, it reports the missing required files
and exits with status `2`. Once the assets are staged and checksummed, the same command becomes the
pre-conversion gate.

## Why This Shape

Committing a hand-authored or inferred ETH-like SVG would look like a SocNavBench import while not
being one. The manifest records the exact source contract and planned outputs without weakening the
license-safe asset policy in `docs/socnav_assets_setup.md`.

## Planned Outputs After Asset Staging

- `maps/svg_maps/socnavbench/socnavbench_eth.svg`
- `configs/scenarios/single/socnavbench_eth_smoke.yaml`
- parser, route/zone, and smoke-benchmark validation evidence
- source checksums for the official ETH mesh/traversible inputs

Issue #1134 owns this conversion step after official source assets are staged.

## Validation

Focused checks for this first-batch manifest:

```bash
rtk uv run ruff check scripts/tools/validate_socnav_map_batch.py tests/tools/test_validate_socnav_map_batch.py
rtk uv run pytest tests/tools/test_validate_socnav_map_batch.py -q
rtk uv run python scripts/tools/validate_socnav_map_batch.py --batch-id eth_first \
  --report-json output/maps/issue_334_eth_batch_missing_assets.json
```

The last command is expected to exit `2` on this machine because the official ETH source assets are
not staged. It reported `eth_mesh_dir` and `eth_traversible_pickle` as missing required inputs.
