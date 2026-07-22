# Issue #334 SocNavBench ETH Import Batch

Date: 2026-05-10

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/334>
Follow-up issue: <https://github.com/ll7/robot_sf_ll7/issues/1134>

## Decision

The first SocNavBench map-import batch is `ETH`, matching the maintainer decision on issue #334.
The batch is now represented in `configs/maps/socnavbench_import_batches.yaml`.

## Current Status (Updated 2026-07-22)

The ETH traversible-to-SVG conversion is **complete** and committed to the repository.
PR #4693 converted the official SocNavBench ETH traversible (`data.pkl`, SHA-256
`02d450ee57bdab7aa357457085b3ddac6501ea8ec99324f2f02aecaed82562a6`) into `maps/svg_maps/socnavbench/socnavbench_eth.svg` (SHA-256
`9fb9e126fac37b1c24c8849aeee47dfcccc5ef71fd7fc4e0fea7f78f19d1858d`). The raw licensed
S3DIS/ETH source assets remain unstaged in this Git repository as required by the license-safe
asset policy in `docs/socnav_assets_setup.md`.

The follow-up issue #1134 remaining criteria are now complete:
- Commit the real ETH SVG: done by PR #4693.
- Add a smoke scenario YAML: committed as `configs/scenarios/single/socnavbench_eth_smoke.yaml`.
- Add a focused CPU-only test: committed as `tests/scenarios/test_socnavbench_eth_smoke_scenario.py`
  — proves the real committed ETH map (377 obstacles, 1 robot route, 1 pedestrian route, all
  spawn/goal zones) parses through the scenario pipeline and runs through `make_robot_env`, reset,
  and up to three headless steps with no external data root required.

The batch validator (`validate_socnav_map_batch.py --batch-id eth_first`) remains fail-closed
on hosts without the raw source assets, which is the correct behavior — the conversion evidence
and SHA-256 provenance are tracked through PR #4693 and the issue state surfaces.

## Completed Outputs

- `maps/svg_maps/socnavbench/socnavbench_eth.svg` — committed by PR #4693
- `configs/scenarios/single/socnavbench_eth_smoke.yaml` — committed by the #1134 closure PR
- parser, route/zone, and headless environment-smoke evidence — covered by the focused test
- `docs/context/issue_1134_state.yaml` — updated with ready_for_closure entry

The raw source checksums are preserved in PR #4693 metadata and the issue state surface.

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
