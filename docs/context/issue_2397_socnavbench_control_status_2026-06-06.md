# Issue #2397 SocNavBench Control-Pipeline Asset Status

Date: 2026-06-06

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/2397>
- <https://github.com/ll7/robot_sf_ll7/issues/1456>
- <https://github.com/ll7/robot_sf_ll7/issues/562>
- <https://github.com/ll7/robot_sf_ll7/issues/1584>

Evidence sidecar:

- [evidence/issue_2397_socnavbench_control_status_2026-06-06.json](evidence/issue_2397_socnavbench_control_status_2026-06-06.json)

## Result

The SocNavBench control-pipeline asset family is still unavailable on this checkout. This is an
analysis-only fail-closed status report, not benchmark evidence and not an asset restoration.

The repository contains the vendored SocNavBench code subset under `third_party/socnavbench`, but
the required external data assets are missing:

| Required path | Status | Source command |
|---|---|---|
| `third_party/socnavbench/wayptnav_data` | missing | `manage_external_data.py check socnavbench-control` |
| `third_party/socnavbench/sd3dis/stanford_building_parser_dataset` | missing | `prepare_socnav_assets.py` |
| `third_party/socnavbench/sd3dis/stanford_building_parser_dataset/traversibles` | missing | `prepare_socnav_assets.py` |
| `third_party/socnavbench/sd3dis/stanford_building_parser_dataset/mesh/ETH` | missing | `manage_external_data.py check socnavbench-s3dis-eth` |
| `third_party/socnavbench/sd3dis/stanford_building_parser_dataset/traversibles/ETH/data.pkl` | missing | `manage_external_data.py check socnavbench-s3dis-eth` |

The local source hydration path `output/SocNavBench` was not present during the read-only scout, so
`prepare_socnav_assets.py --copy-from-source` has no local source tree to copy from.

## Commands Run

```bash
uv run python scripts/tools/manage_external_data.py --json check socnavbench-control
```

Exit code: `2`. Result: `status=incomplete`, `ok=false`, missing `wayptnav_data`.

```bash
uv run python scripts/tools/manage_external_data.py --json check socnavbench-s3dis-eth
```

Exit code: `2`. Result: `status=incomplete`, `ok=false`, missing the ETH mesh directory and ETH
traversible pickle.

```bash
uv run python scripts/tools/prepare_socnav_assets.py \
  --report-json output/validation/issue2397_socnav_assets_report.json
```

Exit code: `2`. Result: `MISSING_REQUIRED_ASSETS`, missing `wayptnav_data`, `sbpd_dataset`, and
`sbpd_traversibles`. The generated `output/validation/...` JSON is local/ignored; the compact
sidecar linked above is the durable tracked evidence.

## Downstream Row Policy

While these assets remain unavailable, SocNavBench-family rows must use the policy from
[issue_1584_socnav_unavailable_row_policy.md](issue_1584_socnav_unavailable_row_policy.md):

- `row_status`: `unavailable/excluded`
- `availability_status`: `not_available`
- `benchmark_success`: `false`
- `availability_reason`: `missing_socnavbench_control_pipeline_assets`

This is not fallback execution, not degraded success, and not successful benchmark evidence. The
row must not contribute to success, collision, near-miss, SNQI, runtime, ranking, or paper-facing
aggregate evidence.

## Next Empirical Action

Issue #1456 remains the restoration gate. The next action is to stage licensed SocNavBench
control/S3DIS assets under `third_party/socnavbench` through the manual path in
[../socnav_assets_setup.md](../socnav_assets_setup.md), rerun the schematic asset check, and only
then rerun the focused issue #562 fail-fast probe. Do not fabricate placeholders or treat empty
directory shells as availability proof.
