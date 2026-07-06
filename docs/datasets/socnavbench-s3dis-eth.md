# SocNavBench S3DIS ETH External Data

Plain-language summary: Robot SF can check the local shape of SocNavBench ETH map assets, but it
does not download, redistribute, or benchmark with the license-gated dataset unless you stage it
yourself.

Related issues: [#4279](https://github.com/ll7/robot_sf_ll7/issues/4279),
[#1498](https://github.com/ll7/robot_sf_ll7/issues/1498),
[#334](https://github.com/ll7/robot_sf_ll7/issues/334)

## Acquisition

Use the official SocNavBench installation instructions and the Stanford 3D Indoor Spaces
(S3DIS) or Stanford Building Parser Dataset (SBPD) access path referenced there. The upstream code
repository is <https://github.com/CMU-TBD/SocNavBench>. The public Robot SF repository does not
encode a download URL or commit dataset bytes because the S3DIS/SBPD meshes and traversible files
have separate access terms.

The canonical Robot SF registry entry is `socnavbench-s3dis-eth` in
`scripts/tools/manage_external_data.py`. Use it to inspect the expected local layout:

```bash
uv run python scripts/tools/manage_external_data.py explain socnavbench-s3dis-eth
```

## Expected Layout

By default, the registry expects the SocNavBench root at `third_party/socnavbench/`. To share one
staged copy across worktrees, set `ROBOT_SF_EXTERNAL_DATA_ROOT` and place the SocNavBench tree under
`$ROBOT_SF_EXTERNAL_DATA_ROOT/socnavbench/`.

Required files under the SocNavBench root:

```text
sd3dis/stanford_building_parser_dataset/mesh/ETH/
sd3dis/stanford_building_parser_dataset/traversibles/ETH/data.pkl
```

The traversible pickle must load as a mapping with:

- `resolution`: finite positive numeric map resolution.
- `traversible`: non-empty two-dimensional boolean or numeric array.

These checks are shape and layout checks only. They do not assert map content correctness, licensing
status, benchmark readiness, or paper-facing evidence.

## Validation

Without staged data, the structural test skips:

```bash
uv run pytest tests/data/external/test_socnavbench_eth_shape.py
```

With locally staged official assets, the same test loads `data.pkl` and checks the cheap structural
contract. A successful pass is only local staging evidence; it is not a full benchmark campaign, a
SocNavBench ETH map conversion, or a dissertation claim.

To write a reviewable local audit artifact after staging official assets:

```bash
uv run python scripts/validation/check_socnavbench_eth_shape_contract.py \
  --json-out output/validation/issue_4279/socnavbench_eth_shape_contract_audit.json
```

The command exits `0` only when the staged ETH mesh directory and traversible pickle satisfy the
loader shape contract. Missing or malformed staged data exits `2` with a JSON report that names the
missing path or malformed contract.
