# Issue #4224 External Dataset Registry

Plain-language summary: this note records the source, access, and claim boundary for five external
dataset assets registered from Alyassi et al. 2025 Table 7. The PR only adds metadata and local
provenance checks; it does not download, redistribute, ingest, or benchmark any raw dataset.

Related issue: [#4224](https://github.com/ll7/robot_sf_ll7/issues/4224)

Verification date: 2026-07-03.

## Asset Verification

| asset_id | official source URL | license or terms URL | access mode | scripted download permitted | expected local layout | consuming issues |
| --- | --- | --- | --- | --- | --- | --- |
| `atc-pedestrian` | `https://dil.atr.jp/crest2010_HRI/ATC_dataset/` | `https://dil.atr.jp/crest2010_HRI/ATC_dataset/` | `clickthrough_manual` | No. The registry keeps `auto_download_allowed=False` until stable terms and URL approval are recorded. | `atc_pedestrian/` with `atc-*.csv` or other trajectory `.csv`, plus `README*`, `LICENSE*`, or `TERMS*`. | #2918, #3161, #3971, #3813, #3950 |
| `eth-ucy` | `https://vision.ee.ethz.ch/datsets.html` plus UCY Crowds-by-Example source/paper provenance | `https://vision.ee.ethz.ch/datsets.html` | `clickthrough_manual` | No. ETH and UCY terms are not encoded as an automated redistribution or download permission. | `eth-ucy/` with ETH/Hotel `obsmat.txt`, UCY `.vsp` or trajectory `.txt`, plus `README*`, `LICENSE*`, or `TERMS*`; see `docs/datasets/eth-ucy.md`. | #2844, #4013, #4224, #4280 |
| `ind-crossings` | `https://levelxdata.com/ind-dataset/` | `https://levelxdata.com/ind-dataset/` | `research_request_byo` | No. Manual request form + terms acceptance; non-commercial research use and non-redistribution; local bring-your-own staging only. | `ind_crossings/` with per-recording `*_tracks.csv`, `*_tracksMeta.csv`, `*_recordingMeta.csv`, and a `*_background.png` image, plus `README*`, `LICENSE*`, or `TERMS*`; see `docs/datasets/ind.md`. | #4290, #4224, #3161 |
| `crowdbot` | `https://www.epfl.ch/labs/lasa/crowdbot-dataset/` | `https://www.epfl.ch/labs/lasa/crowdbot-dataset/` | `research_request_byo` | No. Research/DataPort access terms must be preserved locally before staging. | `crowdbot/` with `.bag`, `.csv`, or `.json` recordings/metadata, plus `README*`, `LICENSE*`, or `TERMS*`; see `docs/datasets/crowdbot.md`. | #4357, #4224, #3977 |
| `scand-demos` | `https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/0PRYRH` | `https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/0PRYRH` | `clickthrough_manual` | No. The DataVerse release is large and file selection plus terms should be handled manually first. | `scand_demos/` with `.bag`, `.csv`, or `.json` recordings/metadata, plus `README*`, `LICENSE*`, or `TERMS*`; see `docs/datasets/scand.md`. | #4357, #4224, #1470, #1496 |

## Claim Boundary

External-data registry only. Registering an asset does not mean data has been downloaded, staged,
license-approved for redistribution, ingested, or used to support a benchmark, paper, dissertation,
or planner claim.

These assets are alternatives or complements to the blocked Stanford Drone Dataset provenance route
tracked in #4079. This note does not resolve Stanford Drone Dataset canonical equivalence.

## Follow-Up

After a maintainer obtains a dataset under verified terms, stage it locally with:

```bash
uv run python scripts/tools/manage_external_data.py stage <asset-id> \
  --source <local-staged-path> \
  --manifest-out output/external_data/manifests/<asset-id>.provenance.json
uv run python scripts/tools/manage_external_data.py --json provenance-check <asset-id> \
  --manifest output/external_data/manifests/<asset-id>.provenance.json
```

Only pin `expected_tree_sha256` after reviewing that the staged tree corresponds to the official
asset and no raw data is tracked by git.

### Shape-contract loaders (public slice b)

Per the 2026-07-03 maintainer ruling, each asset gets a license-safe shape-contract loader plus
skip-if-absent tests that validate the expected structure and skip when data is absent (no bytes,
ever). Current status:

| Asset | Loader module | Tests |
| --- | --- | --- |
| `atc-pedestrian` | `robot_sf/data/external/atc.py` | `tests/data/external/test_atc_shape.py` |
| `eth-ucy-trajectories` | `robot_sf/data/external/eth_ucy.py` | `tests/data/external/test_eth_ucy_shape.py` |
| `ind-crossings` | `robot_sf/data/external/ind.py` | `tests/data/external/test_ind_shape.py` |
| `crowdbot` | `robot_sf/data/external/crowdbot.py` | `tests/data/external/test_crowdbot_shape.py` |
| `scand-demos` | `robot_sf/data/external/scand.py` | `tests/data/external/test_scand_shape.py` |

`crowdbot` and `scand-demos` are recording-style datasets (ROS bags plus exported CSV/JSON tables and
a license/readme copy) and share the reusable engine in
`robot_sf/data/external/recording_shape_contract.py`. A passing shape contract is local structural
evidence only; it is not a benchmark, ingestion, or paper-facing claim.
