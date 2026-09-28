# 0.0.7 → 0.0.8 Stage-3 comparison input contract

Run `scripts/analysis/compare_release_007_008.py --help` for paths. Supply the
accepted 0.0.7 [Zenodo archive](https://zenodo.org/records/22814343) unchanged,
a **clean** corrected-source checkout, the 0.0.8 candidate campaign root, an
identity JSON, an attribution ledger JSON, and paths for JSON/JSONL reports.
The archive is checked by SHA-256 before its members are read in place. The
[readback receipt](issue_9668_stage3_007_cold_readback.json) records the
executed 0.0.7 source/config/matrix pins; the older checked-in release manifest
is not the comparison reference.

The identity JSON has schema `release_007_008_candidate_identity.v1` and
`release: "0.0.8"`. It declares `source_sha` (the clean checkout HEAD),
`effective_config_path` and `effective_config_sha256`, `scenario_matrix` with
`path` and `sha256`, the sorted canonical 48 `scenario_ids`, seeds 111–140,
`episode_files` mapping each `runs/<arm>__differential_drive/episodes.jsonl`
path to its SHA-256, and exactly 14 `arm_slots`. Each slot has `old_key`,
`new_key`, `implementation_replaced`, `implementation_version`, `config_path`,
`config_sha256`, and `row_algos` (the runtime `algo` keys allowed for that arm).
Only the four #9751 hybrid slots may acquire new v4-named `new_key`s. Historical
adaptive hybrid rows have 60 intentional `algo=orca` selections on
`francis2023_leave_group`; their cohort keys still name the historical arm.
A candidate must declare its own exact allowed runtime keys, not inherit this
historical selection by assumption.

`versioned_changes` names each correction with a unique `id`, `kind`
(`source`, `config`, `map`, `model`, or `planner`), `version`, `old_identity`, and
`new_identity`. Each new identity must equal the bound source/config/matrix/
arm-config hash or a file hash listed under `versioned_inputs` (`path`,
`sha256`). This binding does not prove the correction's causal mechanism.

The ledger has schema `release_007_008_attribution_ledger.v1`, the same
`candidate_source_sha`, and one `entries` item per changed field. An item has
`finding_id`, `change_id`, `receipt_path`, and `receipt_sha256`. Receipt paths
are relative to the ledger directory. A receipt has schema
`release_007_008_causal_receipt.v1`, matching `change_id`, an explicit
`finding_ids` list, a nonempty `mechanism`, a relative `evidence_path` and its
`evidence_sha256`, and `review` with `decision: "accepted"`, `reviewer`, and
`reviewed_at_utc`. The scientific reviewer must inspect the mechanism and
source/input evidence; this tool only verifies the recorded decision, bindings,
and bytes. Unexplained findings, orphaned or malformed ledger entries, invalid
rows, and incomplete/duplicate matrices block structural comparison acceptance.

The findings JSONL contains every changed common outcome/metric with old/new
values at absolute tolerance `1e-12`, plus `implementation replaced` for a v4
slot. The summary reports paired success/collision/timeout rate changes,
rank shifts, SNQI mean/rank changes, and incomplete identities. Collision rate
here means observed `outcome.collision_event`; planner-caused contact requires
separate #9729 validation. Equal historical non-finite metric sentinels are
reported symbolically, never as numeric equality evidence. The H400 doorway
slice uses its separate manifest and is excluded from this H600 comparison.

No 0.0.8 comparison or Chapter 7 claim can be admitted from this tool until
real candidate rows, all accepted causal receipts, and the other #9730 release
gates pass independent review.
