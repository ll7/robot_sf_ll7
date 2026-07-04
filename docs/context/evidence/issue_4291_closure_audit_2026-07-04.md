# Issue #4291 Closure Audit

Date: 2026-07-04

Issue: [#4291](https://github.com/ll7/robot_sf_ll7/issues/4291)

Merged implementation PR: [#4296](https://github.com/ll7/robot_sf_ll7/pull/4296)

Plain-language summary: issue #4291 asked for a reproducible way to generate the missing
SocNavBench ETH traversible file from staged local mesh data. PR #4296 delivered the generator,
tests, and runbook. The issue should not be treated as fully closed by repository code alone
because the live maintainer comment keeps it open for the data-side publication step: run the
real mesh build in the SocNavBench environment, re-seed the internal store, and update the
external-data registry pin.

SocNavBench means Social Navigation Benchmark. ETH is the ETH pedestrian map from that dataset
family.

## Acceptance Criteria Evidence

| Criterion from #4291 | Evidence | Closure status |
| --- | --- | --- |
| Add `scripts/tools/generate_socnavbench_traversible.py` or extend the existing wrapper so staged ETH mesh data can produce `traversibles/<MAP>/data.pkl` in the data root, never in git. | PR #4296 added `scripts/tools/generate_socnavbench_traversible.py`. Its PR body states the wrapper builds `traversibles/<MAP>/data.pkl` from staged per-map mesh and writes into the data root. The durable context note records the same contract in `docs/context/issue_4291_socnavbench_traversible_generation.md`. | Met by PR #4296. |
| Print a tree hash so the registry pin can be updated after the trusted generated artifact exists. | This PR adds `output_tree_sha256`, `output_tree_file_count`, and `output_tree_total_size_bytes` to generator reports for existing and newly generated `data.pkl` outputs, with tests in `tests/tools/test_generate_socnavbench_traversible.py`. The post-merge state surface `docs/context/issue_1498_state.yaml` records generated ETH `data.pkl` size and SHA-256, plus `manage_external_data.py check socnavbench-s3dis-eth` and `validate_socnav_map_batch.py --batch-id eth_first --preflight` ready status. | Code-report gap closed by this PR; trusted registry publication remains maintainer-owned. |
| Provide skip-if-absent smoke behavior: fail clearly when the mesh is not staged, and support `--dry-run` without building. | PR #4296 added `tests/tools/test_generate_socnavbench_traversible.py`; its PR body records `uv run pytest tests/tools/test_generate_socnavbench_traversible.py -q` with 17 passing tests and a `--dry-run` smoke returning `blocked_missing_mesh`, exit 2, on a host without staged mesh. | Met by PR #4296. |
| Document the generation command, expected output path, and derived-data policy. | PR #4296 updated `docs/socnav_assets_setup.md` section 7 and added `docs/context/issue_4291_socnavbench_traversible_generation.md`. The note says generated traversibles are derived data, stay in the data root, and must not be committed. | Met by PR #4296. |
| Do not run generation in continuous integration and do not commit generated artifacts. | PR #4296 body states no generation was run and no artifact was committed. The tracked changes are the tool, tests, docs, and changelog only. | Met by PR #4296. |
| Keep actual internal run, hub re-seed, and registry hash update as maintainer post-merge work. | The live issue comment dated 2026-07-03 says PR #4296 merged and keeps #4291 open for the real mesh-based build, internal store re-seed, and external-data registry pin update. `docs/context/issue_1498_state.yaml` records local generated-file evidence, but `configs/maps/socnavbench_import_batches.yaml` still marks source checksums as `pending_official_asset_staging`, and `scripts/tools/manage_external_data.py` still carries no pinned `expected_tree_sha256` for `socnavbench-s3dis-eth`. | Not fully closed by this PR-only lane. Remaining action is maintainer data publication, not another code guard. |

## Current Closure Decision

Do not create another broad generator or checker micro-slice for #4291. PR #4296 delivered the
generator; this closure slice only adds the missing output-tree checksum report and evidence table.
The only remaining blocker is the data-side publication state:

1. Confirm the generated ETH traversible from the trusted SocNavBench environment.
2. Re-seed the internal store.
3. Pin or otherwise record the trusted external-data registry checksum.
4. Re-run the ETH-first map conversion preflight and proceed with issue #1134 when ready.

This audit did not run a full benchmark campaign, submit Slurm or GPU work, edit paper or
dissertation claims, or commit generated SocNavBench data.
