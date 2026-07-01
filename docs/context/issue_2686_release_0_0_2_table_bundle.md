# Issue #2686 Release 0.0.2 Table Evidence Bundle

Related issue: [#2686](https://github.com/ll7/robot_sf_ll7/issues/2686)

## Status

Current as of 2026-07-01. This note records the compact dissertation-facing table handoff for the
existing scoped benchmark release `0.0.2`. The bundle packages release-backed table artifacts
through the existing `scripts/tools/benchmark_publication_bundle.py dissertation-bundle` workflow.

Issue #3482 later withdrew release `0.0.2` collision-count-derived claims because exact-event
provenance could not be recovered. The release table bundle remains useful for non-collision
release-table provenance, but `total_collision_count`, collision-count-derived rows, and any
paper/dissertation collision-count claims from these tables are not usable unless a future promoted
reconciliation bundle supersedes
`docs/context/evidence/issue_3482_release_0_0_2_claim_disposition_2026_07_01/manifest.json`.

## Source Release

- Release tag: `0.0.2`
- Release URL: `https://github.com/ll7/robot_sf_ll7/releases/tag/0.0.2`
- DOI: `https://doi.org/10.5281/zenodo.19563812`
- Release archive:
  `paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz`
- Archive SHA-256:
  `64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90`
- Source commit: `f7ebdcae2375d085e925213197a75a386e26a79c`
- Release scope: scoped seven-planner release; `socnav_bench` excluded because licensed
  SocNavBench assets were not staged.

## Table Mapping

| Table label | Bundle artifact id | Release payload source | Manuscript use | Boundary |
| --- | --- | --- | --- | --- |
| `tab:results-overview` | `tab_results_overview` | `reports/campaign_table_core.md` | `results` for non-collision fields only | Existing release-backed core-planner overview only; collision-count columns withdrawn by #3482. |
| `tab:robot_sf_release_planner_results` | `tab_robot_sf_release_planner_results` | `reports/campaign_table.md` | `results` for non-collision fields only | Existing release-backed planner table only; collision-count columns withdrawn by #3482; no rerun or new ranking claim. |
| `tab:release_failure_count_slices` | `tab_release_failure_count_slices` | `reports/scenario_family_breakdown.md` | `discussion` for non-collision context only | Existing scenario-family outcome slices; collision-count/failure-count interpretations withdrawn by #3482. |

Scope note for `campaign_table_core`:

- This row set is the implementation-maturity `planner_group=core` partition from the paper release
  contract, not the full manuscript-style ORCA versus PPO comparison table.
- In release `0.0.2`, `ppo` is not present in `campaign_table_core` because it was outside the
  scoped core set for that release.

## Tracked Bundle

- `docs/context/evidence/issue_2686_release_0_0_2_table_bundle/artifact_spec.json`
- `docs/context/evidence/issue_2686_release_0_0_2_table_bundle/artifact_manifest.json`
- `docs/context/evidence/issue_2686_release_0_0_2_table_bundle/checksums.sha256`

The copied payload files remain under disposable `output/dissertation_export/...` during smoke runs
and are not committed. A future checkout can recover the source payload by downloading and
extracting the release archive recorded above.

## Reproduction Commands

- `gh release download 0.0.2 --pattern 'paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz'`
- `sha256sum output/issue_2686_release_bundle/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz`
- `scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/benchmark_publication_bundle.py dissertation-bundle ...`

## Boundary

This work republishes and maps existing release `0.0.2` table evidence for downstream report and
archive use. It does not create new benchmark evidence, rerun the benchmark, expand the scoped
release claim, or make dissertation prose paper-ready by itself. After #3482, it also does not
support release `0.0.2` collision-count claims; those claims are withdrawn pending any future
exact-event provenance recovery.
