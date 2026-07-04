# Issue #4224 Closure Audit

Plain-language summary: the public external-data registry, acquisition-documentation, and
skip-if-absent shape-contract loader slices for the five Alyassi Table 7 dataset families are
merged, but issue #4224 is not closable until private-side data acquisition/seeding and trusted
checksum pinning happen outside this public PR lane.

Related issue: [#4224](https://github.com/ll7/robot_sf_ll7/issues/4224)

Audit date: 2026-07-04.

## Claim Boundary

This is a closure-audit integration report only. It does not download, stage, redistribute, ingest,
or benchmark ATC, ETH/UCY, inD, CrowdBot, SCAND, or Stanford Drone Dataset files. It does not
promote benchmark, paper, dissertation, prediction-comparability, or planner-consumer claims.

## Acceptance Evidence

| Criterion | Status | Evidence |
| --- | --- | --- |
| Register the five Alyassi Table 7 assets in the public external-data registry with source, license/access notes, stable subpaths, related issue links, disabled auto-download, and checksum-pending metadata. | Met for public registry surface. | PR [#4238](https://github.com/ll7/robot_sf_ll7/pull/4238) added `atc-pedestrian`, `eth-ucy-trajectories`, `ind-crossings`, `crowdbot`, and `scand-demos` to `scripts/tools/manage_external_data.py`, updated tests, and created `docs/context/issue_4224_external_dataset_registry.md`. Follow-up PRs [#4287](https://github.com/ll7/robot_sf_ll7/pull/4287), [#4314](https://github.com/ll7/robot_sf_ll7/pull/4314), [#4363](https://github.com/ll7/robot_sf_ll7/pull/4363), and [#4501](https://github.com/ll7/robot_sf_ll7/pull/4501) tightened family-specific docs and registry traceability. |
| Keep acquisition license-safe: no raw bytes, no redistribution, no automated downloads unless terms explicitly permit it. | Met for public repository surface. | PR #4238 set `auto_download_allowed=False` for the registered assets and documented manual or bring-your-own access. Current `docs/external_data_setup.md` states `download` fails closed for these license/agreement-gated groups. |
| Provide per-family acquisition and layout documentation for external users and maintainers. | Met for the public documentation surface. | PR #4287 added ETH/UCY docs, PR [#4295](https://github.com/ll7/robot_sf_ll7/pull/4295) added ATC docs, PR #4314 added inD docs, and PR #4363 wired CrowdBot and SCAND docs into external-data navigation and registry traceability. |
| Add license-safe shape-contract loaders and skip-if-absent tests for all five #4224 dataset families. | Met for public slice (b). | PR [#4300](https://github.com/ll7/robot_sf_ll7/pull/4300) added ETH/UCY loader/tests, PR [#4316](https://github.com/ll7/robot_sf_ll7/pull/4316) added ATC loader/tests, PR [#4322](https://github.com/ll7/robot_sf_ll7/pull/4322) added inD loader/tests, and PR [#4346](https://github.com/ll7/robot_sf_ll7/pull/4346) added CrowdBot and SCAND loaders/tests plus the shared recording-shape engine. |
| Preserve the no-claim boundary: registry and shape-contract checks are not benchmark or paper evidence. | Met. | `docs/context/issue_4224_external_dataset_registry.md` and the merged PR bodies state that passing shape contracts only proves local structural compatibility when data is staged; it is not ingestion, benchmark, prediction-comparability, dissertation, or paper-facing evidence. |
| Complete private-side acquisition/seeding and pin trusted `expected_tree_sha256` values after staged-tree review. | Not met in public repo; intentionally blocked here. | Maintainer comments on 2026-07-03 at 09:17 UTC and 20:53 UTC state the remaining work is maintainer/private-side staging under each dataset's terms, provenance-check, and checksum pinning. This task disallows compute submission and is not authorized for private data acquisition or issue comments. |

## Residual Checklist

- Private ops acquires each dataset under its own terms.
- Private ops seeds or exposes the local staging source on the intended machines or durable store.
- Maintainer runs `manage_external_data.py stage` and `provenance-check` for each staged tree.
- Follow-up public PR pins reviewed `expected_tree_sha256` values without committing raw data.
- Any downstream ingestion, prior extraction, prediction-comparability, or benchmark claim uses a
  separate issue with its own provenance and validation boundary.

## Closure Decision

Do not close issue #4224 from this audit alone. The merged public PRs satisfy the repository-visible
registry, documentation, and shape-contract acceptance criteria, but the issue's latest maintainer
state keeps it open for private-side acquisition/seeding and checksum pinning. Because this run is
not authorized to comment on or close issues, the PR body is the state-propagation surface for this
closure audit.
