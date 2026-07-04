# Issue #3385 Closure Audit

Plain-language summary: issue
[#3385](https://github.com/ll7/robot_sf_ll7/issues/3385) is mostly implemented. The legacy
`robot_sf/benchmark/camera_ready_campaign.py` module is now a thin compatibility facade over the
`robot_sf/benchmark/camera_ready/` package, and the test import surface has been preserved across
the merged slices. The issue should stay open because the extracted
`robot_sf/benchmark/camera_ready/campaign.py` orchestrator still carries
`# noqa: C901, PLR0915`; the final acceptance criterion says the `run_campaign` orchestration path
should no longer need those cyclomatic-complexity and statement-count suppressions.

## Audit Scope

- Live issue thread reviewed on 2026-07-04, including maintainer comments through
  `2026-07-04T12:07:11Z`.
- Linked and merged pull requests reviewed from the issue timeline and pull request search for
  `#3385`.
- This is an integration/closure audit only. It records delivered evidence and the smallest
  remaining implementation gap; it does not run a benchmark campaign or make a paper-facing claim.

## Acceptance Mapping

| Acceptance criterion from #3385 | Delivered evidence | Status |
| --- | --- | --- |
| `camera_ready_campaign.py` becomes a compatibility facade over a `camera_ready/` package. | PR [#4422](https://github.com/ll7/robot_sf_ll7/pull/4422), squash commit `628933552`, reduced the legacy module to a thin facade. Current `camera_ready_campaign.py` is 15 lines and re-exports from `robot_sf.benchmark.camera_ready._legacy_campaign_facade`. | Met |
| Preserve existing imports from `tests/benchmark/test_camera_ready_campaign.py` during extraction. | PRs [#3411](https://github.com/ll7/robot_sf_ll7/pull/3411), [#3412](https://github.com/ll7/robot_sf_ll7/pull/3412), [#3413](https://github.com/ll7/robot_sf_ll7/pull/3413), [#3873](https://github.com/ll7/robot_sf_ll7/pull/3873), [#3882](https://github.com/ll7/robot_sf_ll7/pull/3882), [#4213](https://github.com/ll7/robot_sf_ll7/pull/4213), [#4309](https://github.com/ll7/robot_sf_ll7/pull/4309), [#4355](https://github.com/ll7/robot_sf_ll7/pull/4355), [#4384](https://github.com/ll7/robot_sf_ll7/pull/4384), [#4408](https://github.com/ll7/robot_sf_ll7/pull/4408), [#4422](https://github.com/ll7/robot_sf_ll7/pull/4422), [#4427](https://github.com/ll7/robot_sf_ll7/pull/4427), and [#4436](https://github.com/ll7/robot_sf_ll7/pull/4436) each kept the focused camera-ready tests green according to their pull request or issue-thread validation notes. | Met |
| Extract the planned concern areas: utilities, preflight/config, artifact output, route clearance, summaries, reporting, and campaign orchestration. | Current package files include `_util.py`, `_preflight.py`, `_config.py`, `_config_types.py`, `_artifacts.py`, `_route_clearance.py`, `_summaries.py`, `_reporting.py`, `_run_state.py`, and `campaign.py`. The timeline shows matching merged slices: #3411, #3412, #3413, #3675, #3873, #3882, #4172, #4213, #4309, #4355, #4384, #4408, #4422, #4427, and #4436. | Met |
| Keep behavior unchanged and preserve campaign evidence semantics. | Merged pull request bodies and issue comments consistently classify the work as behavior-preserving. The current audit found no tracked schema or artifact-semantics change in the closure slice. This audit does not independently rerun a full before/after campaign. | Partially met by existing pull request evidence |
| Reduce `run_campaign` orchestration so `C901`, `PLR0912`, and `PLR0915` suppressions are no longer needed. | PR [#4436](https://github.com/ll7/robot_sf_ll7/pull/4436), squash commit `f8a31095`, removed `PLR0912` from `_run_campaign_orchestrator`. The current source still has `def _run_campaign_orchestrator(  # noqa: C901, PLR0915` in `robot_sf/benchmark/camera_ready/campaign.py`. An unsuppressed focused Ruff check reports `C901` complexity `12 > 10` and `PLR0915` statements `141 > 60`. | Not met |

## Merged Pull Request Evidence

| Pull request | Merge commit | Delivered slice |
| --- | --- | --- |
| [#3411](https://github.com/ll7/robot_sf_ll7/pull/3411) | `eefbb71d` | Extracted leaf utility helpers into `camera_ready/_util.py`. |
| [#3412](https://github.com/ll7/robot_sf_ll7/pull/3412) | `7ea5584e` | Extracted preflight payload helpers. |
| [#3413](https://github.com/ll7/robot_sf_ll7/pull/3413) | `5656de03` | Extracted early artifact writer helpers. |
| [#3675](https://github.com/ll7/robot_sf_ll7/pull/3675) | `0487e855` | Extracted scenario kinematics helper. |
| [#3873](https://github.com/ll7/robot_sf_ll7/pull/3873) | `80156ff0` | Moved camera-ready artifact writers into the package. |
| [#3882](https://github.com/ll7/robot_sf_ll7/pull/3882) | `8c9f2e69` | Extracted run-state helpers. |
| [#4172](https://github.com/ll7/robot_sf_ll7/pull/4172) | `aca3cd06` | Refactored preflight orchestration. |
| [#4213](https://github.com/ll7/robot_sf_ll7/pull/4213) | `acafc614` | Extracted report writer behavior. |
| [#4309](https://github.com/ll7/robot_sf_ll7/pull/4309) | `dd09a1a9` | Moved remaining `run_campaign` orchestration into `camera_ready/campaign.py`. |
| [#4355](https://github.com/ll7/robot_sf_ll7/pull/4355) | `925578a3` | Extracted config loader and validation into `camera_ready/_config.py`. |
| [#4384](https://github.com/ll7/robot_sf_ll7/pull/4384) | `fdee080c` | Extracted config dataclasses and constants into `_config_types.py`. |
| [#4408](https://github.com/ll7/robot_sf_ll7/pull/4408) | `f4027d06` | Thinned the config compatibility facade. |
| [#4422](https://github.com/ll7/robot_sf_ll7/pull/4422) | `62893355` | Thinned the legacy `camera_ready_campaign.py` facade. |
| [#4427](https://github.com/ll7/robot_sf_ll7/pull/4427) | `87002ba2` | Split dependency resolution out of the campaign orchestration path. |
| [#4436](https://github.com/ll7/robot_sf_ll7/pull/4436) | `f8a31095` | Extracted planner-matrix, variant, batch, and skipped-combination helpers; removed `PLR0912` from `_run_campaign_orchestrator`. |

## Remaining Checklist

- [ ] Continue thinning `robot_sf/benchmark/camera_ready/campaign.py::_run_campaign_orchestrator`
  until the `C901` and `PLR0915` suppressions can be removed without moving complexity into an
  equally broad replacement helper.
- [ ] Re-run the focused camera-ready test file and Ruff complexity check after that implementation
  slice.
- [ ] Before closing #3385, add a closure comment mapping each acceptance criterion to the pull
  request or commit evidence above, plus the final suppression-removal proof.

## Fragmentation Guard

More than two #3385 pull requests merged in the 24 hours before this audit. This note is therefore a
consolidation/integration slice rather than another narrow guardrail. It records the current
contract boundary and prevents the remaining work from being rediscovered as a fresh issue.
