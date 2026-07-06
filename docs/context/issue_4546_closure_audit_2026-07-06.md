# Issue #4546 Closure Audit

Date: 2026-07-06

Issue: <https://github.com/ll7/robot_sf_ll7/issues/4546>

## Plain-Language Summary

Issue #4546 is ready to close after this slice. Social Navigation Benchmark (SocNavBench)
ETH source assets were staged outside git under `$ROBOT_SF_EXTERNAL_DATA_ROOT/socnavbench`,
the real ETH traversible pickle was converted into
`maps/svg_maps/socnavbench/socnavbench_eth.svg`, and the generated SVG passed the Robot SF
parser smoke check.

This is smoke evidence for the map-import artifact only. It is not a benchmark campaign,
paper claim, dissertation claim, or evidence that planner performance changed.

## Live Audit Inputs

The full issue thread was read on 2026-07-06. The latest maintainer gate comment on the
issue, created at 2026-07-05T04:13:05Z, kept the issue open for:

- real official/user-supplied ETH asset staging;
- converter input/output SHA-256 evidence;
- smoke-validated generated SVG commit.

Open pull request dedupe found no open pull request covering issue #4546 or the exact
SocNavBench ETH closure-audit scope.

## Acceptance Criteria Evidence

| Criterion | Status | Evidence |
| --- | --- | --- |
| Official/user-supplied ETH traversible staged outside git. | **Met** | With `ROBOT_SF_EXTERNAL_DATA_ROOT` set to the external-data root, `uv run python scripts/tools/manage_external_data.py --json check socnavbench-s3dis-eth` exits 0 with `status=available`, `source_path=$ROBOT_SF_EXTERNAL_DATA_ROOT/socnavbench`, and both required ETH paths matched. |
| Converter run succeeds and records input/output hashes. | **Met** | `stage_socnavbench_eth_traversible_svg.py` exits 0 on the external root. Source `data.pkl` SHA-256: `02d450ee57bdab7aa357457085b3ddac6501ea8ec99324f2f02aecaed82562a6`. Output SVG SHA-256: `9fb9e126fac37b1c24c8849aeee47dfcccc5ef71fd7fc4e0fea7f78f19d1858d`. |
| Generated SVG parser + map smoke validation green. | **Met** | Wrapper report `status=ready`, `conversion_ready=true`, `smoke_ready=true`; parser saw 377 obstacles, one robot route, one pedestrian route, robot spawn/goal zones, and pedestrian spawn/goal zones. |
| Runbook states whether SVG committed or regenerated on demand. | **Met** | PR #4574 updated `docs/socnav_assets_setup.md`: commit `socnavbench_eth.svg` only after official/user-supplied source succeeds, input/output hashes are recorded, and parser smoke is green; never commit placeholder, dry-run, or fixture output. |
| Public absent-source behavior remains fail-closed / skip-if-absent. | **Met** | PR #4553 added `scripts/tools/stage_socnavbench_eth_traversible_svg.py` fail-closed statuses. A default-root audit run without the external data root exited 2 with `status=blocked_missing_mesh` and wrote no placeholder SVG. |
| No raw licensed SocNavBench/S3DIS data committed. | **Met** | The source mesh directory and traversible pickle remain under `$ROBOT_SF_EXTERNAL_DATA_ROOT/socnavbench`; this slice commits only the derived SVG plus compact Markdown/YAML evidence. |

## Conversion Evidence

This closure-audit note is the durable compact evidence surface for the conversion.

Generated derived artifact:

- `maps/svg_maps/socnavbench/socnavbench_eth.svg`

Ignored JSON reports from the run are local validation scratch under `output/maps/` and are
not durable dependencies. The source and output SHA-256 values above are the durable
provenance summary.

## Current Host Validation Commands

Commands ran in isolated worktree `issue-4546-closure-audit-20260706` from `origin/main`,
with `ROBOT_SF_EXTERNAL_DATA_ROOT` set to the private external-data root.

| Command | Exit | Result |
| --- | ---: | --- |
| `uv run python scripts/tools/manage_external_data.py --json check socnavbench-s3dis-eth` | 0 | `status=available`; ETH mesh and `traversibles/ETH/data.pkl` matched outside git. |
| `uv run python scripts/tools/validate_socnav_map_batch.py --socnav-root $ROBOT_SF_EXTERNAL_DATA_ROOT/socnavbench --batch-id eth_first --preflight --report-json output/maps/issue_4546_eth_first_preflight.json` | 0 | `status=ready`; `conversion_ready=true`; no missing required paths. |
| `uv run python scripts/tools/stage_socnavbench_eth_traversible_svg.py --socnav-root $ROBOT_SF_EXTERNAL_DATA_ROOT/socnavbench --output-svg maps/svg_maps/socnavbench/socnavbench_eth.svg --report-json output/maps/issue_4546_socnavbench_eth_stage_svg_smoke.json` | 0 | `status=ready`; `conversion_ready=true`; `smoke_ready=true`; generated SVG written. |

## Closure Boundary

This slice satisfies the issue #4546 acceptance criteria and should use `Closes #4546` in
the pull request body. It does not run a full benchmark campaign, submit Slurm/GPU work, or
edit paper/dissertation claim text.
