# Issue #4546 Closure Audit

Date: 2026-07-06

Issue: <https://github.com/ll7/robot_sf_ll7/issues/4546>

## Plain-Language Summary

Issue #4546 is not ready to close. Merged PRs #4553 and #4574 added the Social
Navigation Benchmark (SocNavBench) ETH wrapper and the runbook policy for the
derived SVG, but the licensed Stanford 3D Indoor Spaces (S3DIS) ETH source
assets are not staged on this host. The real
`maps/svg_maps/socnavbench/socnavbench_eth.svg` artifact therefore cannot be
generated, hashed, or smoke-tested here.

This is a closure-audit evidence artifact. It is not a benchmark run, paper
claim, dissertation claim, or proof that the real generated SVG exists.

## Live Audit Inputs

The full issue thread was read on 2026-07-06. The latest maintainer gate comment
was created at 2026-07-05T04:13:05Z and states that PR #4574 merged the
derived-SVG artifact policy, while the remaining work is blocked on
license-gated source data:

- real official/user-supplied ETH asset staging;
- converter run input and output SHA-256 evidence;
- smoke-validated generated SVG commit.

Open pull request dedupe found no open PR covering issue #4546 or this exact
SocNavBench ETH closure-audit scope.

## Acceptance Criteria Evidence

| Criterion | Status | Evidence |
| --- | --- | --- |
| Official/user-supplied ETH traversible staged outside git. | **Not met on host** | `manage_external_data.py --json check socnavbench-s3dis-eth` exits 2 in the default worktree root with missing `sd3dis/stanford_building_parser_dataset/mesh/ETH` and `sd3dis/stanford_building_parser_dataset/traversibles/ETH/data.pkl`. The `$HOME/external_data/socnavbench` root is also missing. |
| Converter run succeeds and records input/output hashes. | **Not met** | The stage/generate/convert/smoke wrapper exits 2 with `status=blocked_missing_mesh` before conversion. No real input `data.pkl` SHA-256 or output SVG SHA-256 can be recorded on this host. |
| Generated SVG parser and map smoke validation are green. | **Not met** | No real SVG was generated because the required ETH mesh is absent. The wrapper reports `conversion_ready=false` and `smoke_ready=false`. |
| Runbook states whether the SVG is committed or regenerated on demand. | **Met** | PR #4574 updated `docs/socnav_assets_setup.md`: commit `maps/svg_maps/socnavbench/socnavbench_eth.svg` only after official/user-supplied source succeeds, records input/output SHA-256, and passes parser smoke validation. Placeholder, dry-run, and fixture output must not be committed. |
| Public absent-source behavior remains fail-closed / skip-if-absent. | **Met** | PR #4553 added `scripts/tools/stage_socnavbench_eth_traversible_svg.py` with fail-closed statuses. Current host validation confirms missing assets return exit 2 instead of writing placeholder output. |
| No raw licensed SocNavBench/S3DIS data committed. | **Met** | PRs #4553 and #4574 did not add raw data. This audit adds only Markdown/YAML state and no generated dataset bytes. |

## Current Host Validation

Commands were run from isolated worktree
`issue-4546-closure-audit-verify-acceptance-criteria-of-4546-against-merged-prs-clos`
at commit `b8eb941f2`.

| Command | Exit | Result |
| --- | ---: | --- |
| `scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/manage_external_data.py --json check socnavbench-s3dis-eth` | 2 | `status=incomplete`; missing ETH mesh and `traversibles/ETH/data.pkl` under the default worktree-local SocNavBench root. |
| `scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/validate_socnav_map_batch.py --batch-id eth_first --preflight --report-json <report-json>` | 2 | `status=blocked_pending_source_assets`; `conversion_ready=false`. |
| `scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/stage_socnavbench_eth_traversible_svg.py --report-json <report-json> --output-svg <scratch-svg>` | 2 | `status=blocked_missing_mesh`; `conversion_ready=false`; `smoke_ready=false`; no SVG committed. |
| `ROBOT_SF_EXTERNAL_DATA_ROOT=$HOME/external_data scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/manage_external_data.py --json check socnavbench-s3dis-eth` | 2 | `status=missing` for `$HOME/external_data/socnavbench`. |
| `ROBOT_SF_EXTERNAL_DATA_ROOT=$HOME/external_data scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/validate_socnav_map_batch.py --socnav-root $HOME/external_data/socnavbench --batch-id eth_first --preflight --report-json <report-json>` | 2 | `status=blocked_pending_source_assets`; `conversion_ready=false`. |
| `ROBOT_SF_EXTERNAL_DATA_ROOT=$HOME/external_data scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/stage_socnavbench_eth_traversible_svg.py --socnav-root $HOME/external_data/socnavbench --report-json <report-json> --output-svg <scratch-svg>` | 2 | `status=blocked_missing_mesh`; next action names `$HOME/external_data/socnavbench/sd3dis/stanford_building_parser_dataset/mesh/ETH`. |

## Next Empirical Action

Stage the official or user-supplied SocNavBench ETH mesh and traversible under
the external-data root, then rerun the full wrapper:

```bash
uv run python scripts/tools/stage_socnavbench_eth_traversible_svg.py \
  --socnav-root /path/to/socnavbench \
  --output-svg maps/svg_maps/socnavbench/socnavbench_eth.svg \
  --report-json <report-json>
```

If the wrapper succeeds on real source data, record the input `data.pkl`
SHA-256, output SVG SHA-256, parser smoke result, and commit only the generated
SVG plus compact evidence. Until then, #4546 should stay open as blocked on
licensed external input.
