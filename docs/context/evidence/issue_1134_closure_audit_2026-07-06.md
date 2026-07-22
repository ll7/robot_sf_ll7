# Issue #1134 Closure Evidence

Current status: 2026-07-22
Issue: <https://github.com/ll7/robot_sf_ll7/issues/1134>

## Claim Boundary

This is **smoke evidence** that the committed SocNavBench ETH SVG loads through the scenario and
headless environment path. It is not benchmark, paper, or planner-performance evidence.

Conclusion: **ready for #1134 closure**. PR #4693 committed
`maps/svg_maps/socnavbench/socnavbench_eth.svg` with source and output SHA-256 provenance. PR #6118
adds the focused scenario proof: it resolves that configured committed map, verifies its exact
route/zone structure, creates a Robot SF environment, resets it, and executes headless steps. No
external data root, raw-asset staging, or GPU is required for this smoke proof.

The raw licensed source assets remain outside Git. The `eth_first` source preflight still correctly
fails closed on hosts where those inputs are absent; that historical host condition does not negate
the committed-SVG smoke evidence.

## Historical Audit (2026-07-06, superseded)

The material below is the original pre-conversion audit. Its `keep #1134 open`, unmet-criterion
rows, and next-action text are historical records only; they describe raw-source availability on
that host before PR #4693 committed the real ETH SVG. They must not be read as the current closure
status.

## Live Audit Inputs

Live issue thread read on 2026-07-06 recorded the latest maintainer gate comment at
2026-07-06T09:34:18Z:

- PR #4610 satisfies the representative benchmark-path criterion at the converter code-path level
  using a synthetic fixture.
- Remaining work: stage licensed official ETH `data.pkl`, generate real `socnavbench_eth.svg`, and
  record checksum/provenance on staged assets.

Open-PR dedupe found no open PR covering #1134 or the exact SocNavBench ETH closure-audit scope.

Fragmentation guard result: #4535 and #4610 both merged within the last 24 hours and were
progression slices, not merely guard-refresh PRs. This report is therefore a consolidation slice:
one criterion-to-evidence table, one current fail-closed host preflight, and one explicit next
empirical action.

## Historical Acceptance Criteria To Evidence

| Criterion | Status | Evidence |
| --- | --- | --- |
| `uv run python scripts/tools/validate_socnav_map_batch.py --batch-id eth_first` passes on staged source assets. | **Not met on this host** | Default worktree preflight exits 2 with `status=blocked_pending_source_assets`; explicit `$HOME/external_data/socnavbench` preflight also exits 2 with both `eth_mesh_dir` and `eth_traversible_pickle` missing. PR #3771 added the fail-closed preflight; #1498/#4189 recorded a prior host-local staged-ready state, but the asset is not present on this host. |
| Source checksums/provenance recorded in context note successor note. | **Partially met, not sufficient for #1134 closure** | `docs/context/issue_1498_state.yaml` records a 2026-07-02 generated `data.pkl` path and SHA-256 from a prior host-local state. Current host validation cannot reproduce that state, so #1134 still needs fresh source checksum/provenance tied to the actual conversion run. |
| `maps/svg_maps/socnavbench/socnavbench_eth.svg` generated from official source assets and small/reviewable enough for git. | **Not met** | PR #4535 added the converter, PR #4553 added the stage/generate/convert/smoke wrapper, and PR #4574 documented the artifact policy. No real `socnavbench_eth.svg` is committed because current host source assets are absent and placeholder/fixture output is forbidden. |
| Converted SVG passes repository SVG/parser validation. | **Met for converter fixture only; not met for real ETH SVG** | PR #4535 added `test_fixture_traversible_converts_to_parser_valid_svg`. Real ETH SVG validation remains blocked until the real SVG exists. |
| Route/zone semantics explicit and non-overlapping. | **Met for converter fixture only; not met for real ETH SVG** | PR #4535 converter emits explicit robot/pedestrian zones and routes; fixture parser inspection asserts runtime routes and zones. Real ETH route/zone semantics still need validation on the generated map. |
| A smoke scenario exercises map through representative benchmark path. | **Met for converter code path; not met for real ETH map artifact** | PR #4610 added `test_converted_fixture_map_runs_headless_env_smoke`, proving converter output loads through `make_robot_env`, `reset`, and headless steps on a synthetic traversible fixture. Real ETH map smoke remains blocked until source assets and generated SVG exist. |
| Raw SocNavBench/S3DIS assets remain untracked. | **Met** | PRs #3771, #4535, #4553, #4574, and #4610 did not commit raw SocNavBench/S3DIS bytes. This audit adds only Markdown/YAML evidence. |

## Historical Host Validation

Commands run from the isolated issue #1134 worktree at commit `c29158050`:

```bash
scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/tools/manage_external_data.py --json check socnavbench-s3dis-eth
# exit 2; status=incomplete

scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/tools/validate_socnav_map_batch.py \
    --batch-id eth_first \
    --preflight \
    --report-json .git/codex-agent-runs/active/issue-1134/eth_first_preflight_default.json
# exit 2; status=blocked_pending_source_assets; conversion_ready=false

ROBOT_SF_EXTERNAL_DATA_ROOT=$HOME/external_data \
scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/tools/manage_external_data.py --json check socnavbench-s3dis-eth
# exit 2; status=missing

scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/tools/validate_socnav_map_batch.py \
    --socnav-root $HOME/external_data/socnavbench \
    --batch-id eth_first \
    --preflight \
    --report-json .git/codex-agent-runs/active/issue-1134/eth_first_preflight_external_root.json
# exit 2; status=blocked_pending_source_assets; conversion_ready=false
```

The explicit external-root preflight reports missing:

- `sd3dis/stanford_building_parser_dataset/mesh/ETH`
- `sd3dis/stanford_building_parser_dataset/traversibles/ETH/data.pkl`

## Historical Next Empirical Action

Stage or hydrate the real SocNavBench ETH mesh/traversible under the external-data root on the
execution host, rerun `validate_socnav_map_batch.py --batch-id eth_first --preflight`, then run
`scripts/tools/stage_socnavbench_eth_traversible_svg.py` without `--dry-run` to generate
`maps/svg_maps/socnavbench/socnavbench_eth.svg`, record input/output SHA-256 values, and validate
the generated SVG with parser and smoke checks. Until that succeeds, #1134 should remain open as
blocked external input.
