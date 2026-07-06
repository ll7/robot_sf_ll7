# Issue #3463 Topology Reselection Cross-Slice Diagnostic

This compact evidence bundle records the executed CPU-only Issue #3463
topology-reselection cross-slice packet. The packet is diagnostic-only and does
not support a benchmark, planner-promotion, paper, or dissertation claim.

## Provenance

- Issue: [#3463](https://github.com/ll7/robot_sf_ll7/issues/3463)
- Commit: `0b0304176a3aa7d34d707e3b7850e27294c175db`
- Manifest: `configs/policy_search/topology_reselection_cross_slice_issue_3463.yaml`
- Claim boundary: `diagnostic_only_not_benchmark_or_paper_evidence`
- Command:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 \
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
scripts/validation/run_topology_reselection_cross_slice.py \
  --manifest configs/policy_search/topology_reselection_cross_slice_issue_3463.yaml \
  --output-dir <ignored-diagnostic-output-dir>
```

## Result

- Classification: `blocked`
- Rationale: at least one progress-gated row did not produce diagnostic evidence.
- Rows: 20 total; 15 `diagnostic_complete`, 5 `not_available`.
- Blocked slice: all `doorway_transfer` rows ended `not_available` after
  obstacle collision.

The raw ignored diagnostic output tree was about 19 MB and is intentionally not
mirrored here. This bundle preserves only the compact report files needed for
review.

## Files

- `summary.json`: machine-readable report with commands and row metrics.
- `report.md`: rendered decision table from the runner.
