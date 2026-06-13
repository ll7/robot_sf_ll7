# Issue #2751 Topology Reselection Runtime Evidence

Issue: [#2751](https://github.com/ll7/robot_sf_ll7/issues/2751)

This bundle preserves the compact result of the clearance-targeted topology-reselection successor
runtime launched from the Issue #2742 packet. It is diagnostic-only evidence, not benchmark or
paper-facing proof.

## Result

- Classification: `revise`
- Runtime rows: 20
- Hard slices: `bottleneck_transfer`, `doorway_transfer`, `t_intersection_transfer`
- Negative control: `simple_negative_control`
- Outcome: all hard-slice rows completed diagnostics but remained `horizon_exhausted`; all
  negative-control rows succeeded with zero topology switching.

## Reproduction

```bash
uv run python scripts/validation/run_topology_reselection_cross_slice.py \
  --manifest configs/policy_search/topology_reselection_cross_slice_issue_2742.yaml \
  --output-dir <ignored-worktree-output>/diagnostics/issue_2751_topology_reselection_runtime
```

Runtime proof was collected on commit `952eff7e2f35bfe29fd65d90c7c43fa458ab8bb9`.
The raw per-row trace files are reproducible from the tracked command and are not mirrored here.
The compact row metrics in [summary.json](summary.json) and [report.md](report.md) preserve the
durable decision evidence.

## Files

- [summary.json](summary.json): compact provenance, decision, and row summary.
- [report.md](report.md): generated cross-slice decision table.
