# Issue #2752 Topology Reselection Mechanism Diagnosis (2026-06-13)

Issue: [#2752](https://github.com/ll7/robot_sf_ll7/issues/2752)

This bundle preserves the mechanism-level failure classification of three hard slices from
Issue #2751 runtime evidence. It is analysis-only evidence, not benchmark or paper-facing proof.

## Result

- Classification: `analysis_only`
- Hard slices diagnosed: `bottleneck_transfer`, `doorway_transfer`, `t_intersection_transfer`
- Failure labels:
  - `bottleneck_transfer`: `no_useful_topology_alternative` (medium confidence)
  - `doorway_transfer`: `no_useful_topology_alternative` (high confidence)
  - `t_intersection_transfer`: `candidate_route_blocked` (medium confidence)
- Outcome: all hard slices show no topology alternative produced clearance; two slices are
  likely scenario/geometry insufficiency, one is ambiguous between blocked geometry and
  excessive switching.

## Source Evidence

Runtime evidence from Issue #2751:
- `docs/context/evidence/issue_2751_topology_reselection_runtime/summary.json`
- `docs/context/evidence/issue_2751_topology_reselection_runtime/report.md`

## Reproduction

```bash
uv run python scripts/validation/run_topology_reselection_cross_slice.py \
  --manifest configs/policy_search/topology_reselection_cross_slice_issue_2742.yaml \
  --output-dir <ignored-worktree-output>/diagnostics/issue_2751_topology_reselection_runtime
```

Mechanism classification was performed manually from the durable `summary.json` and scout analysis.
Runtime proof was collected on commit `952eff7e2f35bfe29fd65d90c7c43fa458ab8bb9`.

## Files

- [summary.json](summary.json): compact provenance, per-slice failure labels, confidence, and
  evidence fields.
- [report.md](report.md): human-readable decision table, caveats, and diagnostic gaps.
