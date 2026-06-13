# Issue #2742 Topology Reselection Successor Launch Packet

Issue: [#2742](https://github.com/ll7/robot_sf_ll7/issues/2742)

Status: current, launch-packet evidence only.

## Claim Boundary

This note defines the next clearance-targeted topology-reselection successor after the
cross-slice `revise` result from
[Issue #2716](issue_2716_topology_reselection_cross_slice.md). It is not benchmark-strengthening
or paper-facing evidence. The tracked output is a reproducible launch packet: a manifest, runner
generalization, tests, and dry-run command expansion.

Compact evidence:

- [summary.json](evidence/issue_2742_topology_reselection_successor/summary.json)

## Hypothesis

Progress-gated topology reselection can clear the non-canonical hard slices that remained
`horizon_exhausted` in Issue #2716.

## Comparator

The successor keeps the #2716 comparator set:

- `topology_guided_hybrid_rule_v0`
- `topology_guided_hybrid_rule_v0_reuse_penalty`

The candidate under test remains
`topology_guided_hybrid_rule_v0_progress_gated_reselection`, with threshold-specific rows for
`0.05`, `0.1`, and `0.2` meters.

## Manifest

The versioned manifest is
`configs/policy_search/topology_reselection_cross_slice_issue_2742.yaml`.

It reuses the #2716 hard slices:

- `classic_bottleneck_medium`
- `classic_doorway_medium`
- `classic_t_intersection_medium`

It also preserves the negative control:

- `empty_map_8_directions_east`

## Decision Rule

Promote-style interpretation requires actual hard-slice clearance: every hard slice must have at
least one progress-gated terminal success, and the negative-control slice must show zero topology
switching.

Classify as `stop` or `revise` if progress-gated reselection loses topology-command influence on
any hard slice, worsens negative-control switching, or only changes route labels without
route-progress or terminal-outcome signal.

## Validation

```bash
uv run ruff check scripts/validation/run_topology_reselection_cross_slice.py \
  tests/validation/test_run_topology_reselection_cross_slice.py
uv run ruff format --check scripts/validation/run_topology_reselection_cross_slice.py \
  tests/validation/test_run_topology_reselection_cross_slice.py
uv run pytest tests/validation/test_run_topology_reselection_cross_slice.py -v
uv run python scripts/validation/run_topology_reselection_cross_slice.py \
  --manifest configs/policy_search/topology_reselection_cross_slice_issue_2742.yaml \
  --dry-run \
  --output-dir output/diagnostics/issue_2742_topology_reselection_cross_slice
```

Observed result:

```yaml
classification: launch_packet
dry_run_classification: dry_run
dry_run_command_count: 20
runtime_rows: 0
```

The dry-run output under `output/` is disposable. The summary JSON above is the durable tracked
record for this launch packet.

## Issue #2751 Runtime Decision (2026-06-13)

Issue [#2751](https://github.com/ll7/robot_sf_ll7/issues/2751) executed the clearance-targeted
successor manifest on commit `952eff7e2f35bfe29fd65d90c7c43fa458ab8bb9`.

Durable evidence:

- [runtime summary](evidence/issue_2751_topology_reselection_runtime/summary.json)
- [runtime report](evidence/issue_2751_topology_reselection_runtime/report.md)

Observed result:

```yaml
classification: revise
runtime_rows: 20
hard_slices:
  bottleneck_transfer: horizon_exhausted
  doorway_transfer: horizon_exhausted
  t_intersection_transfer: horizon_exhausted
negative_control:
  simple_negative_control: success_zero_switching
```

The successor did not satisfy the promotion rule because no hard slice achieved terminal success.
The negative-control rows succeeded without topology switching, so the result does not trigger the
negative-control stop condition. The outcome remains diagnostic-only and not benchmark or
paper-facing evidence.
