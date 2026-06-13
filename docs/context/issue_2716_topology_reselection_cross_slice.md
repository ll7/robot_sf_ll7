# Issue #2716 Topology Reselection Cross-Slice Diagnostic

Issue: [#2716](https://github.com/ll7/robot_sf_ll7/issues/2716)

Status: current, diagnostic-only evidence.

## Claim Boundary

This note records a cross-slice diagnostic for
`topology_guided_hybrid_rule_v0_progress_gated_reselection`. It does not promote the planner,
establish benchmark evidence, or make a paper-facing claim. The purpose is to decide whether the
progress-gated topology successor from [#2704](issue_2704_progress_gated_topology_successor.md)
generalizes beyond the canonical h160 double-bottleneck slice.

Compact evidence:

- [summary.json](evidence/issue_2716_topology_reselection_cross_slice/summary.json)
- [report.md](evidence/issue_2716_topology_reselection_cross_slice/report.md)

## Manifest

The versioned manifest is
`configs/policy_search/topology_reselection_cross_slice_issue_2716.yaml`.

It predeclares:

- three non-canonical hard slices:
  `classic_bottleneck_medium`, `classic_doorway_medium`, and
  `classic_t_intersection_medium`;
- one negative-control slice: `empty_map_8_directions_east`;
- candidates:
  `topology_guided_hybrid_rule_v0`,
  `topology_guided_hybrid_rule_v0_reuse_penalty`, and
  `topology_guided_hybrid_rule_v0_progress_gated_reselection`;
- progress-gate thresholds: `0.05`, `0.1`, and `0.2` meters.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python \
  scripts/validation/run_topology_reselection_cross_slice.py \
  --output-dir <local-diagnostic-output-dir>
```

The script produced 20 `diagnostic_complete` rows in the local artifact directory. Raw traces remain
worktree-local disposable artifacts; the compact tracked evidence above preserves the reviewable
result.

## Result

```yaml
classification: revise
claim_boundary: diagnostic_only_not_benchmark_or_paper_evidence
row_count: 20
all_rows_diagnostic_complete: true
hard_progress_gated_terminal_outcomes:
  horizon_exhausted: 9
negative_control_progress_gated_terminal_outcomes:
  success: 3
negative_control_progress_gated_switch_count_max: 0
hard_progress_gated_collision_rate_max: 0.0
```

Interpretation:

- The progress-gated candidate activated across all selected hard slices.
- The negative-control slice stayed clean: all progress-gated rows succeeded and had zero topology
  switch count.
- No hard progress-gated row cleared by h160. All hard progress-gated rows remained
  `horizon_exhausted`.

## Decision

Classify progress-gated reselection as `revise`, not `promote`.

The mechanism appears safe on the selected negative control and still exposes useful diagnostic
signals, but it does not close the hard-slice blocker. A successor should target actual clearance
or terminal-outcome movement rather than another single-slice threshold chase.

## Validation

```bash
uv run pytest tests/validation/test_run_topology_reselection_cross_slice.py -q
uv run ruff check scripts/validation/run_topology_reselection_cross_slice.py \
  tests/validation/test_run_topology_reselection_cross_slice.py
uv run ruff format --check scripts/validation/run_topology_reselection_cross_slice.py \
  tests/validation/test_run_topology_reselection_cross_slice.py
git diff --check
```
