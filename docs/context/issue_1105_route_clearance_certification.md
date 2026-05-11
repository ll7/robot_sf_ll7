# Issue 1105 Route-Clearance Certification

Issue: [#1105](https://github.com/ll7/robot_sf_ll7/issues/1105)

## Goal

Resolve the attribution blocker from the 18 route-clearance warnings present in the paper matrix
and h500 preflights without hiding the warnings or changing planner behavior.

## Decision

The paper-facing benchmark configs now point to
`configs/benchmarks/route_clearance_certifications_v1.yaml`. Preflight artifacts still emit
`route_clearance_warnings`, but each known warning row also carries:

* `certification_status`
* `certification_claim_scope`
* `certification_rationale`
* review metadata linking the decision to issue `1105`

The preflight payloads also include `route_clearance_warning_summary`, which reports the total
warning count, certified count, unresolved count, status counts, and unresolved scenario IDs.

## Classification

The three negative-clearance scenarios remain runnable but are excluded from planner-mechanism
attribution until their route or map geometry is repaired:

* `classic_merging_low`
* `classic_merging_medium`
* `classic_station_platform_medium`

The remaining 15 warnings are certified as intentional stress geometry. They are benchmark-ready
only with a caveat: failures on those rows should be interpreted as performance under tight route
geometry, not as uncaveated planner mechanism evidence.

## Why Certification Instead Of Repair

The warning set mixes three negative route/obstacle overlaps with several intentional bottleneck,
doorway, elevator, and low-margin corridor stress cases. Repairing all low-margin stress cases would
change the benchmark semantics. This change keeps the geometry intact, makes the claim boundary
machine-readable in preflight output, and leaves the negative-overlap cases clearly caveated.

## Validation

Expected proof for this issue:

```bash
rtk uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1.yaml \
  --mode preflight \
  --campaign-id issue1105_paper_preflight \
  --log-level WARNING

rtk uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --mode preflight \
  --campaign-id issue1105_h500_preflight \
  --log-level WARNING

jq -e '.route_clearance_warning_count == 18 and
       .route_clearance_warning_summary.unresolved_warning_count == 0'
```

The generated `output/benchmarks/...` preflight directories are reproducible local evidence and
should stay ignored. The durable source of truth is the tracked certification registry plus this
context note.
