# Issue #8829 compute-window capacity planner

The `compute_window_capacity_plan.v1` helper ranks workload metadata for a finite
compute window. It is operational planning infrastructure: it does not submit
jobs, mutate GitHub or scheduler state, or establish benchmark/scientific
evidence.

## Contract

Input packets must declare a fresh inventory, live admission, timezone-aware
access deadline, per-lane capacities and queue limits, storage/transfer limits,
a fresh dependency graph, and workload rows. Rows must be ready, fresh,
authorized, input-complete, not duplicated, and not already running. Missing or
malformed estimates make a row ineligible; no optimistic defaults are used.

The score exposes all eight declared components: time-to-loss, expected wall
time, queue uncertainty, resource footprint, transfer volume, completion
probability, unlock value, and later executability. Components are operational
heuristics, not evidence quality. Expected and upper values remain visible in
the row uncertainty report.

After hard gates, the planner greedily packs dependency closures subject to
per-lane queue limits, aggregate resource capacity, and storage/transfer
capacity. It emits score, deadline, and unlock orderings with deterministic
workload-ID tie-breaking. `--check` prints JSON or text without creating the
output path; a blocked or incomplete plan exits 2.

The fixture at
`tests/fixtures/compute_window_capacity_inventory.json` covers dependency
ordering, blocked/running rows, deadline and storage infeasibility, resource
conflict, uncertain queue bounds, and stable ties. This is contract proof only;
it is not a compute allocation decision or a research result.
