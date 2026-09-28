# Release-row collision-count gate (#9855)

## Contract

The benchmark metric producer records `ped_collision_count`,
`obstacle_collision_count`, and `agent_collision_count` as nonnegative integer
episode counts. `total_collision_count` is their sum; `collisions` is an alias for
that total. The 0.0.8 release-row anomaly gate uses an explicit
`release_0_0_8` config and checks all five fields
on every execution-eligible row. It emits `collision_metric_inconsistent` for a missing,
nonfinite, noninteger, negative, or inconsistent value. The detector blocks the
gate even when the general unannotated-finding threshold is raised. Counts
must be nonnegative integers and are compared by exact integer equality; no
value is repaired or imputed. JSON integer values stay integers even beyond
`2**53`. Integral float values are accepted only through `2**53 - 1`, where
adjacent integers remain representable; larger floats block as ambiguous.

For a typed `EpisodeEventLedger.v2`, the gate checks that the ledger's recorded
collision metric value equals `metrics.total_collision_count`, its source names
that field, and its exact collision boolean matches `outcome.collision_event`.
The ledger value follows the same exact-integer rule.
An unsupported typed-ledger schema blocks 0.0.8 admission. The number of exact
contact-event records is **not** equated to sampled collision counts; they have
different collection semantics. A legacy partial ledger with no schema version
does not assert typed-ledger reconciliation, while metric arithmetic remains
mandatory for the explicit `release_0_0_8` contract.
Absent or unversioned ledgers are counted in
`missingness.typed_collision_ledger_unavailable` but do not fail this arithmetic
gate; the separate release provenance gate must establish ledger completeness.

The earlier `release_row_anomalies_v1.json` remains a historical diagnostic
configuration. Its `legacy_diagnostic` mode preserves the prior detector
registry and does not classify a missing historical component as a 0.0.8
candidate failure or as current-valid. A retrospective result under that
configuration cannot satisfy the 0.0.8 collision-count gate. The candidate
manifest and receipt must bind the new config bytes and report the selected
`collision_metric_contract` value; a missing or wrong config blocks release
admission even if a legacy diagnostic report has `gate.blocked=false`.

`configs/benchmarks/release_row_anomalies_0_0_8.template.json` is deliberately
**not an admission config**. It has no pedestrian-aware roster and declares
`collision_roster_status=unfrozen_template`, which always blocks the gate.
After #9751 freezes the 14-arm v4 roster, create a new versioned config with
`collision_roster_status=frozen`, `collision_expected_arm_count=14`, and the
exact baseline plus pedestrian-aware planner IDs. Strict mode compares that
configured set and arm count with the verified bundle's `source.planner_ids`;
any mismatch blocks. A stale v2/v3 hybrid roster therefore cannot silently
omit the v4 arms from the cohort check.

## Proof and custody boundary

The focused regression reproduces the VV-5 two-row control and doubled-total
mutation and checks missing components, typed-ledger reconciliation, and the
event-count distinction. This code repair is not an admitted 0.0.8 campaign.
Before release acceptance, run the frozen-config gate on every row of the pinned 0.0.8
candidate matrix, account for blocked or missing rows, and preserve its exact
source/config hashes and receipt. Any 0.0.7 retro-check is diagnostic only;
historical rows and release assets remain unchanged. #9855 stays open until that
candidate-wide check and independent exact-head review are complete.
