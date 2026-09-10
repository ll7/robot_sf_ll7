# Typed simulator snapshot prototype

Status: preparation-only (`RW-03`/`RW-04`, cross-reference #7394). The durable
format is `simulator_typed_snapshot.v2`; v1 artifacts are rejected because the
group-state and complete-RNG requirements are not backward-compatible. This document
describes a typed continuation seam over the existing
`SimulatorCounterfactualModel`; it does not admit shortened benchmark execution,
replace the replay engine, or establish a scientific speed-up.

## Boundary and compatibility

Snapshots are captured at the existing pre-step/action boundary. `step_index`,
absolute world time, and remaining episode budget are stored separately from the
world state. `next_observation_ready=false` is explicit: the caller must compute
the next observation after restore using the same wrapper contract.

Before a destination simulator is mutated, the snapshot checks the map, config,
code-revision, planner, checkpoint (when present), platform, and timestep identity.
Actor counts/order, every restored numeric-array shape, and every restored numeric-array
dtype are checked as well.
Pedestrian group membership and its reverse lookup are serialized as integer-entry
lists so JSON object-key coercion cannot change force-relevant group state. A
typed snapshot is restorable only when both the global NumPy and stdlib `random`
streams were captured and the destination adapter also supports those streams;
RNG-incomplete captures fail closed before mutation.
Such a mismatch raises `SnapshotCompatibilityError`; malformed metadata, unknown schema
versions, missing arrays, descriptor drift, object dtypes, non-finite values, and
payload digest/size drift raise `SnapshotPayloadError`. If the live adapter fails
after beginning a restore, the previous adapter snapshot is restored before the
failure is reported.

The durable pair is a JSON metadata file and a compressed `.npz` numeric payload.
Loading uses `allow_pickle=False`. JSON values carry typed tuple and array references;
robot drive state and route-navigator fields are rebuilt into destination-owned
objects. Behavior and route keys use stable actor/behavior identities rather than
process-local object IDs.

## Supported subset

The first native subset is one-robot `SimulatorCounterfactualModel` state:

- PySocialForce state, headings, angular velocities, and pedestrian speed caps;
- pedestrian group membership and pedestrian-to-group reverse lookup, including
  synchronization of the backend force model;
- robot pose, differential-drive velocity, wheel integration fields, and route
  progress;
- single-pedestrian runtime fields and route-group waypoint state;
- global NumPy RNG, stdlib `random`, owned behavior generators when discoverable,
  and instantiated residual-adversary fields;
- pre-step clock and remaining budget.

Planner/controller hidden memory, sensor and observation history, and metric/event
accumulators remain unsupported. A snapshot in this subset is a conditional
continuation, not a full fresh-environment checkpoint. The complete field-by-field
inventory is [simulator_state_inventory.v1.json](simulator_state_inventory.v1.json).

## No-op proof contract

`compare_continuation_traces` compares every declared per-step field: state,
observation, decision, applied action, events, terminal status, metrics, and RNG.
It reports the first deterministic nested path and step that differs. Equal contact
time or terminal status alone is not a pass. The focused native test covers a same-
object durable round trip and a route-progress mutation; recurrent/history omissions
are visible as unsupported rather than silently treated as equivalent.

## Cost receipt

The focused doorway fixture records artifact sizes through `SnapshotArtifact`.
The current native fixture produced a 19,657-byte JSON metadata file and a 3,893-byte
compressed numeric payload in the local headless environment (7 pedestrians,
`float64` state arrays). These are engineering measurements, not a benchmark result;
size varies with actor count and state contents. End-to-end accounting must still
include parent generation, capture, selection, validation, load, and continuation:

```text
C_plain = K * C(T)
C_window = C_parent + C_selection + C_validation
            + K * (C_load + C(W))
```

No positive break-even or cross-planner transfer claim is made by this prototype.
