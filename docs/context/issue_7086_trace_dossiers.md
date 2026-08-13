# Issue #7086: Trace Dossier Representative Selection

This first slice adds a reusable selector for choosing one representative seed
or episode from a single campaign cell. It supports the later trace-export and
dossier-rendering work without fabricating trace data or changing benchmark
semantics.

## Selector contract

`robot_sf.benchmark.trace_dossier_selection.select_representative` accepts
candidate mappings (or validated `TraceDossierCandidate` values) with these
required fields:

- `cell_id`: stable campaign-cell identity;
- `verdict`: the recorded terminal/verdict label;
- `label_strength`: an explicit numeric label-strength tie-break value, where
  the smallest value is the weaker label;
- `primary_order`: the numeric primary order parameter used for representative
  proximity;
- `seed_id`: stable seed or episode identity.

The deterministic selection order is:

1. retain the unique majority-verdict pool;
2. retain the weakest-label candidates (smallest `label_strength`);
3. choose the candidate closest to the median `primary_order`;
4. resolve an exact numeric tie by lexicographically smallest `seed_id`.

An input with no unique majority verdict, mixed cells, duplicate seed
identities, missing fields, blank identities, or non-finite numbers fails
closed. The returned `SelectionManifest` uses schema version
`trace_dossier_selector.v1`, contains no wall-clock fields, and records the
selection reason.

## Evidence boundary

The selector chooses a representative for a future trace export or dossier
render. It does not compute a benchmark metric, establish a verdict, rank a
planner, or admit an artifact as benchmark or paper-facing evidence. The
selected row must still carry the source trace, release pin, cell metadata,
and checksum before downstream provenance or publication review.

## Deferred work

The remaining #7086 slices are intentionally separate:

- trace-capable export from an actual scenario/seed/planner/release pin;
- deterministic cell-binding metadata and verdict counts;
- the multi-panel trajectory, speed, clearance, and event-timeline renderer;
- compute routing and artifact admission for any newly materialized traces.

Those slices require their own runtime/provenance proof and do not follow from
this selector-only implementation.
