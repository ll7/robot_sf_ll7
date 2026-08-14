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

1. retain the highest-count verdict pool; if verdict counts tie, retain the
   uniquely weaker label by smallest `label_strength`;
2. retain the weakest-label candidates within that verdict (smallest
   `label_strength`);
3. choose the candidate closest to the median `primary_order`;
4. resolve an exact numeric tie by lexicographically smallest `seed_id`.

An input with no unique weaker label for a tied verdict count, mixed cells,
duplicate seed identities, missing fields, blank identities, or non-finite
numbers fails closed. Finite values outside the supported float range also fail
closed rather than raising during normalization. The returned `SelectionManifest` uses schema version
`trace_dossier_selector.v1`, contains no wall-clock fields, and records the
selection reason.

## Shared campaign representative-run rule

`robot_sf.research.representative_selection` is the single source of truth for
the *campaign-side* version of the same guarantee, used wherever one seed has
to stand in for a whole cell:

- `VERDICT_SEVERITY` — verdict labels ordered weakest-first;
- `verdict_label_strength(label)` — the label-to-number mapping that turns this
  verdict vocabulary into the `label_strength` the selector contract above
  expects;
- `majority_verdict(verdicts_or_counts)` — most common verdict, tie-breaking
  toward the weaker label (and, between two unrecognized labels, toward the
  lexicographically smaller one so the result never depends on input order);
- `RepresentativeCandidate` / `select_representative_index(candidates)` —
  majority-verdict pool, median primary order parameter, lower seed on an exact
  tie;
- `PRIMARY_ORDER_PARAMETER` / `primary_order_parameter(scenario)` — the order
  parameter that defines "median run" per scenario.

`robot_sf.research.emergent_phenomena_campaign`,
`scripts/validation/render_issue_5149_emergent_phenomena_videos.py`, and
`scripts/validation/build_issue_5149_emergent_phenomena_campaign.py` all call
into this module instead of carrying their own copies. The extraction is
behaviour-preserving: `tests/test_render_emergent_phenomena_videos.py` and
`tests/test_emergent_phenomena_campaign.py` re-derive the archived
`issue_5149_emergent_phenomena_multiseed_2026-08` bundle's four replay seeds
and six majority verdicts from its own `runs.jsonl` and require an exact match.

### Known divergence from `trace_dossier_selector.v1`

The two selectors agree on odd-sized pools and differ on even-sized pools:
`select_representative_index` takes the lower of the two middle runs *by rank*,
while `select_representative` takes whichever middle run is closest to the
interpolated median *value* and breaks the resulting tie on seed identity.
Reconciling them changes which run the archived exhibits point at, so it is a
domain decision rather than a refactor and is tracked separately. Do not
silently align one with the other.

## Evidence boundary

The selector chooses a representative for a future trace export or dossier
render. It does not compute a benchmark metric, establish a verdict, rank a
planner, or admit an artifact as benchmark or paper-facing evidence. The
selected row must still carry the source trace, release pin, cell metadata,
and checksum before downstream provenance or publication review.

## Deferred work

The remaining #7086 slice is intentionally separate: acquiring trace-capable
runs for campaign cells that have no trace yet. That slice needs its own
compute routing, authorization, and artifact-admission proof, and does not
follow from any of the tooling described here.
