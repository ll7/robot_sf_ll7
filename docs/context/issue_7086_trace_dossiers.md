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

The two selectors are intentionally distinct contracts, not accidental duplicate
implementations. Issue #7131 records the following four decisions:

1. **Median convention:** keep the campaign selector's lower-of-two-middle
   choice *by rank* for an even pool because it is compatibility-pinned to the
   archived exhibits. Keep `trace_dossier_selector.v1`'s nearest candidate to
   the interpolated median value because it is a separately versioned,
   provenance-neutral contract. The selectors therefore agree on odd-sized
   pools but can choose different runs on even-sized pools.
2. **Seed ordering:** keep numeric seed ordering in
   `select_representative_index`; its input is an integer and the archived
   campaign behavior is pinned. Keep lexicographic `seed_id` ordering in
   `select_representative`; `seed_id` is an opaque stable identity, not a
   number to reinterpret.
3. **Tied verdict labels:** do not change the campaign majority tie behavior to
   v1's fail-closed rule. The campaign has a declared total severity order and
   a deterministic lexicographic fallback for unknown labels. Replacing that
   behavior would alter the pinned campaign contract without resolving an
   ambiguity in its vocabulary. v1 continues to fail closed when tied verdicts
   have no unique weaker `label_strength`.
4. **Archived exhibits:** do not regenerate the committed campaign bundle in
   this decision. Its exact replay-seed regression remains in
   `tests/test_render_emergent_phenomena_videos.py`; the bundle predates and
   continues under the pinned campaign convention. This decision performs no
   trace acquisition, benchmark run, evidence admission, or paper-facing
   claim.

Do not silently align one selector with the other: changing either convention
requires a new compatibility or schema decision and its own regression proof.

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
