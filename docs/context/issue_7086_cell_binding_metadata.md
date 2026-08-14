# Issue #7086: Trace Dossier Cell-Binding Metadata

This slice adds a small metadata contract for binding one future trace dossier
to one campaign cell. It lets a dossier manifest name the campaign cell,
selected trace identity, terminal verdict counts, and selected verdict
denominator mechanically.

## Contract

`robot_sf.benchmark.trace_dossier_cell_binding.build_trace_dossier_cell_binding`
returns a frozen `trace_dossier_cell_binding.v1` block with:

- stable campaign-cell identity;
- selected trace episode, seed, artifact URI, checksum, and terminal verdict;
- sorted terminal verdict counts;
- total cell episode count;
- selected verdict count;
- an explicit metadata-only evidence boundary.

Malformed identities, non-integer seeds, missing or uppercase trace checksums,
empty counts, zero totals, and selected verdicts absent from the cell counts
fail closed.

## Boundary

This is provenance metadata only. It does not run a simulator, read or validate
trace files, render a panel, compute metrics, rank planners, admit benchmark
evidence, or make paper-facing claims. The remaining #7086 work still needs an
actual pinned trace export, renderer integration, artifact checksums, and
separate review before any dossier can support a scientific statement.
