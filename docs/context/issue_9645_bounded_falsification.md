# Bounded falsification pilot

The pilot comparison is persisted in
`docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/pilot_comparison.json`.
Its report can be regenerated from that JSON without running a search, replay, or simulator:

```bash
uv run python scripts/tools/compare_adversarial_samplers.py \
  --repo-root . \
  --render-existing-json path/to/pilot_comparison.json \
  --render-execution-mode empirical \
  --out-md path/to/pilot_comparison.md \
  --render-provenance-json path/to/pilot_comparison_render_provenance.json
```

The execution mode is an explicit caller declaration. The renderer validates the stored comparison,
preserves its recorded candidate outcomes, and writes the output and renderer provenance; it does
not infer empirical execution from the input or repeat search and simulation. The provenance records
the source and rendered report digests, renderer revision, and `search_or_simulation_rerun: false`.

For the pilot, the bounded Random/TPE comparison recorded no attributed critical failure in four
runs at 16 evaluations each. The corrected #9646 report classifies severe-intrusion status as
unknown for all 64 candidates because that evidence is absent. This remains a finite-budget **NO-GO**
for scaling this fixed design in issue #9648; it does not establish that the scenario space has no
counterexample.

The #9646 report uses schema v3 and is regenerated in a clean evidence root from the byte-bound
comparison, four derived checksum-pinned manifests, and all 64 retained episode-record files. Those
episode records are byte-verified and analysis-eligible for the report's objective summaries. The
producer `analysis_eligibility.trace_present` field means an episode-record path was present; it does
not mean detailed traces were captured. Both detailed trace-capture flags are false for all 64
episodes, and the two matched Random/TPE seed pairs qualify for descriptive comparison. The records' embedded source revision is consistently `58e516aa4f69ff3098bf518199f483006589758c`.
Detailed step-level and planner-decision trace capture is disabled for all 64 records, and their
embedded scenario/map digests are null for all 64. The packet retains exact producer bytes under
`payload/source_episode_records/` and separate path-normalized report inputs under
`payload/path_normalized_episode_records/`, with SHA-256 and byte-count bindings for both. The exact
source bytes retain absolute producer route-override paths, which the convergence report does not
dereference. Candidate-specific route-override inputs are not all retained, so these records are not
portable replay bundles by themselves. Collision/severe-intrusion status remains `unknown` for every candidate because
severe-intrusion evidence is absent, and unknown criticality is separate from known critical and
known non-critical counts. The experiment commit is recorded separately in
`payload/run_metadata.json`. This report does not reinterpret the historical v1 objective score or
claim safety from objective eligibility.

The mixed distance columns in `payload/candidate_evaluations.csv` are defined in its sibling
`payload/metadata.json`: `distance_to_human_min_m` is center-to-center distance, and
`min_clearance_m` is surface clearance after subtracting both agents' radii. Entries in
`distance_convention_by_column` override the scalar `distance_convention` for their named columns;
the scalar applies where no column override is listed. Both values are simulation-derived
diagnostics, not physical measurements.
