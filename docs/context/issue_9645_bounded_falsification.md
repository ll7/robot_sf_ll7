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
embedded scenario/map digests are null for all 64. The packet retains separate path-normalized report inputs under
`payload/path_normalized_episode_records/` and withholds the exact producer bytes because they
contain absolute producer route-override paths; SHA-256 and byte-count bindings are recorded for both,
so the original bytes stay verifiable by anyone holding the private copy. Candidate-specific route-override inputs are not all retained, so these records are not
portable replay bundles by themselves. The exact copies were recovered from commit
`7588b785a607680400adcc6398d8c983c160e4bd`. The repository absolute-path guard has no exemption for
this packet. The normalized copies remain portable analysis inputs. Collision/severe-intrusion status remains
`unknown` for every candidate because severe-intrusion evidence is absent, and unknown criticality is
separate from known critical and
known non-critical counts. The experiment commit is recorded separately in
`payload/run_metadata.json`. This report does not reinterpret the historical v1 objective score or
claim safety from objective eligibility.

## Objective evidence limitation and versioned correction

The recorded campaign used the frozen `constraints_first_lexicographic_v1` objective at source
revision `58e516aa4f69ff3098bf518199f483006589758c`. An independent methodology review found that
v1 can return a negative safety composite when collision is observed false but severe-intrusion
evidence is absent. The 64 historical `0.0` scores remain recorded, but they cannot support a
non-critical safety tier; the report's `unknown` criticality classification is controlling for that
claim. The recorded source and results are not rewritten or re-scored.

The implementation now provides `constraints_first_lexicographic_v2` in
`robot_sf.adversarial.objectives_v2` for future searches. It uses
three-valued OR: known collision or intrusion evidence establishes a safety failure, both components
must be explicitly false to establish a negative safety result, and otherwise the objective returns
no score. Within each tier its soft degradation score uses near-miss count, SNQI, and path
inefficiency (`1 - path_efficiency`) when those metrics are available. Conflicts within one
component make only that component unknown; a confirmed positive in the other component still
establishes the safety-failure tier. A primary failure label cannot override missing or conflicting
episode outcome details; attribution-derived criticality stays unknown without corroborating
evidence. A score also requires resolved,
available execution provenance (`execution_mode` native, adapter, or mixed; `readiness_status`
native or adapter; and `availability_status` available). Fallback, degraded, unavailable, failed,
or missing execution status remains recorded but cannot steer an optimizer. The frozen v1
implementation remains available for exact reproduction of existing contracts. This code
correction does not authorize another campaign or change the current NO-GO for
scaling #9648. A future bounded pilot would need v2, complete intrusion metrics, a domain containing
known hard cases, and multiple simulator seeds before it could reconsider that gate.

The mixed distance columns in `payload/candidate_evaluations.csv` are defined in its sibling
`payload/metadata.json`: `distance_to_human_min_m` is center-to-center distance, and
`min_clearance_m` is surface clearance after subtracting both agents' radii. Entries in
`distance_convention_by_column` override the scalar `distance_convention` for their named columns;
the scalar applies where no column override is listed. Both values are simulation-derived
diagnostics, not physical measurements.
