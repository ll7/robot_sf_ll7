# Falsification search convergence report

The report command reads a persisted `adversarial-sampler-comparison.v3` index and the
`adversarial-search-manifest.v1` files named by its rows. It does not run a search, replay, or
simulator. The comparison index supplies sampler, seed, budget, objective, and manifest path; the
search manifests supply ordered candidate outcomes.

```bash
uv run python scripts/tools/report_falsification_search.py \
  --comparison /path/to/adversarial-sampler-comparison.json \
  --output-dir output/falsification-report \
  --repo-root .
```

The command writes `falsification_report.json`, `falsification_report.md`, and one static PNG per
objective under report schema `adversarial-search-convergence-report.v2`. The JSON retains every
per-candidate outcome and evaluation order, both raw and
analysis-eligible best-so-far values, budget membership, artifact/config digests, source revision
status, per-run counts, and matched Random-versus-TPE seed summaries. It preserves execution mode,
readiness status, availability status, and eligibility as separate candidate fields. The Markdown
table distinguishes raw observations from budget-bounded analysis-eligible summaries. It labels
sampler `optuna` as TPE, matching the existing comparison runner's sampler choice.

Budget-limited accounting uses only the first B candidate rows, where B is the comparison-indexed
budget. A valid candidate count within B is `candidate rows within B - invalid candidates - failed
evaluations`; scoreless valid evaluations remain valid. Every over-budget row remains in the JSON
audit history, with its budget flag and observed score, but cannot change budget-limited best-so-far,
critical counts, aggregates, or Random/TPE deltas. The current search runner's legacy
`summary.num_valid_candidates` subtracts invalid rows but not evaluator failures. The report preserves
that legacy value and emits a warning when it differs from row-derived full-manifest accounting.
Invalid, failed, scoreless, duplicate, missing, fallback, degraded, analysis-ineligible, and
over-budget attempts remain represented in the machine-readable output. Missing manifest rows are
shown as unavailable runs with their expected budget rather than silently removed. Per-run and
aggregate accounting distinguish missing evaluations inside the comparison-indexed budget from all
missing slots across the larger of the comparison and manifest budgets; an index/manifest budget
mismatch therefore cannot inflate the within-budget missing count.

Search curves maximize the objective recorded by the runner and carry the prior best score across
invalid, failed, or scoreless attempts. Solid curves summarize only candidates with explicit
`analysis_eligibility.eligible=true` that agrees with the canonical scored, native-execution,
trace-path, and effective-scenario-hash requirements, with no fallback/degraded execution; dashed
curves show all observed scored outcomes. Both curves stop at the comparison-indexed budget, and missing budget
slots remain explicit rather than being drawn as completed evaluations. A budget panel without
scored observations is labeled, so an empty curve is not mistaken for a zero-valued result. Runs
whose comparison row disagrees with manifest objective, seed, or budget are retained but excluded from aggregates. A
Random/TPE seed pair additionally requires the same normalized scenario/search/planner configuration
fingerprint, eligible scores from both runs, and no missing within-budget evaluations; exclusions
retain per-method reason codes and missing-slot counts in JSON. Invalid and failed attempts remain
accounted as attempted slots, while unrecorded within-budget slots cannot contribute to a matched
best-of-budget delta. Runtime is
read only from an explicit runtime field; it remains `null` / “Not recorded” when the source
artifacts do not contain search-level duration. Planner-step runtime and file timestamps are not
substitutes for search runtime.

Source revision is taken from an `adversarial_execution_context.v1` sidecar when present, otherwise
from candidate episode provenance. An identifier is exact only when it is a full 40-character
hexadecimal commit SHA; a context sidecar must also declare the supported schema. Unverified tokens
remain visible as unverified identifiers while revision status stays unknown or partial. The report
records the comparison and search-manifest SHA-256 digests and hashes accessible config files.
Fixture-based output demonstrates report behavior only; it is not evidence that a planner is safe
or that the adversary exhaustively searched its space.

All seed aggregates and observed min/max ranges are descriptive, not confidence intervals. No
inferential test is performed. With a small pilot, do not overinterpret differences between Random
and TPE. “No counterexample found” means only that the stated method found none under the recorded
finite budget.
