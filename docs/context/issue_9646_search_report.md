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
objective. The JSON retains per-candidate outcomes and evaluation order, best-so-far values by
attempted evaluation index, artifact/config digests, source revision status, per-run counts, and
matched Random-versus-TPE seed summaries. It preserves execution mode, readiness status, and
availability status as separate candidate and run-level fields, and renders execution/availability
counts in the Markdown table. It labels sampler `optuna` as TPE, matching the existing comparison
runner's sampler choice.

Candidate accounting is derived from manifest rows. A valid candidate count is
`candidate rows - invalid candidates - failed evaluations`; scoreless valid evaluations remain
valid. The current search runner's legacy `summary.num_valid_candidates` subtracts invalid rows but
not evaluator failures. The report preserves that legacy value and emits a warning when it differs
from row-derived accounting. Invalid, failed, scoreless, duplicate, missing, fallback, and degraded
attempts remain represented in the machine-readable output. Missing manifest rows are shown as
unavailable runs with their expected budget rather than silently removed.

Search curves maximize the objective recorded by the runner and carry the prior best score across
invalid, failed, or scoreless attempts. They stop at the number of evaluations actually present;
missing budget slots remain explicit and are not drawn as completed evaluations. The report displays
raw runner best-so-far and separately tracks analysis eligibility. Runtime is read only from an
explicit runtime field; it remains `null` / “Not recorded” when the source artifacts do not contain
search-level duration. Planner-step runtime and file timestamps are not substitutes for search
runtime.

Source revision is taken from an `adversarial_execution_context.v1` sidecar when present, otherwise
from candidate episode provenance. It is marked unknown, partial, or conflicting if exact revision
provenance cannot be established. The report records the comparison and search-manifest SHA-256
digests and hashes accessible config files. Fixture-based output demonstrates report behavior only;
it is not evidence that a planner is safe or that the adversary exhaustively searched its space.

All seed aggregates and observed min/max ranges are descriptive, not confidence intervals. No
inferential test is performed. With a small pilot, do not overinterpret differences between Random
and TPE. “No counterexample found” means only that the stated method found none under the recorded
finite budget.
