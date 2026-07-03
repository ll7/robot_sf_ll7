# Issue #4206 policy-structure mechanism cross-cut

status: blocked_missing_input_artifacts

Current blocker next action: Retrieve/hydrate the missing h600 run artifacts (reports/seed_episode_rows.csv and any mechanism sidecars) to this host, then re-run the builder. No exporter mechanism-label change is required until the rows are present.

This evidence packet is bounded to CPU-only diagnostic analysis for issue #4206. It does not run a
benchmark campaign, submit Slurm/GPU work, edit paper/dissertation claims, or promote generalized
causal claims.

Mechanism-level F-C4(ii) conclusions are allowed only when episode rows carry
`failure_mechanism_taxonomy.v1` fields with accepted confidence labels. Geometry buckets are used
only for the agreement/disagreement comparison table and never as substitute mechanism labels.

Blocked statuses distinguish two different next actions:

- `blocked_missing_input_artifacts`: the declared h600 run outputs are not present on this host;
  retrieve/hydrate them, then re-run. This is not a mechanism-instrumentation gap.
- `blocked_missing_trace_verified_mechanism_labels`: rows exist but lack trace-verified mechanism
  labels; add the labels to the exporter or a declared sidecar, then re-run.
