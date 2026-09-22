# Review context (SREV-06)

SREV is the scenario-review portfolio. SREV-06 produces a diagnostic-only
cohort context report from one explicitly selected, recorded campaign-result
source. It does not execute a simulator or establish population, benchmark,
causal, safety, or paper-facing claims.

## Contract and command

The component consumes the shared `component-request.v1` request and emits the
shared `component-result.v1` envelope. Its descriptor is available from
`robot_sf.analysis_workbench.review_context.descriptor()` and its leaf-owned
machine-readable payload registry from `output_schemas()`. The registry covers
`review-context.v1` and `missing-capability-report.v1`; the shared envelope
schemas remain owned by SREV-01.

```bash
uv run python -m robot_sf.analysis_workbench.review_context \
  --input tests/fixtures/scenario_review/review_context/request.json \
  --config tests/fixtures/scenario_review/review_context/config.json \
  --output output/scenario_review/srev-06-smoke
```

Every source reference must declare its expected family schema, SHA-256,
40-character source commit, and non-empty configuration identity. The loader
checks the exact bytes, rejects absolute/traversal/symlink paths, and requires
the source payload's schema, execution status, commit, and configuration
identity. Campaign sources may use either the leaf `campaign-result.v1` JSON
document or the canonical `campaign-result-store.v2`/case-workbench directory
contract; the latter is adapted through the case-workbench result loader and
does not redefine its schema. For the canonical directory shape, `study_id` in
the owner manifest (or per-row `campaign_id`) must bind the campaign, and the
owner manifest/rows must provide one matching source commit and configuration
identity (`config_hash` or `config_identity`); `config_digest` is independent
provenance rather than an identity alias. The requested
source declarations must match those owner values; caller-only identity is
rejected as `unavailable` with `canonical_identity_unbound` or
`canonical_identity_mismatch`, without a report. Multiple campaign or selection
candidates require an explicit `canonical_*_artifact_id` config key; they are
never silently merged.

When the same campaign source enters the Benchmark Auditor scanner, its
`campaign_id`, `study_id`, and `campaign` declarations (including nested
`config.campaign`) must resolve to one campaign identity across the source
and rows. Conflicting aliases are rejected before a selected episode can be
materialized. This scanner rule does not make `config_digest` a configuration
identity alias for the canonical owner contract above.

For canonical owner rows, `config_hash` is the configuration identity bound to
the request's `config_identity`. `config_digest` is a separate resolved-config
provenance value: it is retained in each source-provenance record and never
substituted for or compared as an alias of `config_hash`. If a source contains
several independent digests, the singular field is `null` and `config_digests`
retains the sorted set. Supported nested
source-commit aliases (`source_commit`, `git_hash`, `commit_sha`, `commit`) and
configuration-identity aliases (`config_identity`, `config_hash`) must agree;
contradictions are rejected.

Canonical v2 owner rows use `row_status` as the execution-mode field. Their
`status`/`execution_status` columns may contain descriptive outcomes such as
`success` or `collision`; those values do not upgrade a `fallback` or `degraded`
`row_status`, and contradictory recognized execution modes fail closed.

Campaign and selection sources must share source-commit and configuration
identities. Each source's declared digest is compared with its observed digest,
and both identities are retained in the report provenance. The independent
owner `config_digest`, when present, is retained alongside the canonical
configuration identity. The optional
`episode-selection` stream is consumed whenever supplied. If it is explicitly
listed in `skip_optional_capabilities`, the report is `partial` and records
`selection_coverage.status: skipped`; an absent optional stream is reported as
`not_supplied` rather than treated as selected coverage.

Every supplied selection must carry a non-empty `campaign_id` matching the
requested campaign before any episode IDs are interpreted. Missing or mismatched
selection campaign identity returns a failed result without a cohort report.

## Outputs

- `context-report.json` (`review-context.v1`) records planner/scenario/config/
  seed grain, the denominator of admitted rows, outcome frequencies, missing-aware
  metric summaries, deterministic linear-interpolation percentiles, selection
  coverage, the selected campaign source, and excluded row statuses.
- `context-report.html` is a deterministic standalone rendering of the same
  diagnostic context, including campaign identity, selection status and
  coverage, source commit/configuration identity, and observed digest.
- `missing-capability-report.json` (`missing-capability-report.v1`) records
  optional streams that were not supplied and structured diagnostics.
- A `complete` result lists all three files in its component-result envelope.
  Partial, unavailable, and failed results do not list usable artifacts;
  partial reports may remain on disk as diagnostic output.

All three output files are materialized through an fsync'd same-directory
temporary file and an atomic no-replace publication held by retained no-follow
directory descriptors. Publication uses `dir_fd` operations and fails closed
if the reserved output entry is replaced, so a symlink/TOCTOU swap cannot
redirect reports outside the reserved directory. An existing final name is an
output collision; it is never overwritten.

Control documents are bounded to 1 MiB and bounded nesting, collections, nodes,
and strings. CLI control paths are opened no-follow and non-blocking, then
required to be regular files, so FIFO, directory, symlink, special-file,
oversized, and invalid inputs return a stable failed result envelope without
waiting on a writer. Source files and canonical source directories are bounded
to 64 MiB and 4,096 regular files; symlink and special-file entries are
rejected. Before canonical owner loading, the complete inventoried directory is
read once through descriptor-backed no-follow/non-blocking opens into a private
bounded snapshot; owner loaders never reopen the original source paths after the
digest check. Non-standard JSON constants such as `NaN` and `Infinity` are
rejected in in-memory control documents, canonical JSONL, and final result/output
serialization. Existing malformed finite metric handling remains a diagnostic
partial result only where the leaf contract requires it.

Fallback, degraded, unavailable, and failed rows are excluded from the
denominator and are recorded in `exclusions`. A requested capability without an
actual valid source returns `unavailable`; malformed or conflicting source
content returns `failed`. An absent or unresolved `campaign_id` returns
`unavailable` without writing a cohort report. Unknown selections and invalid
metric values prevent a complete result and remain visible as diagnostics.

The shared request/result contracts and central discovery/registration remain
owned by SREV-01 and SREV-29. This leaf does not edit those schemas, registries,
or shared test fixtures; a missing central registration remains an owner-level
hosted-check blocker and is not represented as benchmark success.

The checked-in fixture is synthetic and diagnostic-only. Generated output is
temporary under `output/` and is not a durable evidence dependency.
