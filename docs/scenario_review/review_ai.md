# SREV-26 review AI

Draft structured, evidence-linked explanations with validated row/number
citations, so researchers can inspect saved or fixture evidence without
waiting for the whole workbench. See the [`docs/glossary.md`](../glossary.md)
for project terms such as review bundle and component result.

## Evidence boundary

Diagnostic tooling only. Drafting launches nothing — no simulation,
training, remote scheduling, model calls, or scientific publication — and
admits no scientific claim, benchmark result, or planner/simulator behavior
change. Explanations are marked draft, packet text is quoted as data (never
instructions), and fabricated citations are rejected: every number traces to
an owner-validated source row or an explicitly configured highlight.

## Interfaces

- **Input**: a `component-request.v1` request for component
  `srev26-review-ai`. Sources cite evidence by identity; citable formats are
  `trace_annotation_set.v1` (validated through
  `robot_sf.analysis_workbench.trace_annotation`, rows cited per annotation)
  and `failure_diagnosis.v1` (validated through
  `robot_sf.benchmark.failure_diagnosis`, rows cited per record). Source URIs
  resolve relative to the request file's directory. Other formats are skipped
  with a diagnostic and never block supported sources.
- **Config**: an `explanation` mapping with non-empty `focus` and optional
  `highlights` (each a `metric`/`value`/`units` triple bound to a usable
  `source_artifact_id`; non-finite values, missing units, and unresolvable
  sources fail). `provider` selects the renderer: only `fake-local` (the
  deterministic offline template) is shipped; any other name is unavailable,
  with or without `allow_remote`. Free-form `notes` are quoted verbatim as
  data. Credential-looking config keys are redacted from outputs.
- **`run(request)`**: returns a `component-result.v1` result with status
  `complete` (artifacts `explanation.json` plus
  `component-descriptor.json`), `partial` (some sources skipped, diagnosed
  per source), `unavailable` (unsupported component, capability, evidence,
  version, or provider, with an actionable reason), or `failed` (corrupt
  config, missing evidence, or output collision; failed outputs never carry
  artifacts).
- **Output**: one `review-explanation.v1` document with draft captions that
  only restate cited rows, per-point citations (annotation rows, diagnosis
  rows, highlight values with units), quoted packet text, redaction list, and
  a canonical `explanation_sha256` over the logical content. `descriptor()`
  exposes the versioned capability descriptor.

Unknown required versions fail; output collisions and unsafe paths are
rejected. The explanation copies the source identities accepted by the v1
request schema; that schema carries no source-byte hash. Repeated fixture
runs compare equal artifact bytes.

## Usage

```bash
uv run pytest tests/analysis_workbench/test_review_ai.py -q
uv run python -m robot_sf.analysis_workbench.review_ai \
  --input tests/fixtures/scenario_review/review_ai/request.json \
  --config tests/fixtures/scenario_review/review_ai/config.json \
  --output output/scenario_review/srev-26-smoke
```

`run(request)` drafts the explanation offline; `descriptor()` lists what this
component provides. Outputs are written atomically into the component-owned
output directory, which must not exist beforehand. Generated output stays
under ignored `output/` and is disposable.
