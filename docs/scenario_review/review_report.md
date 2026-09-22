# Traceable captions and scenario review reports (SREV-14)

Command-line glossary: SREV is the scenario-review portfolio; a scenario review report
aggregates numerical citations and assertions into structured JSON reports (`review-report.v1`),
editable caption records (`traceable-captions.v1`), Markdown reports (`review-report-markdown.v1`),
and standalone HTML reports (`review-report-html.v1`); a component request (`component-request.v1`)
is the envelope that invokes one scenario-review workbench component; shared contracts live in
`robot_sf/analysis_workbench/review_contracts.py` (SREV-01).

## What it does

`python -m robot_sf.analysis_workbench.review_report` generates traceable captions and structured
scenario review reports from simulation trace exports, failure diagnosis records, and trace annotation sets:
- **Numerical citations & provenance**: every cited number resolves to a source artifact id, source URI,
  source SHA-256 digest, RFC 6901 field pointer, numeric value, and unit. If any citation reference
  is invalid (missing field, type mismatch, non-finite number, or digest mismatch), the affected claim
  fails closed with an explicit diagnostic reason.
- **Explicit assertion categories**: claims and captions are categorized strictly into:
  - `observation`: direct empirical measurements from simulation trace exports (clearance, speeds, timestamps).
  - `diagnostic_hypothesis`: failure mechanism diagnoses and hypotheses (failure types, onset times, severity scores).
  - `manual_text`: human reviewer editorial commentary and notes provided in configuration.
- **Excerpt and cut disclosures**: when excerpts are presented, the active interval (`source_interval`),
  omitted cuts (`omitted_intervals`), and a pointer to the unedited full episode (`full_episode_link`)
  are explicitly recorded.
- **Missing measurement disclosures**: unavailable metrics (e.g. missing actor radii, absent failure
  diagnoses, or unobserved pedestrian interactions) emit explicit unavailable text explaining the omission.
- **Untrusted HTML escaping**: all user-provided strings and dynamic telemetry metadata rendered into
  HTML are escaped with `html.escape(..., quote=True)` to prevent cross-site scripting.
- **Diagnostic-only evidence boundary**: reports and captions are strictly diagnostic inspection tooling;
  they do not constitute benchmark, causal, or paper-facing evidence.

## Command

Run report generation from fixture inputs:

```bash
uv run python -m robot_sf.analysis_workbench.review_report \
  --input tests/fixtures/scenario_review/review_report/request.json \
  --config tests/fixtures/scenario_review/review_report/config.json \
  --output output/scenario_review/srev-14-smoke
```

Inspect capability descriptor:

```bash
uv run python -m robot_sf.analysis_workbench.review_report --descriptor
```

## Output Artifacts

- `report.json` (`review-report.v1`): structured machine-readable scenario review report:
  - `report_id`, `request_id`, `title`, `evidence_boundary`.
  - `provenance`: component metadata and verified SHA-256 source digests.
  - `excerpt`: `source_interval`, `omitted_intervals`, `full_episode_link`.
  - `claims`: list of assertion records with category, statement, status (`valid`/`failed`), and citation IDs.
  - `citations`: list of numerical citations with source artifact, pointer, unit, value, digest, and resolution status.
  - `missing_measurements`: list of unavailable telemetry items with explicit reasons.
  - `limitations`: explicit causal and statistical claim boundaries.
- `captions.json` (`traceable-captions.v1`): editable caption set linking each statement to underlying citation IDs.
- `report.md` (`review-report-markdown.v1`): human-readable Markdown report with footnote citations (`[^cite-...]`).
- `report.html` (`review-report-html.v1`): standalone, responsive HTML report with citation tables and styling.

## Unavailable Reasons and Failure Modes

- **Missing capabilities**: requesting undeclared capabilities returns `status: "unavailable"` with
  reason `unsupported required capability: <name>`.
- **Incompatible version**: requesting an incompatible major component version returns `status: "unavailable"`
  with reason `incompatible required version: <version>; component implements 0.1.0`.
- **Output collision**: existing output directories fail closed (`status: "failed"`) with
  reason `output collision, already exists: <path>` to prevent overwriting prior artifacts.
- **Corrupt input / integrity mismatch**: corrupted source JSON or SHA-256 checksum mismatches fail closed
  with `status: "failed"` and emit empty artifacts.
- **Invalid citations**: an unresolved pointer or value mismatch marks the affected citation and claim as `failed`,
  preventing false complete claims.
