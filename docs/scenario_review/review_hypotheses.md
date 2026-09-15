# SREV-21 review hypotheses

Turn one explicit diagnostic hypothesis into a finite, executable experiment
recipe, so researchers can plan a bounded follow-up without waiting for the
whole workbench. See the [`docs/glossary.md`](../glossary.md) for
project terms such as review bundle and experiment recipe.

## Evidence boundary

Diagnostic tooling only. Recipe creation launches nothing — no simulation,
training, remote scheduling, or scientific publication — and admits no
scientific claim, benchmark result, or planner/simulator behavior change.
Generated recipes are not campaign or evidence-admission authority.

## Interfaces

- **Input**: a `component-request.v1` request for component
  `srev21-review-hypotheses` whose config carries one `hypothesis` mapping
  with `template` (`single-pedestrian-speed` or
  `single-pedestrian-start-delay`), numeric `factor_value`, `pedestrian_id`,
  `expected_direction` (`increase`/`decrease`), `priority`, and
  `terminal_condition`. An optional `required_component_version` gates
  compatibility.
- **`run(request)`**: returns a `component-result.v1` result with status
  `complete` (artifacts `experiment-recipe.json` plus
  `component-descriptor.json`), `unavailable` (unsupported component,
  capability, hypothesis template, or required version, with an actionable
  reason), or `failed` (corrupt hypothesis fields or output collision; failed
  outputs never carry artifacts).
- **Output**: one `experiment-recipe.v1` document with three deterministically
  ordered candidate interventions (priority, then stable ID), an unchanged
  control bound to the source config hash, measurements with units and
  expected direction, an evaluation rule using the
  survived/falsified/inconclusive vocabulary, a default budget of three
  candidates / six simulator executions / 600 elapsed seconds / one concurrent
  local CPU process, stop rules including the terminal condition, and a
  preservation destination. `descriptor()` exposes the versioned capability
  descriptor; unsupported optional streams never block supported operations.

Unknown required versions fail; output collisions and unsafe paths are
rejected. Canonical hashes bind logical content, versions, and config — not
absolute paths. Repeated fixture runs compare equal artifact digests.

## Usage

```bash
uv run pytest tests/analysis_workbench/test_review_hypotheses.py -q
uv run python -m robot_sf.analysis_workbench.review_hypotheses \
  --input tests/fixtures/scenario_review/review_hypotheses/request.json \
  --config tests/fixtures/scenario_review/review_hypotheses/config.json \
  --output output/scenario_review/srev-21-smoke
```

`run(request)` composes the recipe offline; `descriptor()` lists what this
component provides. Outputs are written atomically into the component-owned
directory; sources are never modified.

## Errors

Failure reasons carry stable codes: `missing-hypothesis`,
`unsupported-hypothesis-template`, `corrupt-hypothesis`,
`incompatible-required-version`, `missing capabilities`, and
`output-collision`. Missing measurements are unavailable with reasons; source
integrity is separate from evidence admission.
