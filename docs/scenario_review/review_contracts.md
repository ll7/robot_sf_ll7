# SREV-01 review contracts

Versioned, executable contracts for scenario review: `review-bundle.v1`,
`visualization-spec.v1`, `component-request.v1` / `component-result.v1`
(with `component-descriptor.v1`), and `experiment-recipe.v1`.

## Evidence boundary

Diagnostic tooling only. These contracts validate structure, identity, and
provenance of review inputs — they admit no scientific claim, benchmark
result, or planner/simulator behavior change.

## Interfaces

- **ReviewBundle**: index of episode/trace/geometry/media/diagnostic
  references (artifact ID, URI, format/schema, SHA-256, source commit/config,
  units, coordinate frame). Episode-scoped IDs preserve original IDs.
  Availability carries a typed reason; execution mode and scientific admission
  stay separate.
- **VisualizationSpec**: source refs, actor/event selection,
  camera/layout/layers/style, annotations, comparisons, ordered source
  intervals. Source time and presentation time stay separate; half-open
  intervals with explicit terminal-frame inclusion prevent duplicate
  cut-boundary frames. Units are seconds/metres/radians unless
  source-declared transforms convert them. Default presentation export is
  1920x1080 at 30 fps, original speed; tests use a smaller preset.
- **ComponentRequest/Result**: component ID/version, source refs, config,
  output directory; descriptors declare supported input versions and
  required/optional capabilities. Results are exactly
  complete/partial/unavailable/failed/cancelled; partial or failed outputs
  cannot carry complete status. Unsupported capabilities are never
  synthesized. Output collisions and unsafe paths are rejected; artifacts
  cannot request imports or shell execution.
- **ExperimentRecipe**: hypothesis, source/config identity, finite candidate
  interventions, controls, measurements/units, evaluation rule, budget, stop
  rules, preservation destination, admission reference.

Unknown required versions fail; optional extensions round-trip. Canonical
hashes bind logical content, versions, and config — not absolute paths.

## Usage

```bash
uv run python -m robot_sf.analysis_workbench.review_contracts \
  --input tests/fixtures/scenario_review/review_contracts/request.json \
  --config tests/fixtures/scenario_review/review_contracts/config.json \
  --output srev-01-smoke --base output/scenario_review
```

`run(request)` executes the `srev01-inspect` and `srev01-capability-report`
components offline; `capability_report()` lists what this module executes.
Outputs are written atomically into the component-owned directory; sources
are never modified.

## Errors

Validation failures carry stable reason codes with source pointers
(`ReviewContractsValidationError.errors`). Missing measurements are
unavailable with reasons; source integrity is separate from evidence
admission.
