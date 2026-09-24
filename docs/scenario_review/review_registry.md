# SREV-29 review registry

Component registry and extension development kit
(`srev29-review-registry`, version `1.0.0`). It discovers installed trusted
component descriptors through the `robot_sf.scenario_review` Python
entry-point group, invokes one discovered component per request after
capability/version validation, and ships a reusable conformance battery
(`check_component_conformance`) that other leaves can run against their own
components — all usable standalone on fixture inputs without waiting for the
whole workbench.

## Evidence boundary

Diagnostic tooling only. Discovery, invocation, and conformance reports
describe installed fixture components and their deterministic smoke behavior.
They admit no scientific claim, benchmark result, or planner/simulator
behavior change. Entry points resolve in source-checkout (editable) installs;
on surfaces where an entry point cannot be imported, that component is
reported `unavailable` with a reason instead of failing the registry.

## Usage

```bash
uv run python -m robot_sf.analysis_workbench.review_registry \
  --input tests/fixtures/scenario_review/review_registry/request.json \
  --config tests/fixtures/scenario_review/review_registry/config.json \
  --output output/scenario_review/srev-29-smoke
```

`run(request)` executes the request; the CLI merges `--config` over the
request config, overrides the output directory with `--output`, and exits `0`
only when the result status is `complete`.

The request config is a closed allowlist (validated, never arbitrary code):
`mode` (`execute`, `discover`, or `conformance`), `target_component_id` for
`execute`/`conformance`, `target_config` passed to the invoked component,
`conformance_probe` (`sources` + `config`) for `conformance` mode, and an
optional `required_component_version`. The component descriptor
(`descriptor()`) declares the `bounded-execution` required capability and the
`registry-index.v1` / `conformance-report.v1` output types.

## Modes and outputs

- **execute** (default): discovers components, checks the target's supported
  input versions and required capabilities *before* invocation, then invokes
  the target's `run(request, base=...)` inside its own output subdirectory.
  Target artifacts pass through with registry routing provenance; a failing
  target yields the same status with a prefixed reason.
- **discover**: writes `registry-index.json` listing every entry point with
  `available`, `unavailable` (missing optional import, invalid descriptor,
  missing `run`), or `untrusted` (target outside the allowlisted module
  prefixes) status plus stable reasons. Conflicting component IDs fail
  closed; untrusted entries are never imported.
- **conformance**: stages the probe sources into an isolated inputs
  directory, runs the battery (`descriptor_validates`, `version_supported`,
  `missing_capability_refused`, `probe_completes`, `envelope_valid`,
  `namespace_contained`, `output_collision_refused`, `deterministic_rerun`),
  and writes `conformance-report.json` on success. A failing battery returns
  `failed` with per-check diagnostics and no artifacts.

A `complete` result references its artifacts; every other status carries
diagnostics and provenance only. Artifact URIs are relative,
traversal-free, and contained in the requested output directory — artifacts
can never nominate imports or shell execution.

## Stable reason codes

`invalid_config`, `output_collision`, `missing capabilities`,
`incompatible_version`, `unknown component`, `conflicting_component_id`,
`untrusted entry-point target`, `execution_error`, `corrupt source artifact`,
plus the target's own reasons prefixed with its component ID in `execute`
mode. Missing optional imports disable only the affected component.

## Fixture components

Two example components are registered through the entry-point group (no
central dispatch or browser edits):

- `srev29-example-analyzer` (`examples/scenario_review/components/episode_analyzer.py`):
  summarizes a tiny fixture episode trace to `analyzer-summary.json`
  (episode counts, displacements, mean step lengths, digests).
- `srev29-example-renderer` (`examples/scenario_review/components/telemetry_renderer.py`):
  renders the same trace to a 320x180 Matplotlib PNG plus
  `renderer-caption.json` (dimensions, digest, episode count). The savefig
  bounding box is pinned to `standard` inside an `rc_context`, so a
  worker-global plotting style cannot crop the canvas and change the raster
  dimensions.

Both follow the `run(request, base=...)` convention, validate their own
capabilities, refuse collisions, and return schema-validated results, so the
conformance battery in this module doubles as their executable contract.
