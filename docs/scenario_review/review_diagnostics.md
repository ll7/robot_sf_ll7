# Scenario Review (SREV-18) recorded diagnostics

SREV-18 is an offline, read-only companion to the SREV-16 synchronized review
panels. It exposes only values retained in a supported trace inventory:

- planner-visible candidate actions, recorded costs, constraints, and selected
  action fields;
- commanded/requested versus executed/applied controls, with an explicitly
  labelled post-hoc difference only when both values and units are present;
- simulator-recorded pedestrian actors and diagnostics; and
- optional post-hoc `failure_diagnosis.v1` records.

The model keeps the source artifact, digest, source identity/revision, source
time, units, coordinate frame, actor selection, and SREV-16 context/selection
revision on each panel and exported evidence reference. Missing dimensions or
actors remain unavailable with a reason. Planner-visible input, simulator
ground truth, post-hoc values, and absent measurements are never merged into a
single unlabelled value.

## API and CLI

```python
from robot_sf.render.review_diagnostics import build_diagnostic_model, run

model = build_diagnostic_model(component_request, base=source_root)
result = run(component_request, base=source_root)
```

The standalone descriptor and CLI use the shared `component-request.v1` /
`component-result.v1` contracts:

```bash
uv run python -m robot_sf.render.review_diagnostics --descriptor
uv run python -m robot_sf.render.review_diagnostics \
  --input tests/fixtures/scenario_review/review_diagnostics/request.json \
  --output output/scenario_review/srev-18-smoke \
  --base tests/fixtures/scenario_review/review_diagnostics
```

The output directory must be new and relative to `--base`. A complete result
contains `review-diagnostics.v1.json`, the compatible
`evidence-reference-export.v1.json`, a self-contained HTML shell, the local
dependency-free browser component, and a missing-capability report. Partial,
unavailable, or failed runs remain diagnostic-only and do not advertise their
files as complete artifacts.

## Source and time rules

The component accepts the documented `simulation_timeline.v1`,
`simulation_trace_export.v1`, and `analysis-trace.v1` inventories, plus the
optional `failure_diagnosis.v1` sidecar. Simulation time is authoritative. The
cursor selects the nearest retained sample within the declared/observed
resolution; no interpolation, zero filling, stale-tail holding, or inferred
planner/private-observation value is permitted.

Source files are opened as bounded regular files beneath the configured root
with no-follow path checks. Corrupt/incompatible/unadmitted input, source or
unit/coordinate mismatches, stale context revisions, unsafe paths, symlinks,
output collisions, and invalid JSON fail closed. The browser module renders
text with DOM text nodes and makes no network requests, so recorded strings are
not interpreted as markup or executable code.

Source admission requires a declared SHA-256 that matches the bytes read. A
valid source without that binding is retained for diagnostic inspection with an
explicit `source_integrity_unbound` reason, but it cannot produce a complete
result. `analysis-trace.v1` is additionally checked through its owning
`trace_coverage` contract: identity, timing, finite actor state, controls,
units, coordinate frame, provenance, and the canonical artifact digest must all
be complete before the inventory is admitted.

The checked-in fixtures under
`tests/fixtures/scenario_review/review_diagnostics/` cover both timeline and
analysis-trace inventories, optional diagnosis, missing values, actor
disappearance, source/unit failures, symlinks, output collisions, and browser
runtime/XSS safety. This is diagnostic review tooling, not native planner
instrumentation, benchmark evidence, causal attribution, or a paper-facing
result.
