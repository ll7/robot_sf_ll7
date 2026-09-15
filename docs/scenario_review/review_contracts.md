# SREV-01 review contracts

Versioned, executable contracts for scenario review validate structure and
source identity before diagnostic tooling uses a fixture: `review-bundle.v1`,
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

## Admitted fixture-source receipts (SREV-22)

`admitted-source-receipt.v1` is a separate companion contract for a checked-in
fixture or diagnostic source. `resolve_admitted_source()` returns protected
`source_bytes` only when all of these checks pass. The accompanying
`source_path` is retained for location-compatible reporting and must not be
reopened as the admitted source.

- the receipt has `status: admitted`, `source_kind: fixture` or `diagnostic`,
  `evidence_boundary: diagnostic_only`, and `scientific_claim_allowed: false`;
- the receipt's request and recipe SHA-256 values match the current validated
  `component-request.v1` and `experiment-recipe.v1` identities. The request
  identity includes every request field except `output_directory`, including
  `config`, and every source declaration (`artifact_id`, `uri`, `format`,
  `schema`, `sha256`, `source_commit`, `config_identity`, `units`, and
  `coordinate_frame`). Absent optional source declarations normalize to empty
  strings;
- the request contains the receipt URI and format, and a recipe supplied to the
  resolver contains an `admission_reference` equal to `receipt_id`;
- the source commit and config identity in the current recipe's
  `source_identity` match the receipt. Optional explicit resolver expectations
  may corroborate those identities but cannot replace the recipe context; and
- the recipe's `source_identity` preserves the receipt's source kind and the
  diagnostic-only boundary: `kind` is the receipt kind,
  `evidence_boundary` is `diagnostic_only`, `scientific_claim_allowed` is
  `false`, and `dependent_family_status` is `standalone_fixture_only`; and
- the URI is a relative local path beneath the caller-supplied `allowed_root`,
  and a fresh SHA-256 rehash matches the receipt's source bytes. The source is
  opened through no-follow directory/file descriptors, checked as a regular
  file, and hashed/read from that same protected descriptor.

The resolver requires both the current request and recipe documents for public
admission; caller-supplied digests or source identities cannot stand in for
those documents. It supports relative local paths such as `source.json` and
rejects URI schemes, query/fragment components, absolute paths, traversal, and
symlink escapes. Supplied request source metadata is checked for exact equality
with the receipt, with SHA-256 and commit values compared case-insensitively.
`format` and `schema` are required receipt metadata; the request format and any
recipe `source_schema` are checked for exact equality. The resolver does not
parse arbitrary source formats or choose a simulator map.
Receipt files are limited to 256 KiB, receipt identity fields have bounded
lengths, and source hashing stops at 16 MiB. Validation and resolver
diagnostics are bounded to 200 characters and at most 32 retained validation
errors. These are resource-safety limits; they do not widen source admission.
An oversized source returns `unavailable` with reason `source_too_large`.

The resolver keeps source integrity separate from scientific evidence
admission. `status: admitted` means that this fixture source is available and
exactly bound for a diagnostic invocation. It does not admit a benchmark,
planner, simulator, or paper-facing claim. Existing public v1 request and
recipe schemas remain unchanged. A v1 recipe without `admission_reference`,
source commit/config expectations, or the required diagnostic-only boundary
fields is returned as `unavailable` with reason `receipt_stale` and a migration
detail; callers must add the receipt companion and source identities before
integrating this resolver into execution preflight.
Native selection and executor integration belong to
[#9293](https://github.com/ll7/robot_sf_ll7/issues/9293) / [PR #9333](https://github.com/ll7/robot_sf_ll7/pull/9333).

Stable result pairs are:

| Status | Reason | Meaning |
| --- | --- | --- |
| `admitted` | `admitted` | Source was opened as a no-follow regular file under the allowed root and its protected bytes match. |
| `unavailable` | `receipt_missing`, `receipt_unreadable` | The receipt cannot be loaded. |
| `failed` | `receipt_malformed` | The receipt is readable JSON but fails the v1 schema. |
| `unavailable` | `receipt_unsupported` | The version or diagnostic-only boundary is unsupported. |
| `unavailable` | `receipt_stale` | Request, recipe, admission, source-identity, or request-source binding differs. |
| `unavailable` | `source_missing`, `source_not_regular` | The approved source is absent or not a regular file. |
| `unavailable` | `source_too_large` | The approved source exceeds the bounded hashing input limit. |
| `unavailable` | `source_protection_unavailable` | The platform cannot provide the required protected descriptor boundary. |
| `failed` | `source_mutated` | The source exists but its rehashed bytes differ from the receipt. |
| `unavailable` | `source_escaped_root` | The URI or resolved symlink leaves `allowed_root`. |
| `unavailable` | `allowed_root_missing`, `allowed_root_invalid` | The caller supplied no usable local trust root. |

The checked-in synthetic fixture provides a callable smoke path:

```python
import json
from pathlib import Path

from robot_sf.analysis_workbench.review_contracts import resolve_admitted_source

root = Path("tests/fixtures/scenario_review/admitted_source")
result = resolve_admitted_source(
    root / "receipt.json",
    allowed_root=root,
    request=json.loads((root / "request.json").read_text()),
    recipe=json.loads((root / "recipe.json").read_text()),
)
assert result.status == "admitted", result.to_dict()
assert result.source_path == (root / "source.json").resolve()
assert result.source_bytes == (root / "source.json").read_bytes()
```

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

## Review workbench (SREV-15)

```bash
uv run python -m robot_sf.render.review_workbench \
  --input <component-request.json> \
  --config <config.json> \
  --output review-workbench --base output/scenario_review
```

The `srev15-review-workbench` component consumes `review-bundle.v1` and
`visualization-spec.v1` sources and writes an offline, network-free
`review-workbench.v1.html` plus the `review-workbench.v1.json` document:
episode-scoped index, per-artifact availability and integrity, the shared
simulation-time base, and named extension slots. Behaviour boundaries:

- a video stream without an explicit `presentation_timestamp_map` is reported
  `unavailable` (`presentation_timestamp_map_missing`); frame/fps alignment is
  never guessed, and the presentation plan records declared cuts/pauses/crops
  with the default 1920x1080/30 fps/original-speed export (smaller `test` preset);
- integrity is reported per bundle reference and never grants admission;
- a declared but absent source degrades the run to `partial` with a reason; an
  explicitly required `threejs-scene` capability delegates to
  `robot_sf.render.threejs_viewer` instead of reimplementing playback;
- output collisions fail closed, and only the requested output directory is written.

`--descriptor` prints the component descriptor (`component-descriptor.v1`) without
executing a request.

## Errors

Validation failures carry stable reason codes with source pointers
(`ReviewContractsValidationError.errors`). Receipt-loader path and parse
failures use the typed `ReviewContractsValidationError`; invalid filesystem
and JSON diagnostics are bounded. Missing measurements are unavailable with
reasons; source integrity is separate from evidence admission.
