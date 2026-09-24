# Review encode (SREV-10)

Command-line glossary: SREV is the scenario-review portfolio; an edit plan is
the declared list of cuts, pauses, speed changes, and crop applied to a source;
a component request (`component-request.v1`) is the fixture envelope that
invokes one workbench component; shared contract shapes live in
`robot_sf/analysis_workbench/review_contracts.py` (SREV-01).

## What it does

`python -m robot_sf.render.review_encode` consumes actual source pixels from a
bounded frame-sequence manifest or a source clip. It applies a declared edit
plan and emits a playable MP4 with an encoder receipt and a piecewise
source-time/presentation-time map. It is a diagnostic review aid only: it does
not produce benchmark, safety, causal, planner, simulator, training, or
scientific evidence.

The frame-sequence fixture contract is `frame-sequence-manifest.v1`: the
manifest declares `source_fps`, `source_frames`, `frame_paths`, and one SHA-256
`frame_digests` entry per frame. The request's source reference must also carry
the manifest SHA-256. A `source-clip` reference is hashed before bounded
decoding. A changed or mismatched source fails closed.

When exactly one source family is required, that family is authoritative and
an optional reference from the other family is ignored without decoding. Two
required source families, or multiple unqualified source families, fail closed
as an ambiguous multi-source request.

Edits use half-open source-time intervals. Speed edits split the timeline at
their boundaries and support both fast motion (`factor > 1`) and slow motion
(`factor < 1`); cuts concatenate retained spans; endpoint and sub-frame pauses
are rounded up to at least one presentation frame. A terminal pause is accepted
only when the retained timeline reaches the source endpoint; a tail cut drops
that pause with a stable partial diagnostic. Pause and speed expansion is
projected against the output-frame limit before repeated frames are allocated.
Conflicting speed maps, cut-away pauses, and clamped crops are explicit
failures or partial diagnostic results as described below.

## Command

```bash
uv run python -m robot_sf.render.review_encode \
  --input tests/fixtures/scenario_review/review_encode/request.json \
  --config tests/fixtures/scenario_review/review_encode/config.json \
  --output output/scenario_review/srev-10-smoke
```

## Output

- `edit.mp4`: playable H.264 edit at the configured preset (tiny fixture
  preset: 160x120 at 10 fps; default export 1920x1080 at 30 fps).
- `encode-receipt.json`: encoder environment (backend, codec, versions,
  platform), verified input digests, edit-plan digest, output geometry, the
  normalized crop operation (`null` or `{"operation": "crop", "box":
  [left, top, right, bottom]}`), and the silent audio policy. No wall-clock
  fields are emitted.
- `time-map.json`: piecewise segments with presentation/source intervals and
  the producing operation (`cut`, `speed`, `pause`, `passthrough`), plus the
  full frame-order source index list with first/terminal frames and the same
  normalized crop provenance.
- The printed `component-result.v1` envelope carries `complete`, `partial`,
  `unavailable`, `failed`, or `cancelled` with stable reason codes. Cancellation
  is available to programmatic callers through the optional `cancel` predicate
  or event passed to `run`. It is checked during preflight and before source
  loading; a request cancelled at that boundary returns `cancelled` with reason
  `cancellation_requested_before_source_read` and publishes no artifacts.
  Cancellation cannot interrupt an active source read, decoder, or encoder, and
  the standalone CLI does not expose a cancellation hook.
  Complete artifact records contain `artifact_id`, relative `uri`, and the file
  SHA-256. A handled non-complete CLI result exits nonzero; invalid CLI input
  exits with a separate nonzero code.

## Unavailable reasons and limits

- Missing encoder backend, a missing required source family (frame-sequence or
  source-clip), or unsupported required capabilities make the result
  `unavailable`, never silently complete.
- `config.min_component_version`, when present, accepts one to three ASCII
  decimal components with an optional `v`/`V` prefix: `1`, `1.0`, `1.0.0`, `v1.0`, and
  `V1.0.0` are examples. Components are compared numerically after zero-padding
  to major/minor/patch, including arbitrarily long numeric components without
  converting untrusted text to a native integer. A malformed value fails with
  the stable reason prefix `config_invalid` and detail
  `min_component_version must be a dotted version string`; a valid minimum
  newer than this component's `1.0.0` fails with the stable reason prefix
  `incompatible_component_version`.
- MP4 byte identity is promised only within the declared encoder environment
  recorded in the receipt. No generated media is stored in git.
- Output paths must remain below the supplied base, contain no symlink or
  non-directory component, and use no-overwrite publication. Existing output
  directories, including empty ones, are collisions.
- Admission bounds include 3,000 source frames, 300 seconds, 240 source fps,
  6,000 output frames, 1920x1080 output pixels, 120 output fps, 256 MiB of
  encoded source bytes and retained decoded-frame buffers, 512 MiB output
  bytes/buffer, 30-second decoder and encoder deadlines, speed factors from
  0.125x to 8x, and pauses up to 300 seconds. These are hard limits, not
  tuning hints. Decoder and encoder calls from non-main-thread API callers fail
  closed because the signal-based deadline is unavailable there.
- The evidence boundary is machine-readable in the result, receipt, and map:
  `evidence_status` is `diagnostic-only`, `benchmark_success` is `false`, and
  `scientific_claim_allowed` is `false`. An encoded review clip reflects the
  declared plan and provided sources, not scientific merit or benchmark
  success.
