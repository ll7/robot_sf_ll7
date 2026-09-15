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

Edits use half-open source-time intervals. Speed edits split the timeline at
their boundaries and support both fast motion (`factor > 1`) and slow motion
(`factor < 1`); cuts concatenate retained spans; endpoint and sub-frame pauses
are rounded up to at least one presentation frame. Conflicting speed maps,
cut-away pauses, and clamped crops are explicit failures or partial diagnostic
results as described below.

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
  platform), verified input digests, edit-plan digest, output geometry, and the
  silent audio policy. No wall-clock fields are emitted.
- `time-map.json`: piecewise segments with presentation/source intervals and
  the producing operation (`cut`, `speed`, `pause`, `passthrough`), plus the
  full frame-order source index list with first/terminal frames.
- The printed `component-result.v1` envelope carries `complete`, `partial`,
  `unavailable`, or `failed` with stable reason codes. Complete artifact
  records contain `artifact_id`, relative `uri`, and the file SHA-256. A
  handled non-complete CLI result exits nonzero; invalid CLI input exits with a
  separate nonzero code.

## Unavailable reasons and limits

- Missing encoder backend, missing frame-sequence family, or unsupported
  required capabilities make the result `unavailable`, never silently complete.
- MP4 byte identity is promised only within the declared encoder environment
  recorded in the receipt. No generated media is stored in git.
- Output paths must remain below the supplied base, contain no symlink or
  non-directory component, and use no-overwrite publication. Existing output
  directories, including empty ones, are collisions.
- Admission bounds include 3,000 source frames, 300 seconds, 240 source fps,
  6,000 output frames, 1920x1080 output pixels, 120 output fps, 256 MiB
  source bytes/buffer, 512 MiB output bytes/buffer, 30-second decoder and
  encoder deadlines, speed factors from 0.125x to 8x, and pauses up to 300
  seconds. These are hard limits, not tuning hints.
- The evidence boundary is machine-readable in the result, receipt, and map:
  `evidence_status` is `diagnostic-only`, `benchmark_success` is `false`, and
  `scientific_claim_allowed` is `false`. An encoded review clip reflects the
  declared plan and provided sources, not scientific merit or benchmark
  success.
