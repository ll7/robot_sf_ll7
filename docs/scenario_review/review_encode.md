# Review encode (SREV-10)

Command-line glossary: SREV is the scenario-review portfolio; an edit plan is
the declared list of cuts, pauses, speed changes, and crop applied to a source;
a component request (`component-request.v1`) is the fixture envelope that
invokes one workbench component; shared contract shapes live in
`robot_sf/analysis_workbench/review_contracts.py` (SREV-01).

## What it does

`python -m robot_sf.render.review_encode` applies a declared edit plan to a
source frame sequence (rendered deterministically in-component) or an optional
source clip, and emits a playable MP4 with an encoder receipt and a piecewise
source-time/presentation-time map. Overlapping speed changes with different
factors fail as conflicting time maps; clamped crops and pauses landing in
cut-away regions are partial with explicit diagnostics, never silent. Audio
output is always silent (v1); any other audio policy fails closed.

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
  platform), input digests, edit-plan digest, output geometry, and the silent
  audio policy. No wall-clock fields, so receipts compare byte-identical.
- `time-map.json`: piecewise segments with presentation/source intervals and
  the producing operation (`cut`, `speed`, `pause`, `passthrough`), plus the
  full frame-order source index list with first/terminal frames.
- The printed result envelope carries `complete`, `partial`, `unavailable`, or
  `failed` with stable reason codes. Only `complete` results list envelope
  artifacts; files stay on disk either way.

## Unavailable reasons and limits

- Missing encoder backend, missing frame-sequence family, or unsupported
  required capabilities make the result `unavailable`, never silently complete.
- MP4 byte identity is promised only within the declared encoder environment
  recorded in the receipt; determinism proof compares time-map bytes, receipt
  bytes, and decoded pixels. No generated media is stored in git.
- Evidence boundary: the edit reflects the declared plan and provided sources,
  not scientific merit. No benchmark, safety, or paper-facing claim follows
  from an encoded review clip.
