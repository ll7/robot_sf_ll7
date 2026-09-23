# Review media QA (SREV-13)

Command-line glossary: SREV is the scenario-review portfolio; a media quality report
(`media-quality-report.v1`) records explicit pass/fail/unavailable checks for fixture media plus a
contact sheet (`contact-sheet.v1`); a component request (`component-request.v1`) is the fixture
envelope that invokes one workbench component; shared contract shapes live in
`robot_sf/analysis_workbench/review_contracts.py` (SREV-01).

## What it does

`python -m robot_sf.render.review_media_qa` runs standalone video and visual-quality checks on
synthetic fixture clips:
- **Clip integrity**: corrupt (non-JSON/wrong-schema), blank (no frames or all blank), and
  truncated (declared vs parsed frame count) clips fail with stable reason codes.
- **Duration match**: declared clip duration is compared against measured frame timestamps within
  a parameterized tolerance.
- **Timestamp monotonicity**: decreasing timestamps fail; repeated timestamps outside a declared
  storyboard pause are unexplained freezes and fail, while storyboard-declared pauses pass.
- **Timestamp sync**: synchronized checks require an explicit presentation-timestamp map; a
  missing map disables them with a reason (frame/fps guessing is forbidden).
- **Label quality**: unreadable text, pairwise label overlap, and frame clipping are reported per
  frame.
- **Contact sheet**: a deterministic PNG grid plus a JSON cell manifest are written to the
  requested output directory (runtime `output/` only, never committed).
- **Container gating**: real video containers are capability-gated and reported as unsupported
  rather than silently passed.

## Command

```bash
uv run python -m robot_sf.render.review_media_qa \
  --input tests/fixtures/scenario_review/review_media_qa/request.json \
  --config tests/fixtures/scenario_review/review_media_qa/config.json \
  --output output/scenario_review/srev-13-smoke
```

Inspect capability descriptor:

```bash
uv run python -m robot_sf.render.review_media_qa --descriptor
```

## Output

- `media-quality-report.v1.json`: quality report containing:
  - `overall_verdict`: `"pass"`, `"fail"`.
  - `checks`: per-check records (`clip_integrity`, `duration_match`,
    `timestamp_monotonicity`, `timestamp_sync`, `label_quality`, optional
    `container_decode`) with `verdict` and stable `reason_code`.
  - `thresholds`: effective duration tolerance and minimum label characters.
  - `qa_tool_version`: component version that produced the report.
- `contact-sheet.v1.json`: contact-sheet manifest (grid geometry plus per-frame cells).
- `contact-sheet.v1.png`: rendered contact-sheet grid (runtime output only).
- Component result JSON printed to stdout: status (`complete`, `partial`, `unavailable`, `failed`), written artifacts, diagnostics, and provenance.

## Unavailable reasons and limits

- If no clip source is declared, the component reports `unavailable` with reason `no clip
  source declared`.
- If timestamp sync is requested without an explicit map, the check reports `unavailable` with
  reason `missing_timestamp_map`.
- If a real container source is declared without an available decoder, the container check
  reports `unavailable` with reason `decoder_unavailable`.
- If unsupported capabilities are requested, the component reports `unavailable` with `missing
  capabilities: <list>`.
- Corrupt, blank, truncated, non-monotonic, or mislabeled clips produce `partial` component
  status with failing checks; partial/failed outputs never carry complete status.
- Output directory collisions are rejected to prevent overwriting prior runs or durable data.
- Evidence boundary: quality reports provide diagnostic visualization tooling only. No benchmark
  admission, planner change, or scientific claim follows from media QA.
