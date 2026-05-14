# Helper Catalog

This page lists small reusable helpers that scripts and examples should call
instead of duplicating rendering, artifact, or benchmark glue.

## Render Helpers

Module: `robot_sf.render.helper_catalog`

- `ensure_output_dir(path: Path) -> Path`
  - Resolves the path through the canonical artifact-root rules, creates it,
    and returns the resolved directory.
- `capture_frames(env, stride: int = 1) -> list[np.ndarray]`
  - Samples RGB frame arrays from common render buffers such as `env.frames` or
    `env.sim_ui.frames`.
  - If no buffer is populated, it calls `env.render()` once and returns the RGB
    frame array when that render method provides one.
  - Returns an empty list when a render-capable input produces no frame data.
  - Raises `ValueError` for invalid stride values and `AttributeError` when the
    input has no render method.

## Contact Sheets

Module: `scripts.generate_video_contact_sheet`

- `generate_contact_sheet(episodes_jsonl: Path, output_path: Path, *, columns: int = 3) -> Path`
  - Reads episode JSONL rows with image frame paths in either `frame_paths` or
    `video.frame_paths`.
  - Resolves relative frame paths next to the JSONL file.
  - Writes a deterministic PNG grid to `output_path` and returns that path.
  - Raises a clear error when no frame image paths are available. Rows that only
    contain MP4 paths should be pre-extracted to frame images before calling
    this helper.

