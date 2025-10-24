# Contract: Visualization Helper API

Purpose: Define expected inputs/outputs and error modes for the visualization helpers to be moved from the example into `robot_sf/benchmark/visualization.py`.

## Functions (Phase A)

1. frame_shape_from_map(map_svg_path: str) -> tuple[int,int]
   - Returns: (width_px, height_px)
   - Errors: FileNotFoundError if path missing; ValueError on invalid SVG

2. overlay_text(canvas, text: str, pos: tuple[int,int], font: Optional[str]=None) -> None
   - canvas: an abstraction supporting a draw_text(text, pos, font) method (duck-typed)
   - Side effects: mutates canvas
   - Errors: TypeError if canvas missing draw_text

3. format_summary_table(metrics: dict[str, float]) -> str
   - Returns: Markdown table string and optional LaTeX-safe string
   - Errors: ValueError on empty metrics

## Contract Tests (Phase 1 intent)
- One failing test per function that imports the contract and asserts signature & common errors.
- The tests live under `tests/unit/benchmark/` and will be implemented after moving the helpers.

