# Phase 1 — Data Model: Helpers extracted for visualization & utils

Date: 2025-09-29

Entities

- Helper: { name: str, module: str, signature: str, side_effects: list[str], deps: list[str] }

Example helper entries (Phase A candidates):

- frame_shape_from_map(map_svg_path: str) -> tuple[int,int]
  - side_effects: []
  - deps: [xml.etree.ElementTree, svg parsing helper]

- overlay_text(draw_ctx, text: str, pos: tuple[int,int], font=None) -> None
  - side_effects: draw on provided context
  - deps: [Pillow or matplotlib.text]

- format_summary_table(metrics: dict[str,float]) -> str
  - side_effects: []
  - deps: [textwrap, pandas (optional)]

Validation rules
- Helpers must accept explicit inputs; no implicit global state.
- Avoid top-level imports of optional heavy deps; use lazy imports inside function bodies.

State transitions: N/A (helpers are pure or accept contexts)
