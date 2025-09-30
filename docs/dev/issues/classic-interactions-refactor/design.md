# Design: Extract visualization & formatting helpers (feature 139)

Status: Draft

Summary
-------
This design note documents the Phase A extraction of small, reusable visualization and
formatting helpers from `examples/classic_interactions_pygame.py` into the
`robot_sf.benchmark` package. The goal is to reduce duplication, improve testability,
and provide a stable API for downstream tooling (plots, videos, and CLI-driven
figure generation).

Scope (Phase A)
----------------
- Pure helpers only: frame size parsing, in-memory overlay text helpers, Markdown
  table formatting.
- No model loading, no environment mutation, and no changes to the public example
  behavior.
- Lazy imports for optional heavy dependencies (matplotlib, moviepy).

Rationale
---------
- Low-risk refactor: small surface area and well-scoped tests.
- Improves reusability: benchmark aggregation and figure generators can reuse
  formatting and frame utilities without importing example-specific code.
- Aligns with repository Constitution: avoids heavy top-level imports and
  enforces explicit inputs and deterministic behavior.

Public API (Phase A)
--------------------
- `robot_sf.benchmark.visualization.frame_shape_from_map(map_svg_path: str) -> tuple[int,int]`
  - Parse SVG width/height or viewBox; raises FileNotFoundError/ValueError on errors.
- `robot_sf.benchmark.visualization.overlay_text(canvas, text: str, pos: tuple[int,int], font: Optional[str]=None) -> None`
  - Duck-typed draw helper that calls `canvas.draw_text(text, pos, font)`.
- `robot_sf.benchmark.utils.format_summary_table(metrics: dict[str,float]) -> str`
  - Returns a Markdown table string from metric-name → value entries. Raises ValueError on empty input.

Testing Strategy
----------------
- Contract unit tests (already added under `tests/unit/benchmark/`) validate API shape and
  common error modes.
- Integration smoke test `tests/unit/benchmark/test_example_dry_run.py` uses
  `run_demo(dry_run=True)` to ensure example-level wiring remains correct.
- Video/encoding helpers use lazy imports; subsequent Phase B may add mocked tests
  for moviepy behaviors.

Migration Notes for Consumers
-----------------------------
- Example code is unchanged from the user's perspective; imports replaced with the new helpers.
- Consumers that previously imported utility code from examples should be updated to
  use `robot_sf.benchmark.visualization` and `robot_sf.benchmark.utils`.

Operational Checklist
---------------------
- [x] Add contract tests for `frame_shape_from_map`, `overlay_text`, `format_summary_table`.
- [x] Implement the helpers in `robot_sf/benchmark/`.
- [x] Add integration dry-run smoke test for the example.
- [ ] Add design note to docs (this file).
- [ ] Add CHANGELOG entry and docs index link.
- [ ] Open PR and link spec + docs.

Risks & Mitigations
-------------------
- Missing optional deps at import time — mitigated by lazy imports and defensive error messages.
- Slight behavior drift in formatting — mitigated by contract tests and example dry-run.

Authors
-------
- Automated feature implementer (branch: `139-extract-reusable-helpers`)

References
----------
- Spec: `specs/139-extract-reusable-helpers/spec.md`
- Tasks: `specs/139-extract-reusable-helpers/tasks.md`
