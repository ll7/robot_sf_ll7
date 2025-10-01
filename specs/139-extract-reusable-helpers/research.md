# Phase 0 — Research: Extract reusable helpers (feature 139)

Date: 2025-09-29

Decision: Proceed with a small, low-risk Phase A extraction consisting of pure helpers only (frame/recording, plotting/overlay, formatting/table helpers).

Rationale:
- Minimizes blast radius: these helpers are predominantly pure functions or small adapters that depend minimally on simulation state.
- Improves testability: isolating formatting and frame helpers enables concise unit tests and reduces reliance on heavy optional deps like moviepy or SimulationView during tests.
- Aligns with Constitution principles: preserves reproducibility (I), testing requirements (IX), and avoids heavy runtime imports at module top-level (XII guidance on lazy imports).

Alternatives considered:
- Big-bang extraction of all helpers including model and map loaders — rejected due to high dependency surface and higher risk of introducing runtime errors and CI flakes.
- Postpone extraction until a larger refactor — rejected because small extractions unblock reuse and reduce duplication now.

Open questions (resolved):
- Phase A contents: resolved via clarification session 2025-09-29 (included in spec). 

Next research tasks (Phase 0 outputs that drove Phase 1):
- Enumerate helper functions in `examples/classic_interactions_pygame.py` and identify pure adapters.
- Determine minimal public signature for visualization helpers (inputs/outputs and error modes).

References:
- Spec: `/Users/lennart/git/robot_sf_ll7/specs/139-extract-reusable-helpers/spec.md`
