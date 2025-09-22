# Phase 0 Research: Enhance Benchmark Visual Artifacts

Purpose: Resolve all unknowns and solidify decisions for SimulationView replay rendering, streaming MP4 encoding, JSON Schema validation, and performance/memory instrumentation per spec FR-001..FR-020.

## Inputs
- Feature Spec: `spec.md`
- Constitution: `.specify/memory/constitution.md` (Reproducibility, determinism, contracts, documentation, test coverage)

## Requirements Mapping (Condensed)
- Video path upgrade (FR-001, FR-002, FR-007, FR-013, FR-014, FR-015–FR-017)
- Replay adapter creation (implicit for FR-001, FR-008)
- Determinism (FR-003) and filename reproducibility (FR-014)
- JSON Schemas (FR-004–FR-006, FR-009)
- Performance + memory budgets (FR-009–FR-012)
- Documentation & demo updates (FR-018–FR-019)
- Non‑regression of existing data contracts (FR-020)

## Unknowns Inventory & Resolution
| Area | Unknown / Prior Gap | Resolution | Alternatives Considered | Rationale |
|------|---------------------|------------|--------------------------|-----------|
| Replay Data Source | Where to obtain per-timestep robot + pedestrians for video | Use existing episode logs / env info captured during benchmark loop; add lightweight capture hook if not present | Full state re-sim vs stored log | Avoid divergence; log once, reuse for rendering (Constitution: determinism) |
| Adapter Interface | Fields minimal set? | `positions: list[(x,y)]`, `headings: list[float]`, `timestamps: list[float]`, plus optional `ped_positions: list[list[(x,y)]]]` | Rich physics state (forces) | Keep schema lean, avoid bloat; forces not required for visualization FR scope |
| SimulationView Availability | How to detect? | Try import + simple construct; cache boolean | Lazy per video attempt | Single upfront probe yields deterministic skip reason |
| MoviePy / ffmpeg | Handling detection & failure | Import moviepy.editor; attempt write of 1-frame temp if uncertain; on ImportError mark moviepy-missing | Defer error until encode call | Early classification improves manifest clarity |
| Streaming Encoding | Mechanism to avoid frame accumulation | Use generator yielding frames from SimulationView `render_frame(step_data)`; pipe into moviepy VideoClip via custom `make_frame` referencing underlying replay arrays; encode directly | Pre-buffer frames into list | Streaming honors FR-012 memory target |
| Peak Memory Measurement | How to measure lightweight? | Sample RSS via `psutil.Process().memory_info().rss` before & after encode; track max delta; log in performance_visuals.json (optional field) | tracemalloc (Python alloc only) | RSS inclusive of libraries, closer to real memory footprint |
| Performance Timing Granularity | Need per-video or first-video only? | Record total plots time + first successful (or attempted) SimulationView video time; additional videos optional | Time every video | Spec only mandates first video; reduces overhead |
| Schema Validation Mode | When to validate? | Always run in dev/test path (if jsonschema installed). In production runtime, skip unless env var `ROBOT_SF_VALIDATE_VISUALS=1` | Always validate | Avoid runtime cost for typical benchmark runs |
| Skip Notes Canonicalization | Guarantee stable note strings | Define constants module `visuals/constants.py` enumerating notes | Inline literals | Central constants reduce typo risk; supports tests |
| Failure Cleanup | Partial MP4 removal | On exception during encoding: if file exists and size < threshold (e.g. 1KB) unlink; mark status failed | Leave artifact | Avoid misleading empty outputs |
| Deterministic Episode Selection | Already deterministic? | Leverage existing ordering list; slice first N eligible episodes after filtering by replay sufficiency | Sort by episode_id | Preserve current invariant (no behavioral change) |
| Smoke Mode Interaction | How skip labeled? | Use note `smoke-mode` before any heavy operation | Use generic skip | Clear classification improves debugging |
| Memory Target Formalization | Cap vs soft flag | Soft target < 100MB; if peak > target set `memory_over_budget=true` inside performance_visuals.json | Hard fail | Soft aligns with spec (flag only) |
| Pedestrian Trajectories | Needed for MVP? | If available cheaply include; else video still valid with robot path + environment backgrounds | Mandatory inclusion | Optional to keep adapter simple now; future enhancement possible |
| Render Frame Rate | Fixed or dynamic? | Fixed 10 FPS (constant) for consistency across videos | Derive from simulation timestep | Stable reproduction, predictable encode time |

## Key Decisions
1. Replay Adapter Contract (initial internal structure, not public schema):
   - Python dataclass `ReplayState(episode_id, positions, headings, timestamps, ped_positions=None)`
   - Validation: lengths equal; timestamps strictly increasing; >=2 frames
2. SimulationView Rendering:
   - Attempt once globally; if available, create a lightweight headless surface per encode.
   - `make_frame(t)` maps t (seconds) → nearest replay index via precomputed `frame_times` (linear map by constant FPS).
3. Encoding Pipeline:
   - Use MoviePy `VideoClip(make_frame, duration=...)` with `write_videofile` and parameters: `codec="libx264"`, `fps=10`, `audio=False`, `preset="ultrafast"` to minimize time, pixel format default.
   - If MoviePy absent, skip with note.
4. Memory Instrumentation:
   - Wrap encoding in sampler capturing max RSS at 50ms interval in a background thread; join after encode; compute delta; store `memory_peak_mb` optional field.
5. Schemas:
   - `video_artifacts.schema.json` enumerates: episode_id (string/int), filename (string nullable), renderer enum [simulation_view, synthetic], status enum, note enum (explicit list), optional memory/time fields not required for each entry.
   - `plot_artifacts.schema.json` similar pattern.
   - `performance_visuals.schema.json` includes timing floats, boolean flags, optional memory fields.
6. Validation Strategy:
   - If `jsonschema` installed import and validate; errors raise during tests; in runtime log warning and continue.
7. Constants Module for Notes & Enums to ensure test alignment.
8. Documentation Section will include dependency matrix: (pygame, moviepy, jsonschema) vs outcome (success, skip reason, fallback), plus performance instrumentation notes.
9. Demo Script Update: `examples/demo_full_classic_benchmark.py` gains mention of real rendering path and how to force fallback (unset pygame, set env var to disable?).

## Reproducibility & Constitution Alignment
- Determinism preserved: episode selection unchanged; FPS constant ensures identical frame indexing.
- No benchmark schema change (FR-020) — manifests are auxiliary artifacts only.
- Documentation & tests ensure transparency (Principles II, III, VIII, IX).
- Memory & performance instrumentation non-intrusive and optional flags do not alter core outputs.

## Risk & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| MoviePy import slowdown | Longer startup | Lazy import inside video path only |
| Pygame surface allocation fails headless | No videos | Fallback synthetic path preserved |
| Memory sampler overhead | Minor CPU usage | Low frequency sampling (20 Hz), disable if psutil missing |
| Frame rounding mismatch | Drift in video length | Precompute frame_count from replay length; t→index clamp |
| Schema drift vs implementation | Validation failures | Co-develop schemas + constants; add schema validation tests |

## Open Items (Deferred or Future Enhancements)
- Optional inclusion of pedestrian trajectories if not easily accessible now (document either way in data-model.md)
- Potential adaptive FPS based on simulation timestep (out of current scope)

No remaining NEEDS CLARIFICATION. Phase 0 complete.
