# Phase 2 Tasks: Integrate Plots & Videos (Planning Output)

Status: Draft task list generated from spec + plan (implementation not yet executed).

## Legend
- [P] Parallelizable
- (T) Test related
- (D) Documentation related

## 1. Orchestrator Integration
1. [X] Inject post-loop hook in `robot_sf/benchmark/full_classic/orchestrator.py` after adaptive convergence to call new `generate_visual_artifacts()` function. [P]
2. [X] Create helper `generate_visual_artifacts(output_root, cfg, episodes_path)` encapsulating plot + video logic, returning (plot_artifacts, video_artifacts, perf_meta). [P]
3. [X] Ensure creation of `plots/` and `videos/` directories if absent (idempotent). [P]
4. [X] Add timing instrumentation around plot + video generation (wall time via `time.perf_counter`). [P]

## 2. Plot Generation Wiring
5. [X] Import and call existing plotting entrypoint (adapted to `generate_plots`) returning artifact list with status fields. [P]
6. [X] Wrap plotting in try/except capturing missing matplotlib → emit skipped entries (note) (implemented via conditional import). [P]
7. [X] Ensure placeholder PDFs are produced (or skipped) respecting smoke mode (still attempt). [P]

## 3. Episode Selection & Replay Data
8. [X] Parse in‑memory records list used earlier; select first N (cfg.max_videos) where N > 0 and not disabled/smoke. [P]
9. [ ] Define minimal replay adapter extracting positions & orientation; if required fields missing, mark all videos skipped (note: "insufficient replay state"). [P] (Deferred – placeholder SimulationView path; real replay adapter future task)

## 4. Video Rendering (SimulationView-first)
10. [X] Attempt import of `SimulationView`; flag `_SIM_VIEW_AVAILABLE` else fallback. [P]
11. [ ] Implement real replay render & mp4 encoding. (Deferred – current placeholder returns empty, triggers synthetic fallback) [P]
12. [ ] MoviePy absence conditional skip (Not explicitly implemented; synthetic fallback currently handles). [P] (Note: treat as satisfied by graceful fallback, refine later if moviepy path added)
13. [X] If SimulationView unavailable or returns empty list: call synthetic generator with renderer=synthetic. [P]
14. [X] Ensure artifact filenames deterministic: `video_<episode_id>.mp4`. [P]

## 5. Manifests & Performance Meta
15. [X] Serialize plot artifacts to `reports/plot_artifacts.json`. [P]
16. [X] Serialize video artifacts to `reports/video_artifacts.json`. [P]
17. [X] Write `reports/performance_visuals.json` with over-budget booleans. [P]
18. [ ] Schema validation step (Optional) not implemented (deferred). [P]

## 6. Tests (pytest)
19. [X] Test: videos disabled → video manifest entries skipped with 'disabled'. (T)
20. [X] Test: missing matplotlib → plot entries skipped. (T)
21. [X] Test: smoke mode → videos skipped with 'smoke mode'. (T)
22. [X] Test: SimulationView empty (simulated) fallback uses synthetic with renderer=synthetic. (T)
23. [ ] Test: successful SimulationView rendering path (Not implemented; pending real replay implementation). (T)
24. [X] Test: deterministic ordering (two runs same selection). (T)

## 7. Documentation & Changelog
25. [X] Update `CHANGELOG.md` under Added: benchmark visual artifact integration. (D)
26. [X] Add docs index link from `docs/README.md` Latest Updates. (D)
27. [ ] Add/update dedicated benchmark doc section (Not done; optional). (D)

## 8. Refactors / Cleanup
28. [X] Ensure no print statements; logging used. [P]
29. [X] Type annotate new functions & docstrings referencing FR IDs in `visuals.py`. [P]
30. [X] Run quality gates (ruff, tests; type check executed – unrelated legacy diagnostics remain). [P]

## 9. Post-Implementation Validation
31. [X] Execute quickstart / benchmark run manually (confirmed manifests creation). [P]
32. [X] Record performance numbers & confirm within soft budgets (runtime meta generated). [P]

## Acceptance Mapping (Traceability)
- FR-001/005: Tasks 5,15
- FR-002/002a/003/004/007/012/013/014/015: Tasks 8-14,16-18,24
- FR-006/010/011: Tasks 6,12,19-23
- FR-008: Excluded from schema changes (verify 28)
- FR-009: Tasks 3,15,16

## Open Questions (None)
All clarifications resolved during research.
