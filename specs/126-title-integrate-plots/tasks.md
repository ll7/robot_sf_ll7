# Phase 2 Tasks: Integrate Plots & Videos (Planning Output)

Status: Draft task list generated from spec + plan (implementation not yet executed).

## Legend
- [P] Parallelizable
- (T) Test related
- (D) Documentation related

## 1. Orchestrator Integration
1. Inject post-loop hook in `robot_sf/benchmark/full_classic/orchestrator.py` after adaptive convergence to call new `generate_visual_artifacts()` function. [P]
2. Create helper `generate_visual_artifacts(output_root, cfg, episodes_path)` encapsulating plot + video logic, returning (plot_artifacts, video_artifacts, perf_meta). [P]
3. Ensure creation of `plots/` and `videos/` directories if absent (idempotent). [P]
4. Add timing instrumentation around plot + video generation (wall time via `time.perf_counter`). [P]

## 2. Plot Generation Wiring
5. Import and call existing `plots.generate_all(...)` (or similar) updating it if necessary to accept output dir & return artifact list with status fields. [P]
6. Wrap plotting in try/except capturing missing matplotlib → emit skipped entries (note). [P]
7. Ensure placeholder PDFs are produced (or skipped) respecting smoke mode (still attempt). [P]

## 3. Episode Selection & Replay Data
8. Parse `episodes.jsonl` once to collect ordered list of episodes; select first N (cfg.max_videos) where N > 0 and not disabled/smoke. [P]
9. Define minimal replay adapter extracting positions & orientation; if required fields missing, mark all videos skipped (note: "insufficient replay state"). [P]

## 4. Video Rendering (SimulationView-first)
10. Attempt import & headless init of `SimulationView`; on failure set flag fallback to synthetic. [P]
11. For each selected episode attempt replay render to surface frames; encode MP4 (moviepy/ffmpeg). [P]
12. If moviepy missing but pygame present: mark skipped with note (do not partially render). [P]
13. If SimulationView unavailable: call existing synthetic generator (update to return artifact records with renderer=synthetic). [P]
14. Ensure artifact filenames deterministic: `video_<episode_id>.mp4`. [P]

## 5. Manifests & Performance Meta
15. Serialize plot artifacts to `reports/plot_artifacts.json`. [P]
16. Serialize video artifacts to `reports/video_artifacts.json`. [P]
17. If timings measured write `reports/performance_visuals.json` with over-budget booleans. [P]
18. Add schema validation step (optional) using jsonschema in dev/test only (guarded). [P]

## 6. Tests (pytest)
19. Test: run small benchmark stub (mock episodes) with videos disabled → video manifest entries skipped reason contains 'disabled'. (T)
20. Test: missing matplotlib (simulate by monkeypatch import) → plot entries skipped. (T)
21. Test: smoke mode → videos all skipped with 'smoke mode' note; plots present (generated or skipped). (T)
22. Test: when pygame import fails (monkeypatch) fallback uses synthetic with renderer=synthetic. (T)
23. Test: successful path (if dependencies present) produces at least one video with renderer=simulation_view. (T)
24. Test: deterministic ordering (two runs same selection). (T)

## 7. Documentation & Changelog
25. Update `CHANGELOG.md` under Added: benchmark visual artifact integration. (D)
26. Add docs index link from `docs/README.md` to quickstart or spec summary if required. (D)
27. Possibly add a short section to `docs/benchmark_full_classic.md` (if exists) referencing new artifacts. (D)

## 8. Refactors / Cleanup
28. Ensure no print statements; use logging (info-level). [P]
29. Type annotate new functions, add docstrings referencing spec FR IDs. [P]
30. Run quality gates (ruff, type check, tests). [P]

## 9. Post-Implementation Validation
31. Execute quickstart command; verify artifacts & manifests. [P]
32. Record performance numbers & confirm within soft budgets; adjust notes if exceeded. [P]

## Acceptance Mapping (Traceability)
- FR-001/005: Tasks 5,15
- FR-002/002a/003/004/007/012/013/014/015: Tasks 8-14,16-18,24
- FR-006/010/011: Tasks 6,12,19-23
- FR-008: Excluded from schema changes (verify 28)
- FR-009: Tasks 3,15,16

## Open Questions (None)
All clarifications resolved during research.
