# FXB controller correction plan

Evidence-critical scope: confirm F1–F3 at PR #9926's latest head, correct timestep
resolution and ORCA differential-drive adaptation, and bind an explicit ORCA
0.0.8 config. Keep frozen and historical release artifacts unchanged. Do not
change the social-force pedestrian method; report the current selection.

1. Run the real map runner before edits for social_force and ORCA: three named
   scenarios, seeds 1001–1010, H600, dt 0.1, one simulation process. Capture
   resolved timesteps, commands, headings, physical speeds, and episode metrics.
2. Add regressions with pre-fix failure evidence, then implement the root-cause
   corrections. Require finite positive observed dt for timestep-dependent
   planners. Preserve flat observation bytes for learned-policy compatibility.
3. Repeat the same diagnostic runs; inspect outcome and action differences.
4. Run relevant tests with `pytest -n0`, OMP/OpenBLAS threads set to one, lint,
   format, and inspect the final diff. Deliver a draft PR against the pinned
   parent branch, explicit-path commit and explicit-refspec push.

Evidence is diagnostic-only, with no held-out inference or release admission.
Native ORCA must run without fallback. Store raw evidence outside `output/`,
and commit compact measurements plus reproduction instructions. Retain base
SHA, config hashes and dependency identity. On interruption, resume from saved
episode records; never treat missing or failed rows as successful evidence.

## FXB2 review repairs

Reuse the clean worktree at reviewed head `39d9932efb802e4f340502f6fb8d4acfcab20795`.
Accept and record top-level `dt` after nested/flat simulation time, preserving
fail-closed validation. Add source/precedence, SF/ORCA missing-clock and isolated
occupancy-penalty tests; prove the dt defect at that head and the coverage gaps
against pre-FXB adapter code. Replace four non-dev episode tests with fixed-code
dev calibration (1001–1003), freeze a 25% spin-count margin, then validate named
nodes only. Preserve compact calibration with source hashes in the test fixtures.
Normalize trailing whitespace in the historical failure log and update its digest.
Run the seed-audited focused selection, characterization and new dev tests,
ruff and full PR diff checks. Commit explicit paths and push the existing branch
without force; read back the remote SHA. No main, merges or comments. Full
readiness remains outside this lane because unrelated tests use prohibited seeds.
