# Social Navigation Benchmark — Master TODO

Guiding objective: Release a reproducible, force-field–aware social navigation benchmark (scenarios, metrics, composite index, baselines, scripts) suitable for a short paper / dataset+benchmark track submission.

Update etiquette:
- Mark progress with `[x]`.
- When partially done, append a short status note (e.g. `(impl 70%)`).
- Add date tags for completed milestones (YYYY-MM-DD).
- Keep scope creep isolated in the Stretch section.

---
### Recent Updates (2025-09-16)
### Recent Updates (2025-09-17)
### Recent Updates (2025-09-18)
- [x] Pareto fronts plotting (CLI + programmatic + docs + example) (2025-09-18)
- [x] Deterministic seeding utilities (+ CLI debug-seeds) and tests (2025-09-18)
- [x] SNQI ablation enhancements: `--top` and `--summary-out` (docs updated) (2025-09-18)
- [x] Distribution plots (CLI + docs) (2025-09-18)
- [x] Figure generation script (Pareto, distributions, baseline table) (2025-09-18)
- [x] Scenario montage thumbnails (module + CLI + tests + docs + figure script integration) (2025-09-18) `docs/dev/issues/scenario-montage-thumbnails/todo.md`
- [x] Force-field figure (heatmap + quiver) parameterized and wired into orchestrator; LaTeX include snippets added (2025-09-18)
- [ ] ORCA baseline decision: Deferred until after initial research runs. See `adding_orca.md` for plan & rationale (2025-09-18)
- [x] Define benchmark scope statement (≤150 words) and success criteria (2025-09-02)
- [ ] Choose target venues & submission deadlines (ICRA / IROS / CoRL / NeurIPS D&B)
[x] Caching layer to avoid recomputing unchanged episodes (resume + manifest sidecar with identity hash) (2025-09-17)
- [x] Scenario matrix JSON Schema + CLI validate-config subcommand (2025-09-17)
- [x] CLI: list-scenarios subcommand (2025-09-17)
- [x] Global flags wired end-to-end: seed, progress bar, resume (2025-09-17)
- [x] Progress bar integration for run/baseline via callback (tqdm optional) (2025-09-17)
- [ ] Finalize baseline algorithm list (min: SF default, RL policy, Random, Optional: ORCA/RVO)

[x] Global flags: seed, progress bar, resume (2025-09-17)
The Social Navigation Benchmark provides a reproducible, force-field–aware evaluation suite for robot policies operating amid dynamic pedestrian crowds. It offers a standardized set of procedurally generated scenarios varying density, flow patterns, obstacle complexity, and group behavior. Beyond traditional success and collision counts, it emphasizes comfort and social compliance via force, proximity, and smoothness metrics, aggregated into a transparent composite index (SNQI). The benchmark supplies baseline planners (social-force, RL, random) with deterministic seeding, locked dependencies, and schema-validated outputs to enable fair comparison, ablation, and rapid iteration. Its objective is not to maximize scenario realism initially, but to establish a rigorous, interpretable, and extensible foundation that can be incrementally enriched (e.g., real data calibration, risk-aware planning) while preserving backward compatibility and reproducibility.

### Success Criteria
- Scenario coverage: ≥ 12 core scenarios spanning density × flow × obstacle classes.
- Metric discriminative power: each baseline differs on ≥ 2 core metrics in ≥ 60% of scenarios.
- Reproducibility: identical aggregate metrics (within floating tolerance) across 3 independent runs/seeds batches.
- Stability: coefficient of variation for success & comfort exposure < 10% across ≥5 seeds per baseline.
- Composite index (SNQI): ranking shifts > 1 position for ≥50% of baselines when any one major metric term is removed (shows component influence).
- Artifact completeness: public repo + schema + lockfile + figure regeneration scripts + minimal usage tutorial.
- CI green run: lint + unit tests + smoke benchmark (≤ 5 min) pass on clean clone.

## 🚦 Release Readiness (Gating Checklist)
Focused list of MUST‑complete items before announcing the benchmark as a stable, reproducible release. Keep this block short; strike through (or check) as each is finished and remove the entire section once all are done.

### A. Core Definition & Baselines
- [ ] Finalize target venues & internal submission timeline (ICRA / IROS / CoRL / NeurIPS D&B) → add dates
- [ ] Finalize baseline set (SF, PPO, Random) + explicit ORCA/RVO decision documented (`adding_orca.md` linked & status line updated)
- [ ] Formal metric specification document (single source: definitions, formulae, units, edge cases)

### B. Metrics & Data Integrity
- [ ] Implement per‑pedestrian force magnitude quantiles (current: only aggregated) + tests
- [ ] Contact / collision stress scenario (narrow corridor or tuned density) ensures non‑zero collision counts appear in seed batch
- [ ] SNQI baseline med/p95 stats persisted to `results/baseline_stats.json` & automatically consumed by orchestrator (documented)

### C. Video Artifacts Minimal Viable Set
- [ ] Micro batch integration test (1–2 episodes) asserts: MP4 exists, size>0, frames == steps
- [ ] `--no-video` (or config toggle) implemented and documented
- [ ] Manifest / JSON Schema extension: video artifact entries (format=mp4, file_size>0)
- [ ] Performance sample recorded (encode ms/frame + % overhead <5%) attached to docs

### D. Reproducibility & Tooling
- [ ] Pre-commit hook / CI guard preventing silent schema drift (metric or episode schema)
- [ ] Figure output canonical naming + `--auto-out-dir` implemented (`<episodes-stem>__<gitsha7>__v<schema>` + `_latest.txt` alias)
- [ ] One-shot regeneration script (episodes → baseline → aggregate → figures) documented (`scripts/generate_full_benchmark.sh` or Python driver)
- [ ] Zenodo deposition draft metadata (title, authors, abstract, keywords) prepared

### E. Validation & Evidence
- [ ] Cross-seed stability run (≥5 seeds) with CV <10% for success & comfort exposure (store JSON summary)
- [ ] Discriminative power table: each baseline differs on ≥2 metrics in ≥60% scenarios (stored under `docs/figures/` or `results/`)
- [ ] SNQI component ablation ranking shift analysis (≥50% baselines move >1 slot) – figure or table

### F. Documentation & Onboarding
- [ ] `video-artifacts/design.md` (capture approach, fallbacks, perf notes) + linked from `docs/README.md`
- [ ] Metrics spec linked from docs index & README
- [ ] Add “Release Reproduction Guide” section: exact commands to regenerate all published artifacts

### G. Paper Readiness (Minimum Pre-outline)
- [ ] Draft outline (sections + bullet list) committed under `docs/paper/outline.md`
- [ ] Populate placeholders in TODO: mark Paper Writing section start date

### Progress Tracking Meta
- [ ] Add date when all items above are checked → then remove this section and announce release readiness milestone.

---

## 1. Scenario & Dataset Specification
- [x] Enumerate scenario dimensions (density, flow pattern, obstacles, groups) (2025-09-02)
- [x] Draft scenario matrix (table of N core scenarios, each with parameter ranges) (see `scenario_matrix.yaml`) (2025-09-02)
- [x] Implement deterministic scenario generator (seeded) (2025-09-02)
- [x] Implement map variants (simple hall, bottleneck, obstacle maze) (2025-09-08)
	- Note: "crossing" currently represented via flow topology, not a distinct obstacle layout.
- [x] Add grouping / crowd heterogeneity flags to scenario config schema (2025-09-17)
- [x] Validate scenario diversity via quick summary script (histograms of min distances, avg speeds) (2025-09-09)
- [ ] (Optional) Import small real trajectory stats (ETH/UCY aggregate) for parameter calibration

## 2. Metric Definition & Implementation
- [ ] Finalize metric list & definitions (formal doc)
	- [x] Success rate (2025-09-10)
	- [x] Time-to-goal / normalized path efficiency (2025-09-10)
	- [x] Collision count / near-miss count (distance < threshold) (2025-09-10)
	- [x] Min / mean interpersonal distance distribution (mean implemented) (2025-09-16)
	- [ ] Force magnitude quantiles (per ped & aggregated) (aggregated implemented; per-ped pending)
	- [x] Force exceedance events (above comfort threshold) (2025-09-10)
	- [x] Comfort exposure time (% of steps above threshold) (2025-09-10)
	- [x] Path smoothness (jerk + curvature stats) (2025-09-10)
	- [x] Robot energy proxy (sum |accel|) (2025-09-10)
	- [x] (Optional) Divergence / gradient norm of force field along path (gradient norm implemented) (2025-09-10)
- [x] Implement metric computation module (`metrics/`) (2025-09-08)
- [x] Add unit tests for key metrics (edge cases; expand as needed) (2025-09-08)
- [x] Define JSON schema for per-episode metric output (2025-09-02)

## 3. Composite Index (SNQI)
- [x] Draft formula (weighted normalized metrics) (2025-09-08)
- [x] Implement normalization strategy (percentile baseline: median/p95) (2025-09-08)
	- [x] Provide script to recompute weights / sensitivity analysis (CLI: `snqi optimize|recompute`) (2025-09-16)
- [x] Ablation: measure discriminative power with and without each component (2025-09-17)

## 4. Configuration & CLI Harness
- [x] YAML/JSON schema for benchmark suite definition (list of scenario specs + repetitions) (2025-09-17)
- [x] Implement CLI: `robot_sf_bench run ...` (uses `--matrix` instead of `--suite`; includes `--algo`, `--snqi-*`, `--quiet`, `--fail-fast`) (2025-09-16)
- [x] Subcommand: `robot_sf_bench list-scenarios` (2025-09-17)
- [x] Subcommand: `robot_sf_bench list-algorithms` (2025-09-16)
- [x] Subcommand: `robot_sf_bench validate-config` (2025-09-17)
- [x] Global flags: seed, progress bar, resume (2025-09-17)
- [x] Parallel workers flag (`--workers`) in CLI and Python API (2025-09-16)
- [x] Logging: structured (JSONL) plus human-readable summary (per-run JSON summary + plots via `summary`) (2025-09-16)
	- [x] Python API batch runner `run_batch(...)` with JSONL writing and schema validation (2025-09-08)
	- [x] CLI baseline subcommand (`robot_sf_bench baseline`) to produce baseline med/p95 JSON (2025-09-08)

## 5. Baseline Algorithm Integrations
- [x] Standard Social Force planner wrapper (2025-09-12)
- [x] Existing RL policy loader (PPO model) adapter (2025-09-17)
- [x] Random / naive reactive baseline (RandomPlanner + CLI + tests) (2025-09-17)
- [ ] (Optional) ORCA integration (licensing check)
	- Status: Deferred (see `adding_orca.md`)
- [x] Unified interface (step(obs) -> action) with timeouts / safety clamp (runner timeout + final clamp) (2025-09-17)
- [x] Per-baseline config file (hyperparameters, seeds) via `--algo-config` YAML (2025-09-16)

## 6. Evaluation Pipeline & Aggregation
- [x] Aggregation script: merges JSONL episodes -> metrics CSV + SNQI per algo (2025-09-08)
- [x] Confidence interval computation (bootstrap or across seeds) (episode-level CIs via aggregate --bootstrap; SNQI bootstrap options available) (2025-09-17)
- [x] Seed variance analysis script (CLI: seed-variance; docs added) (2025-09-17)
- [x] Failure case extractor (collisions/low comfort) (CLI: extract-failures) (2025-09-17)
- [x] Automatic ranking table generator (Markdown/CSV) (2025-09-17)
- [x] Failure case extractor (episodes with collisions / low comfort) (duplicate; see above) (2025-09-17)
- [x] Caching layer to avoid recomputing unchanged episodes (resume + manifest) (2025-09-17)

## 7. Visualization & Reporting Assets
- [x] Force field heatmap + vector overlays figure (script + docs + PDFs + orchestrator) (2025-09-18)
- [x] Distribution plots (CLI and docs) (2025-09-18)
- [x] Pareto fronts (Time vs. Comfort, Collisions vs. SNQI) (2025-09-18)
- [x] Scenario montage thumbnails (2025-09-18)
- [x] Baseline comparison table auto-generation (CLI + docs) (2025-09-18)
- [x] Scripts to regenerate paper figures (initial: pareto, dists, table) (2025-09-18)

### 7a. Episode Video Artifacts (NEW 2025-09-22)
Goal: Provide lightweight, reproducible per-episode MP4s for qualitative inspection without introducing backend‑specific fragility or large runtime overhead.

Completed (2025-09-22):
- [x] Backend‑agnostic frame capture via in‑memory PNG fallback (replaces brittle `tostring_rgb` usage). (2025-09-22)
- [x] Removed abandoned ARGB / HiDPI buffer manipulation path (simpler + portable). (2025-09-22)
- [x] Fixed `moviepy` invocation (dropped unsupported `verbose` / `logger` kwargs). (2025-09-22)
- [x] Integrated video generation in classic benchmark script producing per‑episode MP4 artifacts. (2025-09-22)

Planned / Pending:
- [ ] Add integration test: run a 1–2 episode micro batch and assert MP4 exists, >0 bytes, and frame count matches expectation.
- [ ] Performance sampling: record encode ms/frame & cumulative overhead; target <5% added wall time for default batch.
- [ ] Add `--no-video` / config toggle to disable video generation for pure metric runs.
- [ ] Add renderer selection flag (`--video-renderer=synthetic|sim-view|none`) anticipating SimulationView path.
- [ ] Optional SimulationView high‑fidelity renderer (deferred until post core benchmark freeze).
- [ ] JSON Schema extension / manifest spec for video artifact entries (validate size >0, format=mp4).
- [ ] Documentation: create `docs/dev/issues/video-artifacts/design.md` (rationale, capture method, perf notes) and link from `docs/README.md` & this TODO.
- [ ] Benchmark figure regeneration script: add optional gallery sheet (contact-sheet style) of first frame per video.
- [ ] Perf regression guard: add lightweight CI smoke that runs video generation for 1 episode (skipped on Windows if needed).

Notes:
- PNG fallback ensures portability across macOS (FigureCanvasMac), Agg, and headless CI without backend feature assumptions.
- Future SimulationView integration should reuse the same writer abstraction; avoid duplicating encode logic.

### Next picks (2025-09-18) — Research kickoff
1) Lock scenario matrix for the “core” suite (12+ scenarios) and commit final YAML.
2) Run baseline batch for SF, PPO, Random with resume enabled; store JSONL under `results/episodes.jsonl`.
3) Compute baseline stats (med/p95) for SNQI normalization and save to `results/baseline_stats.json`.
4) Generate figures and tables via `scripts/generate_figures.py` into `docs/figures/`.
5) Seed variance check: run `seed-variance` and record summary JSON (attach to docs).
6) Discriminative check: verify metric separation criteria across baselines; record quick table in docs.
7) Decision: ORCA inclusion — proceed or defer based on licensing/time; update Section 5 accordingly.

### Next steps (2025-09-18) — Additions after long-batch fix
- [ ] Contact sanity validation: add a narrow-corridor or crossing scenario (or temporarily set `D_COLL=0.35`) to exercise the collision metric end-to-end and confirm non-zero collision counts appear. Wire into scenario matrix with a small repetition count (e.g., 3) for quick checks.
- [ ] SNQI baselines: compute per-metric median/p95 per baseline and persist to `results/baseline_stats.json`; update orchestrator and docs to read these when rendering SNQI-normalized figures/tables.
- [ ] Figures output naming: migrate to canonical folder names under `docs/figures/` using the pattern `<episodes-stem>__<gitsha7>__v<schema>`. Example: `docs/figures/episodes_sf_long_fix1__a1b2c3d__v1/`. Maintain a light alias via `docs/figures/_latest.txt` pointing to the recommended folder for the paper draft.
- [ ] Orchestrator flag: add `--auto-out-dir` to `scripts/generate_figures.py` to compute the canonical output folder name automatically from inputs and git state; accept `--out-dir` to override.
- [ ] Cleanup artifacts: remove stray logs and ad-hoc exports outside `results/` and `docs/figures/`. Add a pre-commit check that blocks new top-level non-source files (allowlist-based).
- [ ] Docs update: add a short design note describing the figures naming and alias approach (see `docs/dev/issues/figures-naming/design.md`) and link it from `docs/README.md`.

## 8. Reproducibility & Packaging
- [x] Deterministic seeding audit (numpy, torch, random) (2025-09-18)
- [x] Version stamping (git hash + config hash in outputs) (2025-09-08)
- [x] Lock dependencies (uv lock confirmed) (2025-09-02)
- [x] Provide `environment.md` / exact reproduction instructions (2025-09-09)
- [ ] Pre-commit hook to validate metric schema changes
- [x] Lightweight CI pipeline (lint + unit tests + sample benchmark smoke test) (2025-09-09)
- [ ] Zenodo deposition draft (metadata prepared)

## 9. Paper Writing Tasks
- [ ] Outline (sections & bullet points)
- [ ] Related work pass (benchmarks in social navigation, force-based metrics)
- [ ] Method section draft (scenarios, metrics, SNQI)
- [ ] Experimental setup description
- [ ] Results & discussion with ablations
- [ ] Threats to validity / limitations
- [ ] Artifact & reproducibility statement
- [ ] Final figure polish
- [ ] Submission formatting & checklist

## 10. Validation & Ablations
- [ ] Metric sanity tests (manually crafted edge episodes)
- [ ] Sensitivity: vary density → observe monotonic comfort change
- [ ] SNQI weight sensitivity plot
- [ ] Remove each metric from SNQI → Δ ranking analysis
- [ ] Cross-seed stability (≥5 seeds / baseline)
- [ ] Runtime profiling (episodes/sec per baseline)

## 11. Stretch / Optional Enhancements
- [ ] Force divergence & curl metrics (interpretability)
- [ ] Residual learned predictor baseline (SF + small MLP)
- [ ] Bayesian parameter calibration for SF (posterior summary only)
- [ ] Risk-aware planner variant (chance-constrained) baseline
- [ ] Minimal web dashboard (interactive metric explorer)
- [ ] User study design draft (comfort perception) — if time

## 12. Risk Log
- [ ] Scenario diversity insufficient → mitigation: expand parameter grid
- [ ] Metrics not discriminative → mitigation: add gradient/jerk or comfort exposure variants
- [ ] Time constraints on RL training → mitigation: reuse existing trained checkpoints
- [ ] ORCA licensing complexity → mitigation: exclude or reimplement simplified reciprocal avoidance
- [ ] Compute resource limits → mitigation: batch evaluation + caching layer

## 13. Immediate Next Actions (Pick 3 to start)
- [x] Draft scenario dimension list (see `scenario_dimensions.md`) (2025-09-02)
- [x] Write metric formal definitions doc stub (see `metrics_spec.md`) (2025-09-02)
- [x] Implement JSON schema draft for episode output (see `episode_schema.json`) (2025-09-02)

Next picks (2025-09-08):
- [x] Baseline normalization data: script to run `run_batch` over a baseline policy suite, compute per-metric med/p95, persist as JSON for SNQI (2025-09-08)
- [x] Minimal CLI shim `robot_sf_bench run` that wraps `run_batch` (matrix path, out path, seeds, flags) (2025-09-08)
- [x] Scenario diversity summary script: quick histograms (min distances, avg speeds) as sanity check (2025-09-09)
- [x] Lightweight CI job: lint + unit tests + tiny batch run as smoke test (2025-09-09)

---
Last updated: 2025-09-22 (Video artifact pipeline + PNG fallback; moviepy invocation fix; added video subsection)

