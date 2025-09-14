# SNQI Weight Recomputation & Sensitivity – Improvement Checklist

This checklist captures all outstanding improvements required to bring the SNQI weight recomputation & sensitivity analysis PR in line with the repository's Definition of Done and quality standards.

> Design doc: See the companion architectural draft in [DESIGN.md](./DESIGN.md) for context, rationale, data contracts, and planned refactors. Keep both files synchronized when items are completed.

---
## Legend
- [ ] Not started
- [x] Done (check off during execution)
- (🔧) Indicates refactor / structural change
- (🧪) Indicates test-related work
- (📄) Documentation-focused
- (⚙️) Tooling / integration
- (🎯) Functional correctness / methodology

---
## 1. Design & Documentation (📄)
- [ ] Create full design doc (convert / extend this into design doc) including:
  - [ ] Context, goals, non-goals
  - [ ] Rationale for weight strategies & chosen objective proxies
  - [ ] Alternatives considered (e.g., true multi-objective NSGA-II vs heuristic Pareto sampling)
  - [ ] Data / file contract definitions (episodes.jsonl, baseline_stats.json, output JSON schemas)
  - [ ] Error modes and fallback behavior
  - [ ] Reproducibility & seeding strategy
  - [ ] Performance characteristics (expected runtime vs dataset size)
- [ ] Relocate user-facing docs from `scripts/QUICK_START.md` & `scripts/README_SNQI_WEIGHTS.md` into `docs/snqi-weight-tools/README.md` (or link them if partial duplication needed)
- [ ] Add short section + link in root `README.md` describing SNQI tooling
- [ ] Add cross-reference in `docs/ENVIRONMENT.md` or benchmark docs if applicable
- [ ] Document JSON output schema formally (key names, types, meaning, stability guarantee)
- [ ] Explain normalization strategy choice (median/p95) and limitations
- [ ] Justify heuristic formulas (e.g., stability = 1/(1+|std-0.5|)) and consider replacing or labeling experimental
- [ ] Add docstrings (module, classes, functions) per repo style to all new scripts
- [ ] Add doc comment on clamping normalized metrics to [0,1] (implication for outliers)

## 2. Code Structure & Refactor (🔧)
- [ ] Extract shared SNQI computation logic into a single module (e.g. `robot_sf/benchmark/snqi.py` or reuse existing metrics implementation)
- [ ] Replace duplicated `compute_snqi` in all scripts with import from shared module
- [ ] Centralize `WEIGHT_NAMES` constant
- [ ] Factor large `main()` functions (currently ignored via `# noqa: C901`) into smaller helpers (arg parsing, IO, compute, reporting)
- [ ] Remove placeholder values (`pareto_efficiency: 0.8/0.9`) or compute real metric or drop field
- [ ] Add defensive validation of weights file schema (missing keys, wrong types)
- [ ] Add CLI option `--seed` to all stochastic scripts
- [ ] Add progress indication for long loops (grid search, pairwise surfaces) using `tqdm`
- [ ] Add bounds/guard for grid explosion (warn if combinations > threshold)
- [ ] Unify logging style (avoid mixed print/logging; reserve print for final summaries)
- [ ] Normalize return codes & `if __name__ == "__main__"` patterns
- [ ] Consider exposing core functionality as functions that accept data structures (facilitates reuse in notebooks/tests)

## 3. Testing (🧪)
- [ ] Add unit test ensuring recomputed SNQI matches canonical implementation for a fixture episode set
- [ ] Test each strategy output contains required keys & weight ranges
- [ ] Test Pareto sampling is deterministic under fixed seed
- [ ] Test sensitivity analysis ranking correlation monotonic behavior for controlled perturbations
- [ ] Test CLI entry points (use `pytest` `script_runner` or subprocess) returning 0 and producing expected output schema
- [ ] Add regression test for JSON schema (snapshot or structured validation via `jsonschema`)
- [ ] Add test for handling missing optional metrics (e.g., jerk) without crashing
- [ ] Add test for malformed JSONL line skip counting
- [ ] Add test verifying `--compare-normalization` correlations in [0,1] and base strategy present

## 4. Benchmark / CLI Integration (⚙️)
- [ ] Add unified subcommand to existing benchmark CLI (`robot_sf_bench snqi <action>`) wrapping scripts
- [ ] Provide config-based invocation (so weights can be referenced in future runs)
- [ ] Wire optimized weights consumption into evaluation pipeline (document how to pass them)
- [ ] Optionally add a convenience command to apply weights & recompute SNQI over stored results

## 5. Reproducibility & Determinism (🎯)
- [ ] Global seeding support: apply `--seed` to numpy & differential evolution
- [ ] Record seed in output JSON
- [ ] Document non-deterministic elements if any remain (e.g., parallelism inside SciPy)

## 6. Methodology Improvements (🎯)
- [ ] Consider more principled stability metric (e.g., average pairwise Spearman between bootstrap resamples)
- [ ] Consider true Pareto front via multi-objective algorithm (optional follow-up)
- [ ] Add option to constrain sum of weights or normalize them to a simplex
- [ ] Expose objective component breakdown in output (stability vs discriminative power contributions)
- [ ] Provide option to optimize for weighted scenarios (e.g., denser crowd episodes weighted more)
- [ ] Add early stopping for evolution if objective plateaus
- [ ] Provide confidence intervals via bootstrapping (optional)

## 7. Performance & Scalability
- [ ] Add timing instrumentation (per major phase) in verbose mode
- [ ] Add note/warning if dataset size small (< N episodes) affecting stability metrics
- [ ] Allow sampling subset of episodes (`--sample N`) for quick exploratory runs

## 8. Dependency & Extras Management
- [ ] Add optional extra group `analysis` or `viz` for `seaborn` (currently optional import)
- [ ] Update docs to use `uv add` / `uv sync` instead of raw `pip install`
- [ ] Document minimal dependency set for headless (no plots) mode

## 9. Output & Schema Hygiene
- [ ] Add `schema_version` field to JSON outputs
- [ ] Include generation timestamp & git commit hash (if accessible) in outputs
- [ ] Include command-line args echo in output for provenance
- [ ] Validate numeric fields are finite (no NaN) before writing JSON
- [ ] Add compact summary block (top weights, stability, discriminative power)

## 10. Error Handling & UX
- [ ] Aggregate and report count of skipped / invalid JSONL lines
- [ ] Provide helpful message if baseline lacks required metrics
- [ ] Add `--fail-on-missing-metric` toggle
- [ ] Add graceful message if matplotlib missing but user requested plots
- [ ] Provide exit code differentiation (e.g., 2 = input error, 3 = runtime failure)

## 11. Style & Consistency
- [ ] Ensure all new code passes type checking (add type hints where missing)
- [ ] Ensure ruff clean with no inline suppressions except justified (document rationale)
- [ ] Add module-level `__all__` where appropriate for public surfaces
- [ ] Normalize naming: `near_misses` vs `w_near` (ensure doc clarity)

## 12. Follow-Up (Not blocking initial merge but recommended)
- [ ] Add notebook example leveraging the shared module (replaces ad-hoc exploration)
- [ ] Provide a small synthetic fixture dataset under `tests/data/` for deterministic tests
- [ ] Evaluate storing canonical recommended weights in repo (version-controlled) with provenance
- [ ] Add continuous benchmark (optional GitHub Action) exercising optimization on a small sample for drift detection

---
## Acceptance Criteria Mapping
| Category | Must for Merge | Follow-Up |
|----------|----------------|-----------|
| Design Doc | Yes | — |
| Core Tests | Yes | — |
| Refactor shared SNQI | Yes | — |
| CLI Integration | Prefer (can defer) | Yes if deferred |
| Docs Relocation | Yes | — |
| Methodology Enhancements | Partial | Remaining |
| Optional Extras Group | Yes (dependency clarity) | — |
| Advanced Metrics (bootstrap CI) | No | Yes |

---
## Next Step Recommendation
Start with: shared module refactor → tests → docs relocation+design doc → CLI integration → seeding & schema improvements → methodology refinements.

---
Generated: (initial scaffold — to be updated as work proceeds)
