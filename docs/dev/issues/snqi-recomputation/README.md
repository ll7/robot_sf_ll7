# SNQI Weight Recomputation & Sensitivity – Improvement Checklist

This checklist tracks progress against the modernization, validation, and methodology roadmap for SNQI weight optimization & sensitivity analysis.

> Design doc: See [DESIGN.md](./DESIGN.md) for architecture, objectives, data contracts, and rationale. Keep statuses synchronized.

---
## Legend
- [ ] Not started
- [x] Done
- (🔧) Refactor / structural
- (🧪) Tests
- (📄) Docs
- (⚙️) Tooling / integration
- (🎯) Methodology / correctness

---
## 1. Design & Documentation (📄)
- [x] Full design doc (context, goals, alternatives, contracts, error modes, seeding, perf expectations)
- [x] Relocate user-facing docs into `docs/snqi-weight-tools/`
- [x] Root README section referencing SNQI tooling
- [x] Cross-reference in benchmark/environment docs
- [x] Document JSON output schema formally (key names, types, stability) <!-- See docs/snqi-weight-tools/schema.md and snqi_output.schema.json -->
- [x] Explain normalization strategy (median/p95) & limitations <!-- See docs/snqi-weight-tools/normalization.md -->
- [ ] Justify heuristic formulas (stability proxy) or mark experimental <!-- DESIGN.md partial -->
- [ ] Comprehensive docstrings for new helpers/modules <!-- Some helpers missing param/return sections -->
- [ ] Note on clamping normalized metrics to [0,1] and outlier implication

## 2. Code Structure & Refactor (🔧)
- [x] Shared computation module (`compute.py`)
- [x] Remove duplicate `compute_snqi` implementations
- [x] Centralize `WEIGHT_NAMES`
- [x] Decompose large CLI functions into helpers (complexity isolated)
- [x] Remove placeholder metrics/fields
- [x] Weight file schema validation (`weights_validation.py`)
- [x] Global `--seed` support in all scripts
- [x] Progress indication (optional `tqdm` flag)
- [x] Adaptive grid bounds & combination cap (sampling) to prevent explosion
- [ ] Unify logging vs prints (prints restricted to final summaries) <!-- Mixed usage remains -->
- [ ] Normalize return codes & `if __name__ == "__main__"` patterns across scripts
- [x] Expose importable `run()` + helper functions for programmatic use

## 3. Testing (🧪)
- [x] Parity test against canonical implementation (`test_snqi_parity.py`)
- [x] Strategy key & range validation (`test_snqi_strategies.py`)
- [x] Pareto sampling determinism (`test_snqi_pareto_determinism.py`)
- [x] Sensitivity monotonicity (`test_snqi_sensitivity_monotonic.py`)
- [x] CLI optimization & recompute tests
- [x] Schema regression tests (`test_snqi_schema_snapshot.py` / `test_snqi_schema.py`)
- [x] Missing optional metric handling (`test_snqi_missing_optional_metrics.py`)
- [x] Malformed JSONL skip counting (`test_snqi_malformed_skip_cli.py`)
- [x] Normalization comparison correlations (`test_snqi_cli_recompute.py`)
- [x] Drift detection vs canonical weights (`test_snqi_drift_detection.py`)
- [x] Minimal fixture smoke with CI placeholder (`test_snqi_fixture_minimal.py`)
- [x] Bootstrap CI tests (add interval assertions; smoke tests exist)

## 4. Benchmark / CLI Integration (⚙️)
- [x] Unified benchmark subcommand (`robot_sf_bench snqi <action>`) wrapper
- [ ] Config-based invocation wiring weights into evaluation pipeline
- [ ] Automated consumption of optimized weights in eval flow (docs + code)
- [ ] Convenience recompute command for stored results

## 5. Reproducibility & Determinism (🎯)
- [x] Deterministic seeding (NumPy + DE seed)
- [x] Seed captured in metadata
- [x] Deterministic episode sampling (`--sample N` with fixed seed)
- [ ] Document remaining nondeterminism (SciPy stochastic internals / threading)

## 6. Methodology Improvements (🎯)
- [ ] Principled stability metric (bootstrap Spearman) <!-- Planned replacement -->
- [ ] True multi-objective Pareto (e.g., NSGA-II) exploration
- [x] Simplex projection option (flag; default off for interpretability)
- [x] Objective component breakdown (stability vs discriminative)
- [ ] Weighted scenario optimization (episode weighting)
- [x] Differential evolution early stopping (patience/min_delta)
- [x] Real confidence intervals via bootstrap (implemented; placeholder deprecated)

## 7. Performance & Scalability
- [x] Phase timing instrumentation (metadata timings)
- [ ] Small dataset warning (< threshold episodes) for stability reliability
- [x] Episode subset sampling (`--sample` deterministic)

## 8. Dependency & Extras Management
- [ ] Optional extras group (`analysis` / `viz`) for plotting deps
- [ ] SNQI docs explicitly reference `uv add` usage (repo-level docs mostly done)
- [ ] Minimal dependency listing for headless mode

## 9. Output & Schema Hygiene
- [x] `schema_version` in metadata
- [x] Timestamp + git commit hash
- [x] Provenance invocation echo
- [x] Finite numeric validation
- [x] Compact summary block
- [x] Objective component decomposition & rationale fields

## 10. Error Handling & UX
- [x] Skipped / invalid JSONL line counting (with test)
- [x] Explicit baseline metric validation error
- [x] `--fail-on-missing-metric` toggle
- [ ] Graceful message if matplotlib absent when plots requested
- [x] Exit code taxonomy (distinct non-zero codes)

## 11. Style & Consistency
- [x] Type checking clean (no new errors)
- [x] Ruff clean except justified complexity suppression
- [ ] Add module-level `__all__` for public surfaces
- [ ] Naming normalization (`near_misses` vs `w_near`) doc clarity
- [ ] Logging style consolidation (Section 2 overlap)

## 12. Follow-Up (Not blocking)
- [ ] Notebook example using shared module
- [x] Synthetic fixture dataset (`episodes_small.jsonl`, `baseline_stats.json`)
- [x] Canonical weights artifact with provenance (`model/snqi_canonical_weights_v1.json`)
- [x] Drift detection test guarding objective & weight keys
- [ ] Continuous benchmark CI job (small-sample optimization) <!-- test exists; CI job missing -->
- [ ] Real bootstrap stability + CI intervals

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
## Current Focus Recommendation
Immediate value: 1) formal schema doc + normalization rationale, 2) unified CLI integration, 3) bootstrap stability prototype to retire heuristic stability proxy.

---

