# Contributor QA Runbook and Test Taxonomy

[Back to Documentation Index](./README.md)

This document is the canonical contributor-facing Quality Assurance (QA) runbook and test taxonomy for `robot_sf_ll7`. It provides a single reference for selecting validation lanes, interpreting test failures, understanding evidence boundaries, and applying explicit Continuous Integration (CI) rerun rules without needing private context or relying on implicit rerun practices.

---

## Plain-Language Summary

Quality Assurance (QA) in `robot_sf_ll7` ensures that code changes preserve simulation stability, metric validity, and reproducible research progress across all Autonomous Mobility Vehicle (AMV) and Vulnerable Road User (VRU) navigation scenarios. This runbook categorizes test types, provides a command matrix for local and CI validation lanes, defines failure classification rules, and establishes clear rerun boundaries.

---

## Maintainer Value Hierarchy & Governance

This QA runbook operates under the maintainer governance rules defined in:
- **[Maintainer Values And Hard Contracts](./maintainer_values.md)** — Proof must be proportional to risk. Substantive or paper-facing claims require reproducible executable evidence. Fallback or degraded execution is never success evidence.
- **[CI Reproducibility & Flaky Acceptance Policy](./context/issue_1436_reproducibility_flaky_acceptance.md)** — Canonical failure-classification criteria and explicit CI rerun boundaries. Its reproducibility-job mapping is reconciled below against the live workflow.
- **[Benchmark Fallback Policy](./context/issue_691_benchmark_fallback_policy.md)** — Fail-closed evaluation rules for `fallback`, `degraded`, and `not_available` execution modes.
- **[Coverage Guide](./coverage_guide.md)** — Code coverage collection, baseline comparison, and reporting rules.
- **[Code Review Guidelines](./code_review.md)** — Benchmark-facing review criteria, provenance checks, and regression traps.

---

### Reproducibility Job Policy Reconciliation

The live [`reproducibility-check` job](../.github/workflows/ci.yml) is the source of truth for its
trigger and gating behavior: it runs on `pull_request` and `workflow_dispatch`, and it is not marked
`continue-on-error`. The linked Issue #1436 policy note has been reconciled with the live workflow
(see Issue #6249): it no longer describes an older `workflow_dispatch`-only or `continue-on-error`
mapping. This runbook uses the live behavior while retaining Issue #1436 for failure classification,
rerun boundaries, and diagnostic-only evidence status.

---

## Test Taxonomy

The repository classifies tests into 12 distinct categories. Understanding these categories ensures that new tests are placed in the appropriate directory and evaluated against the correct evidence standards.

| Category | Purpose | Typical Path / Location | Execution Standard & Gate |
| --- | --- | --- | --- |
| **Unit** | Test isolated module functions, dataclasses, or math primitives in fast execution loops. | `tests/` | Deterministic contract. Runs in fast feedback (`pytest -m "not slow"`). |
| **Contract / Schema** | Verify Pydantic/dataclass schemas, API signatures, JSON/YAML layout, and config structures. | `tests/` | Deterministic contract. Must pass before PR merge. |
| **Integration** | Test interaction between multiple components (e.g., gym environment + SocialForce backend + telemetry). | `tests/` | Deterministic contract. The `slow` marker is assigned at collection time from test paths and allowlists; elapsed-time budgets are reported separately. |
| **Subprocess / CLI** | Test command-line interfaces, script entry points, and process launch wrappers via `subprocess`. | `tests/` | Deterministic contract. Verifies CLI flags, exit codes, and output streams. |
| **Scenario** | Exercise specific robot-pedestrian interaction configurations and map layouts. | `tests/`, `maps/` | Deterministic or seed-controlled. Verifies collision detection and path tracking. |
| **Benchmark** | Measure planner performance, safety metrics, and Social Navigation Quality Index (SNQI) across standard splits. | `robot_sf/benchmark/`, `configs/benchmarks/` | Executable proof required. Requires seed matrices and fail-closed status verification. |
| **Smoke** | Fast sanity checks verifying maps, environment instantiation, cold/warm runs, and artifact creation. | `scripts/validation/` | Runs in CI `smoke-artifacts` lane. Enforces canonical `output/` directory structure. |
| **Visual** | GUI playback, Pygame rendering, telemetry overlay, and video generation. | `tests/pygame/` | Headless execution required (`DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy`). |
| **Performance** | Measure step-throughput, JIT compilation latency, and execution timing against soft/hard budgets. | `scripts/validation/performance_smoke_test.py` | Advisory on PRs; strictly enforced on `main` branch pushes. |
| **Compatibility** | Validate backwards compatibility for model checkpoints, configuration schemas, and dataset intake. | `tests/` | Deterministic contract. Prevents schema drift across releases. |
| **Reproducibility** | Verify that identical seeds yield bitwise or statistical equivalence across environment instances. | `scripts/benchmark_repro_check.py` | Diagnostic/campaign evidence. The CI job runs for pull requests and `workflow_dispatch`; contributors can also run the targeted local check. See the policy reconciliation above. |
| **Acceptance** | End-to-end contributor workflow or feature verification using structured scenario specifications. | `tests/` | High-level user story verification. Follows the selective `pytest-bdd` policy when specified. |

---

## Selective `pytest-bdd` Policy

`robot_sf_ll7` enforces a strict policy on Behavior-Driven Development (BDD) frameworks:
- **Acceptance Workflows Only**: `pytest-bdd` is permitted **only** for high-level, human-facing acceptance test suites where business logic or feature specifications need Gherkin (`Given/When/Then`) feature files.
- **Standard `pytest` Default**: All technical, unit, integration, schema, performance, and scenario tests **must** remain standard `pytest` test functions. Do not introduce BDD feature files for internal code modules or developer-facing unit contracts.

---

## Command Matrix & Validation Lanes

Contributors must run validation commands matching their change class. The repository provides three canonical CI lanes plus targeted local helpers.

```
+-----------------------------------------------------------------------------------+
|                                 VALIDATION LANES                                  |
+------------------------------------+----------------------------------------------+
| 1. Strict Lane (Fast Feedback)     | Ruff lint/format + unit tests (pytest)       |
| 2. Smoke Lane (Integration/Output) | Map verification + smoke + artifact policy   |
| 3. Benchmark Lane (Campaign Proof) | Multi-seed campaigns + reproducibility check |
+------------------------------------+----------------------------------------------+
```

### Command Reference Table

| Validation Lane | Canonical Command | Target / Scope | Primary Use Case & Evidence Level |
| --- | --- | --- | --- |
| **Strict Lane** | `scripts/dev/ci_driver.sh lint typecheck test` | Ruff check/format, advisory typecheck, unit tests (`pytest -n auto`). | Fast PR feedback. Deterministic contract gate for all PRs. |
| **Fast Unit Pass** | `uv run pytest -m "not slow" tests` | Excludes slow/integration test directories auto-marked in `tests/conftest.py`. | Rapid local iteration during active coding. |
| **Smoke Lane** | `scripts/dev/ci_driver.sh smoke artifact-policy` | Map verification, environment smoke, telemetry check, artifact root enforcement. | Validates runtime integration and canonical `output/` layout. |
| **Headless GUI** | `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/pygame` | Pygame GUI rendering and visual overlay tests. | Verifies visual/display components in headless environments. |
| **PR Readiness Gate** | `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` | Ruff, unit tests, coverage delta gate, docstring ratchet, changed paths check. | Full local PR readiness check before pushing feature branches. |
| **Compact Validation** | `uv run python scripts/dev/run_compact_validation.py -- <command>` | Wraps any validation command with compact summary output. | Efficient local validation without verbose log spam. |
| **Coverage Analysis** | `uv run pytest --cov=robot_sf tests` | Measures statement and branch coverage in `robot_sf/`. | Diagnostic coverage collection (see [Coverage Guide](./coverage_guide.md)). |
| **Coverage Comparison** | `uv run python scripts/coverage/compare_coverage.py --current output/coverage/coverage.json --baseline output/coverage/.coverage-baseline.json --format terminal` | Compares current coverage against saved baseline. | Prevents silent coverage regressions. |
| **Reproducibility Diagnostic** | `scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/benchmark_repro_check.py` | Diagnostic reproducibility check across canonical seeds and metrics. | Validates same-seed metric equivalence (pull request, `workflow_dispatch`, or local); see the policy reconciliation above. |
| **Benchmark Campaign** | `uv run python scripts/tools/run_benchmark_release.py --manifest <manifest-path>` | Manifest-driven benchmark release campaign runner. | Nominal or paper-grade benchmark evidence. |
| **Documentation Integrity** | `uv run python scripts/dev/check_docs_evidence_integrity.py --files <file-list>` | Checks markdown links, evidence references, and cited command paths. | Documentation-only and instruction-only PR validation. |
| **Mutation Diagnostics** | `uv run python scripts/dev/mutation_ratchet.py --check` | Weekly or manually dispatched bounded mutation-strength ratchet; see [mutation-testing triage](../mutation_testing_triage.md). | Diagnostic probe for test quality (not a required PR gate). |

---

## Failure Classification & Rerun Boundaries

To preserve benchmark credibility and prevent wasted CI compute, test failures must be triaged according to their root cause. **Rerunning CI jobs is strictly governed by failure class.**

| Failure Class | Root Cause Examples | Rerun Policy | Required Action |
| --- | --- | --- | --- |
| **Deterministic Contract Failure** | Ruff lint violation, syntax error, import error, assertion failure in deterministic logic, schema mismatch, artifact root policy breach. | **NEVER PERMITTED** | Fix the underlying code or test contract. Pushing a fix is required. |
| **Environment-Class Flake** | Runner network/apt timeout, headless display initialization race (`SDL_VIDEODRIVER`), GitHub Actions cache restore miss. | **SINGLE RERUN PERMITTED** | Rerun the failing job once. If failure persists, open an infrastructure debt issue. |
| **Stochastic / Statistical Variance** | Bootstrap CI width variation, seed-draw fluctuation, minor aggregate rank shifts within confidence intervals. | **NEVER PERMITTED** | Inspect sample size and seed count. Do not rerun solely to seek a favorable draw. |
| **Fallback / Degraded Execution** | Planner fell back to default velocity vector or ran in degraded mode due to missing dependency or solver failure. | **NEVER PERMITTED** | Treat as non-success. Mark as `fallback` or `degraded` per [Issue #691](./context/issue_691_benchmark_fallback_policy.md). |
| **Timeout / Performance Breach** | Budget breach in a cold/warm performance test or a test timeout near an execution threshold. | **ONLY AFTER ENVIRONMENT-CLASS DIAGNOSIS** | A threshold breach alone never permits a rerun. The cold/warm smoke is advisory on PRs, but strict-mode regressions, deterministic failures, and hard timeouts require investigation and a fix; rerun once only when evidence identifies an environment-class cause. |
| **Quarantined Test** | A temporarily excluded test with a known blocker that cannot safely contribute to a required lane. | **NOT A PASS OR A RERUN BASIS** | Keep the exclusion visible with a linked tracking issue, reason, and re-enable condition. Record the resulting coverage gap; do not present the quarantined test or lane as green success evidence. |
| **Skipped Test** | A test whose prerequisite or platform condition is unavailable, marked with `@pytest.mark.skip` or `@pytest.mark.skipif`. | **STANDARD EXECUTION** | Include a precise reason and linked GitHub issue when the condition is temporary or repository-owned; restore normal execution when the prerequisite is available. |
| **Expected Failure (`xfail`)** | A test that exercises a known failing behavior, marked with `@pytest.mark.xfail`. | **STANDARD EXECUTION** | Include a clear reason and linked GitHub issue, and investigate an unexpected pass rather than treating it as an unreviewed exemption. |

---

## Evidence Boundaries & Caveats

When presenting test results, contributors and AI agents must adhere to the following claim boundaries:

1. **Proof Proportional to Risk**: Low-risk docs edits require diff inspection and link verification. Runtime code changes require focused test execution. Benchmark or paper-facing claims require full reproducible evidence.
2. **No Overclaiming Coverage**: Coverage percentages reported by `pytest-cov` are diagnostic metrics. High coverage does not guarantee test thoroughness, and low coverage in non-critical modules does not block merges unless a baseline regression occurs.
3. **Fail-Closed Benchmark Policy**: Fallback or degraded execution is never success evidence. If a planner runs in fallback mode during a benchmark evaluation, the run status must be recorded as `fallback` or `degraded` and excluded from success claims.
4. **Diagnostic vs Benchmark Evidence**: Reproducibility diagnostics (`scripts/benchmark_repro_check.py`) provide smoke-level sanity checks; they do not replace multi-seed camera-ready benchmark campaigns.

---

## Related Documentation

- **[Maintainer Values And Hard Contracts](./maintainer_values.md)** — Core project values and claim evidence hierarchy.
- **[Development Guide](./dev_guide.md)** — Primary developer onboarding, setup, and unified test suite commands.
- **[Coverage Guide](./coverage_guide.md)** — Detailed guide to coverage collection, HTML reports, and baseline comparison.
- **[Code Review Guidelines](./code_review.md)** — Review standards for PRs, benchmark code, and test verification.
- **[Glossary](./glossary.md)** — Canonical definitions of repository terms (VRU, AMV, SNQI, ODD, etc.).
- **[CI Reproducibility & Flaky Policy](./context/issue_1436_reproducibility_flaky_acceptance.md)** — Detailed CI lane mapping and flaky failure triage rules.
- **[Benchmark Fallback Policy](./context/issue_691_benchmark_fallback_policy.md)** — Fail-closed requirements for benchmark evaluation.
