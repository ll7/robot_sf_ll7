# Issue #1436 CI Reproducibility and Flaky Statistical Acceptance Policy (2026-05-22)

This note codifies how Robot SF treats reproducibility in CI and separates
deterministic contract failures from stochastic or statistical acceptance
boundaries. It also defines the narrow conditions under which a CI rerun is
permitted.

## Canonical Validation Lanes

The repository uses three validation lanes that map to current CI jobs and local
scripts. All lane definitions are derived from `.github/workflows/ci.yml` and
`scripts/dev/ci_driver.sh` as they exist today.

### 1. Strict Lane (Deterministic Contracts)

**CI job**: `fast-feedback`  
**Local equivalent**: `scripts/dev/ci_driver.sh lint typecheck test`

- `lint` — Ruff lint and format check; any finding is a deterministic contract
  failure.
- `typecheck` — Advisory `ty` run (`--exit-zero`); findings are reported but do
  not fail the phase. A future fail-closed typecheck gate must be proposed
  separately before it becomes a merge requirement.
- `test` — Main pytest suite (`scripts/dev/run_tests_parallel.sh` via `pytest -n
  auto`). Test failures here are treated as deterministic unless they are already
  classified as flaky under the triage rules below.

**Expectation**: Green on every run for the same commit and environment. No
statistical variance is expected.

### 2. Smoke Lane (Integration and Regression Guards)

**CI job**: `smoke-artifacts`  
**Local equivalent**: `scripts/dev/ci_driver.sh smoke artifact-policy`

- `smoke` — Map verification, environment validation, model prediction,
  cold/warm performance regression, telemetry tracker smoke, and a minimal
  benchmark episode run (`robot_sf_bench run` with a CI-scoped matrix).
- `artifact-policy` — Enforces the canonical `output/` artifact root.

**Behavior notes from current CI**:
- Cold/warm regression is **advisory** on pull requests and **enforced** on
  `main` and `workflow_dispatch`.
- Full performance smoke (`scripts/validation/performance_smoke_test.py`) runs
  only in strict mode (`main` or `workflow_dispatch`).
- Reproducibility check (`scripts/benchmark_repro_check.py`) is triggered only
  on `workflow_dispatch`.

### 3. Benchmark-Facing Lane (Campaign Evidence)

**CI job**: `workflow_dispatch` reproducibility check (optional)  
**Local equivalent**: Config-first benchmark campaigns using
`scripts/tools/run_benchmark_release.py` or `robot_sf_bench run` with committed
matrix YAML.

- This lane is explicitly **not** part of the per-PR required check set.
- It is the domain of bootstrap confidence intervals, multi-seed aggregation,
  SNQI ranking, and seed-variability analysis.
- See [Issue #595 Seed-Variability Contract](issue_595_seed_variability_contract.md)
  and [Issue #832 Paper-Matrix Extended Seed Schedule](issue_832_paper_matrix_extended_seed_schedule.md)
  for the current frozen camera-ready artifact contract.

## Deterministic vs Stochastic/Statistical Failures

| Category | Examples | Acceptance Rule |
|----------|----------|---------------|
| **Deterministic contract failures** | Syntax error, import failure, schema validation error, assertion failure in deterministic code, Ruff lint violation, artifact-root policy breach | Must be fixed before merge. Reruns are not a remediation. |
| **Stochastic / statistical variance** | Bootstrap CI width, seed-to-seed success-rate spread, cold/warm timing jitter near threshold, benchmark aggregate rank shifts within confidence intervals | Documented and expected. Do not treat a single unfavorable draw as a code defect, and do not rerun only to obtain a more favorable draw. |
| **Environment-class flakiness** | Transient apt-get failure, cache download timeout, display initialization race in headless CI, network timeout fetching W&B artifact | Rerun permitted once; if it persists, treat as infrastructure debt. |

**Boundary**: A test that fails with the same stack trace on the same commit in
repeated runs is deterministic. A test that passes locally but fails in CI with
a display or timing-related stack trace is a candidate for environment-class
flakiness, not a statistical failure.

## Flaky Failure Triage

When a CI run fails, apply this order:

1. **Read the phase and stack trace.**
   - Lint/typecheck/test failure in `fast-feedback` → deterministic.
   - Smoke failure with a display or subprocess timeout → environment-class.
   - Benchmark aggregate below expectation → stochastic/statistical.

2. **Check for known flaky signatures.**
   - Headless display races (`SDL_VIDEODRIVER=dummy` on Ubuntu runners).
   - First-time JIT/compilation latency in `fast-pysf` causing timeout near
     the hard performance budget.
   - Cache restore miss after a `pyproject.toml` or `uv.lock` change.

3. **Classify and act.**
   - **Deterministic** → open a fix PR; do not rerun the failing job in hope of
     a different outcome.
   - **Environment-class** → one rerun is allowed; if the rerun also fails,
     investigate the infrastructure change (runner image, apt mirror, cache
     backend).
   - **Stochastic/statistical** → inspect the sample size and variance; do not
     rerun solely because the draw was unfavorable.

## Explicit Rerun Rule

A rerun of any CI job is permitted **only** when the failure is classified as
environment-class. Reruns are **not** permitted for:

- **Benchmark contract failures** — schema errors, missing required metadata,
  unsupported runtime contracts, or any `availability_status` other than
  `available` as defined in
  [Issue #691 Benchmark Fallback Policy](issue_691_benchmark_fallback_policy.md).
- **Fallback or degraded execution** — any planner running in `fallback` or
  `degraded` mode; see the canonical fail-closed rule in
  [Issue #691 Benchmark Fallback Policy](issue_691_benchmark_fallback_policy.md).
- **Unfavorable statistical outcomes** — a bootstrap CI that does not overlap
  with a prior claim, a seed draw that produces an outlier aggregate, low
  success rate, high collision rate, poor SNQI, or a cold/warm measurement that
  breaches a threshold without environment-class evidence.

**Rationale**: Rerunning to obtain a more favorable statistical draw erodes
benchmark credibility. Rerunning to recover from a transient runner failure is
standard CI hygiene.

## Relation to Existing Policies

- **Benchmark fallback policy**: [Issue #691 Benchmark Fallback Policy](issue_691_benchmark_fallback_policy.md)
  defines `fallback` and `degraded` as non-success. This note adds that reruns
  must not be used to convert a non-success into a success.
- **Test significance verification**: `docs/dev_guide.md` already instructs
  contributors to challenge flaky or low-value tests before investing fix effort.
  This note adds the CI-side rerun boundary.
- **Performance breach handling**: `docs/dev_guide.md` documents soft vs hard
  tiers. This note clarifies that a soft breach is advisory and does not justify
  a rerun.

## Validation Commands (Current)

Local equivalents of the three lanes:

```bash
# Strict lane
scripts/dev/ci_driver.sh lint typecheck test

# Smoke lane
scripts/dev/ci_driver.sh smoke artifact-policy

# Full local CI equivalent
scripts/dev/run_ci_local.sh

# PR readiness (includes changed-files coverage gate and docstring ratchet)
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

CI workflow: `.github/workflows/ci.yml`.

## Scope and Limitations

- This note does **not** introduce new validation commands or change existing
  thresholds. It documents the rerun boundary that was previously implicit.
- It does **not** assert that any current test is flaky; it provides the
  classification framework for when a flaky symptom appears.
- It does **not** modify benchmark semantics, metrics, formulas, or configs.
