# Issue #5004 ty Advisory Diagnostic Baseline + Per-Module Downward Ratchet

Issue: <https://github.com/ll7/robot_sf_ll7/issues/5004>

## Goal

Make the ~2.7k pre-existing `ty` advisory type-check findings visible and
monotone-decreasing per module, without turning the full legacy backlog into an
immediate cleanup blocker. This is the type-safety counterpart to the security
baseline ratchet (#3477/#3529) and the TODO-docstring ratchet (#1285).

## Implementation

`scripts/dev/ty_advisory_ratchet.py` re-runs `uvx ty check .` (gitlab-JSON,
advisory `--exit-zero`), aggregates findings per module, and diffs against the
committed baseline at `scripts/validation/ty_advisory_baseline.json`.

Modes:

* `--check` — the downward-ratchet gate. Fails when a clean module (baseline
  general count 0) gains any finding, or when any tracked module's general count
  increases. Decreases pass and emit a "ratchet opportunity" notice.
* `--write-baseline` — refresh the baseline after intentionally reducing findings.
* `--aggregate-only` — print per-module counts without reading/writing a baseline.
* `--ty-output FILE` — parse a pre-rendered ty gitlab-JSON report instead of
  re-running ty (offline / test / no-network mode).
* `--emit-baseline-fixture [--fixture FILE]` — reconstruct a deterministic
  raw-findings fixture from the committed baseline (no live ty run). The fixture
  aggregates to exactly the baseline's per-module counts, so the
  baseline-reproduction test is host-independent. See **Issue #5070** below.

The downward ratchet gates on the **general** bucket. The
**optional-import** category is recorded for visibility but EXCLUDED from the
gate to avoid overlap with sibling issues:

* A finding is the optional-import category when
  `check_name == "unresolved-import"` and its description starts with
  `unresolved-import: Cannot resolve imported module` (a whole-module resolution
  failure such as `GPUtil`, `cairosvg`, `adjustText`, `rvo2`, `ompl`, or vendored
  relative imports).
* First-party member-resolution errors
  (`unresolved-import: Module X has no member Y`) are KEPT in the general bucket
  because they are real type errors, not optional-dependency noise.

Sibling issues that own the excluded categories: #4990 (optional-import guard
inventory), #4995 (guard-spelling standardization), #4988 (benchmark-CLI typed
errors).

## Issue #5070 — host-independent baseline reproduction

The live `ty` per-module counts are **host-dependent**: they depend on each
host's dependency-resolution state (which optional deps are installed/resolved).
The original acceptance test `test_committed_baseline_reproduces_on_clean_checkout`
ran the live `uvx ty check` and gated on `returncode == 0`, so a perfectly clean
branch failed the readiness gate whenever its host resolved differently from the
baseline-generation host (e.g. total `2540 -> 2541` on one host and
`2540 -> 2478` on another, with large per-module swings in *both* directions).

Fix (per the issue's suggested change):

* A **deterministic raw-findings fixture**
  (`scripts/validation/ty_advisory_findings_fixture.json`) is reconstructed from
  the committed baseline via `--emit-baseline-fixture`. It aggregates to exactly
  the baseline's per-module `general`/`optional_import_excluded`/`total` counts.
* The baseline-reproduction test now parses this committed fixture
  (`test_committed_baseline_reproduces_from_fixture`) and asserts `--check` passes
  against it. This is deterministic and host-independent — it holds on every
  clean worktree. A sync guard (`test_committed_baseline_fixture_is_in_sync`)
  fails loudly if the baseline and fixture drift apart.
* The **live scan remains a separate advisory diagnostic**
  (`test_live_ty_advisory_scan`): opt-in via `TY_ADVISORY_LIVE=1`, skipped when
  `uvx`/`ty` is unavailable, and non-gating (it asserts the diagnostic ran and
  emitted a ratchet summary, not that the host matched the baseline). The CI
  workflow (`.github/workflows/ty-advisory-ratchet.yml`) keeps running the live
  `--check` as a non-gating (`continue-on-error`) companion.

Refresh the fixture after a baseline refresh:

```bash
uv run python scripts/dev/ty_advisory_ratchet.py --write-baseline   # after reducing findings
uv run python scripts/dev/ty_advisory_ratchet.py --emit-baseline-fixture
```

## CI wiring

* `.github/workflows/ty-advisory-ratchet.yml` runs `--check` as a non-gating
  (`continue-on-error`) companion on PRs that touch Python/config/baseline
  paths, plus a weekly schedule. Promote to gating by removing
  `continue-on-error` once the ratchet has settled.
* `scripts/dev/ci_driver.sh` typecheck phase surfaces the ratchet result as
  advisory (non-failing), matching the existing `uvx ty check . --exit-zero`
  advisory posture.

## Worked example (acceptance criterion)

`robot_sf/data` was driven to zero general findings and removed from the
baseline via two genuine, non-optional-import type fixes:

* `robot_sf/data/external/ind.py` — narrow `Path | None` -> `Path` after the
  `missing`-sibling guard (the guard already proved non-None).
* `robot_sf/data/external/socnavbench_eth.py` — build the 2-D
  `traversible_shape` as an explicit `tuple[int, int]` (the `ndim == 2`
  validation already guarantees a 2-element shape).

## Boundary

This does NOT fix all 2.5k+ general findings. It only prevents new or increased
per-module debt relative to the tracked baseline, and demonstrates one module
cleared. Baseline updates should happen after intentional cleanup or explicit
maintainer approval. It does not touch benchmark metric semantics, and it
deliberately excludes the optional-import subset owned by #4990/#4995.

## Validation

```bash
uv run pytest tests/dev/test_ty_advisory_ratchet.py -v
uv run python scripts/dev/ty_advisory_ratchet.py --check
uv run python scripts/dev/ty_advisory_ratchet.py --aggregate-only | head
uv run ruff check scripts/dev/ty_advisory_ratchet.py tests/dev/test_ty_advisory_ratchet.py
# Deterministic, host-independent baseline reproduction (#5070):
uv run python scripts/dev/ty_advisory_ratchet.py --emit-baseline-fixture
uv run python scripts/dev/ty_advisory_ratchet.py --check \
  --ty-output scripts/validation/ty_advisory_findings_fixture.json
# Opt-in live advisory diagnostic (#5070):
TY_ADVISORY_LIVE=1 uv run pytest tests/dev/test_ty_advisory_ratchet.py::test_live_ty_advisory_scan -v
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```
