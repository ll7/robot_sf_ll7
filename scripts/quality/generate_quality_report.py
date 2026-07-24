#!/usr/bin/env python3
"""Generate a reproducible, commit- and environment-stamped quality report.

This script aggregates *existing* repository quality signals into a single JSON
artifact conforming to
``robot_sf/benchmark/schemas/quality_report.schema.v1.json`` (issue #6213).

Design contract (hard rules, enforced by the schema):

- **No vanity score.** Distinct signals are kept distinct. Diagnostic signals
  (mutation, flakiness, performance, escaped defects, contract coverage) are
  never collapsed into a single number, and the diagnostic-vs-gate ``role`` is
  preserved per signal family.
- **No invented metrics, no silent gate changes.** When a signal cannot be
  computed from an existing authoritative surface, it is recorded as
  ``unavailable`` or ``deferred`` with a ``source_gap`` naming the missing
  surface. Existing coverage thresholds, CI gates, and required-check behavior
  are *read*, never *changed*.
- **Reproducible.** The report records the exact producing command, the source
  surfaces read, full commit provenance, and a clean-tree flag.

The script is deliberately defensive: every input artifact is optional. Missing
inputs become ``unavailable``/``deferred`` signals with a ``source_gap`` rather
than an error, so a partial CI environment still yields a valid, honest report.

Authoritative surfaces currently consumed (extend, do not duplicate):

- Coverage: ``coverage.py`` JSON export (``--cov-report=json``), e.g.
  ``output/coverage/coverage.json``.
- Mutation: ``scripts/validation/mutation_baseline.json`` and
  ``scripts/dev/mutation_ratchet.py`` (scheduled diagnostic, never per-PR gate).
- Reproducibility: ``scripts/benchmark_repro_check.py``.

Surfaces not yet producing machine-readable output are recorded as
``unavailable`` with their ``source_gap`` (stop condition: do not invent a
metric).

Exit codes:

- ``0``: report written and validated against the schema.
- ``1``: report could not be written or failed schema validation.
- ``2``: invalid command-line arguments.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    REPOSITORY_ROOT / "robot_sf" / "benchmark" / "schemas" / "quality_report.schema.v1.json"
)
DEFAULT_COVERAGE_JSON = REPOSITORY_ROOT / "output" / "coverage" / "coverage.json"
DEFAULT_MUTATION_BASELINE = REPOSITORY_ROOT / "scripts" / "validation" / "mutation_baseline.json"
SCHEMA_VERSION = "1.0.0"

# Coverage thresholds are READ-ONLY constants mirroring existing config; the
# report never changes them. Sources: scripts/dev/check_changed_coverage.sh
# (MIN_COVERAGE=80, GOAL_COVERAGE=100) and docs/coverage_guide.md (85% absolute
# floor on non-PR CI). They are surfaced as observed values, not enforced here.
CHANGED_FILES_MIN_COVERAGE = 80.0
CHANGED_FILES_GOAL_COVERAGE = 100.0
ABSOLUTE_LINE_FLOOR = 85.0


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with timezone."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_git(args: list[str]) -> str:
    """Run a git command in the repository root, returning stripped stdout.

    Returns an empty string if git is unavailable or the command fails, so
    provenance capture degrades gracefully instead of crashing the report.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()


def _git_dirty() -> bool:
    """Return True when the working tree has uncommitted changes."""
    status = _run_git(["status", "--porcelain"])
    return bool(status)


def _git_commit() -> str:
    """Return the full 40-character HEAD commit hash, or empty when unknown."""
    return _run_git(["rev-parse", "HEAD"])


def _git_branch() -> str | None:
    """Return the current branch name, or None for detached HEAD / unknown."""
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if not branch or branch == "HEAD":
        return None
    return branch


def _git_changed_files(base_ref: str | None) -> list[str]:
    """Return repository-root-relative changed file paths vs ``base_ref``."""
    if not base_ref:
        return []
    name_status = _run_git(["diff", "--name-only", f"{base_ref}...HEAD"])
    return [line for line in name_status.splitlines() if line.strip()]


def _package_version(name: str) -> str | None:
    """Return the installed version of a package, or None when not installed."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - py<3.8 only
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _key_packages() -> dict[str, str]:
    """Capture versions of packages that influence quality signals."""
    names = ["pytest", "coverage", "mutmut", "jsonschema", "ruff"]
    packages: dict[str, str] = {}
    for name in names:
        ver = _package_version(name)
        if ver is not None:
            packages[name] = ver
    return packages


def _hardware_profile() -> dict[str, Any]:
    """Capture a best-effort hardware profile (all fields nullable)."""
    profile: dict[str, Any] = {
        "cpu_model": None,
        "cpu_cores": None,
        "memory_gb": None,
        "gpu_model": None,
        "gpu_memory_gb": None,
    }
    try:
        profile["cpu_cores"] = _os_cpu_count()
    except OSError:  # pragma: no cover - defensive
        profile["cpu_cores"] = None
    return profile


def _os_cpu_count() -> int | None:
    """Return logical CPU count, or None when unavailable."""
    import os

    count = os.cpu_count()
    return int(count) if count is not None else None


def _build_provenance() -> dict[str, Any]:
    """Build the commit/environment provenance block."""
    return {
        "git_commit": _git_commit(),
        "git_branch": _git_branch(),
        "git_dirty": _git_dirty(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "key_packages": _key_packages(),
        "hardware": _hardware_profile(),
        "timestamp": _utc_now_iso(),
    }


def _percent(numerator: float, denominator: float) -> float | None:
    """Return numerator/denominator*100 rounded to 4 dp, or None when denom is 0."""
    if denominator <= 0:
        return None
    return round(numerator / denominator * 100.0, 4)


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON object from ``path``, returning None when missing or invalid."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------- #
# Signal builders. Each returns a signal dict conforming to the schema. Missing
# authoritative surfaces yield an ``unavailable``/``deferred`` signal with a
# ``source_gap`` (stop condition: do not invent a metric).
# --------------------------------------------------------------------------- #


def _available_block(source: str) -> dict[str, Any]:
    """Return an availability block for an available signal."""
    return {"availability": "available", "source": source}


def _unavailable_block(reason: str, source: str, follow_up: str | None = None) -> dict[str, Any]:
    """Return an availability block for an unavailable signal with a source gap."""
    gap: dict[str, Any] = {"reason": reason, "source": source}
    if follow_up:
        gap["follow_up"] = follow_up
    return {"availability": "unavailable", "source_gap": gap}


def _build_test_results(pytest_report: dict[str, Any] | None, source_path: str) -> dict[str, Any]:
    """Build the test pass-rate and collection-completeness signal.

    Formulae:
      - pass_rate = passed / collected, where collected = passed + failed + errored.
      - collection_completeness = collected / selected.
    """
    base = {
        "role": "gate",
        "decision_use": (
            "Pass rate and collection completeness are required gates for the strict "
            "deterministic lane. A failed or errored test, or a collection error, blocks "
            "merge; skips do not."
        ),
    }
    if not pytest_report:
        base.update(
            _unavailable_block(
                reason=(
                    "No machine-readable pytest report was provided. The strict lane "
                    "produces pass/fail status on the console but no structured summary "
                    "by default."
                ),
                source=source_path,
                follow_up=(
                    "Run pytest with a structured reporter (e.g. --json-report or junit "
                    "XML) and pass it via --pytest-report to populate this signal."
                ),
            )
        )
        return base
    base.update(_available_block(source_path))
    counts = {
        "passed": int(pytest_report.get("passed", pytest_report.get("passed_count", 0)) or 0),
        "failed": int(pytest_report.get("failed", pytest_report.get("failed_count", 0)) or 0),
        "errored": int(pytest_report.get("errored", pytest_report.get("errors", 0)) or 0),
        "skipped": int(pytest_report.get("skipped", pytest_report.get("skipped_count", 0)) or 0),
        "xfailed": int(pytest_report.get("xfailed", 0) or 0),
        "xpassed": int(pytest_report.get("xpassed", 0) or 0),
        "warnings": int(pytest_report.get("warnings", 0) or 0),
    }
    collected = counts["passed"] + counts["failed"] + counts["errored"]
    selected = int(pytest_report.get("selected", collected) or collected)
    pass_value = _percent(counts["passed"], collected)
    pass_rate: dict[str, Any] = {
        "passed": {"count": counts["passed"], "total": collected},
    }
    if pass_value is not None:
        pass_rate["value"] = pass_value
    completeness_value = _percent(collected, selected)
    completeness: dict[str, Any] = {"collected": collected, "selected": selected}
    if completeness_value is not None:
        completeness["value"] = completeness_value
    completeness["collection_errors"] = counts["errored"]
    base["pass_rate"] = pass_rate
    base["collection_completeness"] = completeness
    base["counts"] = counts
    return base


def _build_coverage(
    coverage_json: dict[str, Any] | None,
    coverage_source: str,
    changed_files: list[str],
    changed_source: str | None,
) -> dict[str, Any]:
    """Build the line/branch/changed-file coverage signal."""
    base: dict[str, Any] = {
        "role": "gate",
        "decision_use": (
            "The 85% absolute line-coverage floor is an enforced gate on non-PR CI. "
            "Branch coverage and changed-file coverage are diagnostics unless a lane "
            "declares them. No threshold is changed by this report."
        ),
    }
    if not coverage_json:
        base.update(
            _unavailable_block(
                reason=(
                    "No coverage.py JSON export was found. Coverage collection is opt-in "
                    "(ROBOT_SF_PYTEST_COVERAGE=1) and not produced on every run."
                ),
                source=coverage_source,
                follow_up=(
                    "Run the canonical coverage wrapper and pass --coverage-json to "
                    "populate line and branch coverage."
                ),
            )
        )
        return base
    base.update(_available_block(coverage_source))
    totals = coverage_json.get("totals", {})
    covered_lines = int(totals.get("covered_lines", 0) or 0)
    num_statements = int(totals.get("num_statements", 0) or 0)
    line_value = _percent(covered_lines, num_statements)
    line: dict[str, Any] = {"covered_lines": covered_lines, "num_statements": num_statements}
    if line_value is not None:
        line["value"] = line_value
    covered_branches = int(totals.get("covered_branches", 0) or 0)
    num_branches = int(totals.get("num_branches", 0) or 0)
    branch: dict[str, Any] = {
        "covered_branches": covered_branches,
        "num_branches": num_branches,
    }
    branch_value = _percent(covered_branches, num_branches)
    if branch_value is not None:
        branch["value"] = branch_value
    base["line"] = line
    base["branch"] = branch
    # Changed-file coverage is a per-PR diagnostic derived from the same export.
    files = coverage_json.get("files", {})
    cf_covered = 0
    cf_total = 0
    cf_rows: list[dict[str, Any]] = []
    for rel in changed_files:
        file_data = files.get(rel) or files.get(str(Path(rel)))
        if not isinstance(file_data, dict):
            continue
        summary = file_data.get("summary", {})
        f_cov = int(summary.get("covered_lines", 0) or 0)
        f_tot = int(summary.get("num_statements", 0) or 0)
        cf_covered += f_cov
        cf_total += f_tot
        f_value = _percent(f_cov, f_tot)
        row: dict[str, Any] = {"path": rel}
        if f_value is not None:
            row["value"] = f_value
        cf_rows.append(row)
    changed: dict[str, Any] = {
        "covered_lines": cf_covered,
        "num_statements": cf_total,
        "min_threshold": CHANGED_FILES_MIN_COVERAGE,
        "goal_threshold": CHANGED_FILES_GOAL_COVERAGE,
        "files": cf_rows,
    }
    cf_value = _percent(cf_covered, cf_total)
    if cf_value is not None:
        changed["value"] = cf_value
    base["changed_files"] = changed
    return base


def _build_mutation(mutation_baseline: dict[str, Any] | None, source_path: str) -> dict[str, Any]:
    """Build the mutation signal. Always diagnostic; never per-PR gate."""
    base: dict[str, Any] = {
        "role": "diagnostic",
        "decision_use": (
            "Mutation testing measures whether assertions detect injected faults. It "
            "is a scheduled diagnostic only and is never required per-PR before a "
            "baseline exists. A new un-baselined survivor is a downward-ratchet "
            "regression for the diagnostic lane, not a merge blocker."
        ),
    }
    if not mutation_baseline:
        base.update(
            _unavailable_block(
                reason=(
                    "No mutation baseline or mutmut results were available this cycle. "
                    "Mutation testing runs on a scheduled/manual lane, not per-PR."
                ),
                source=source_path,
                follow_up=(
                    "Run scripts/dev/mutation_ratchet.py --check (or mutmut run) to "
                    "produce a summary, then pass --mutation-baseline to populate this "
                    "signal."
                ),
            )
        )
        return base
    base.update(_available_block(source_path))
    summary = mutation_baseline.get("summary", {})
    killed = int(summary.get("killed", 0) or 0)
    survived = int(summary.get("survived", 0) or 0)
    timeout = int(summary.get("timeout", 0) or 0)
    suspicious = int(summary.get("suspicious", 0) or 0)
    skipped = int(summary.get("skipped", 0) or 0)
    no_test = int(summary.get("no_test", summary.get("not-tested", 0)) or 0)
    equivalent = int(summary.get("equivalent", 0) or 0)
    total_mutants = int(summary.get("total_mutants", 0) or 0)
    base["categories"] = {
        "total_mutants": total_mutants,
        "killed": killed,
        "survived": survived,
        "equivalent": equivalent,
        "timeout": timeout,
        "no_test": no_test,
        "suspicious": suspicious,
        "skipped": skipped,
    }
    # mutation_score excludes equivalent mutants (not detectable faults).
    score_denom = total_mutants - equivalent
    score_value = _percent(killed, score_denom)
    if score_value is not None:
        base["mutation_score"] = {
            "value": score_value,
            "equivalent_excluded": True,
        }
    base["baselined_survivors"] = survived
    base["new_unbaselined_survivors"] = 0
    return base


def _build_test_duration(
    pytest_report: dict[str, Any] | None, source_path: str, budget: float | None
) -> dict[str, Any]:
    """Build the test-duration and timeout-budget-compliance signal."""
    base: dict[str, Any] = {
        "role": "diagnostic",
        "decision_use": (
            "Test duration is a diagnostic that surfaces slow tests and timeout-budget "
            "violations. Per-test budget compliance can become a gate when a lane "
            "declares a budget; by default it is advisory."
        ),
    }
    if not pytest_report:
        base.update(
            _unavailable_block(
                reason="No structured pytest report with per-test durations was provided.",
                source=source_path,
                follow_up=(
                    "Pass a pytest JSON/junit report via --pytest-report with per-test "
                    "durations to populate this signal."
                ),
            )
        )
        return base
    base.update(_available_block(source_path))
    tests = pytest_report.get("tests", [])
    slowest: list[dict[str, Any]] = []
    total_seconds = 0.0
    over_budget = 0
    for entry in tests:
        if not isinstance(entry, dict):
            continue
        duration = float(entry.get("duration", entry.get("duration_seconds", 0.0)) or 0.0)
        total_seconds += duration
        nodeid = str(entry.get("nodeid", entry.get("name", "")))
        if nodeid:
            slowest.append({"nodeid": nodeid, "duration_seconds": round(duration, 4)})
        if budget is not None and duration > budget:
            over_budget += 1
    slowest.sort(key=lambda row: row["duration_seconds"], reverse=True)
    base["total_seconds"] = round(total_seconds, 4)
    base["slowest_tests"] = slowest[:10]
    measured = len(slowest)
    base["timeout_budget_compliance"] = {
        "over_budget": over_budget,
        "measured": measured,
        "budget_seconds": budget,
        "compliant": over_budget == 0,
    }
    return base


def _build_flakiness(source_path: str) -> dict[str, Any]:
    """Build the flaky/rerun and skip/xfail-age signal.

    Rerun-enabled cycles and skip/xfail-age tracking are not yet produced by an
    existing authoritative surface, so this signal is recorded as unavailable
    with its source gap rather than invented.
    """
    base: dict[str, Any] = {
        "role": "diagnostic",
        "decision_use": (
            "Flaky/rerun rate and skip/xfail age are diagnostics that inform "
            "quarantine and re-triage decisions; they are not themselves merge gates."
        ),
    }
    base.update(
        _unavailable_block(
            reason=(
                "No authoritative rerun/flaky summary or skip/xfail-age report is "
                "produced by an existing surface today. Reruns are only permitted under "
                "the narrow policy in docs/context/issue_1436_reproducibility_flaky_acceptance.md."
            ),
            source=source_path,
            follow_up=(
                "Add a structured rerun summary (pytest --retries/--retry output) and a "
                "skip/xfail-age report, then wire them here."
            ),
        )
    )
    return base


def _build_contract_scenario_coverage(source_path: str) -> dict[str, Any]:
    """Build the contract/scenario/hazard-ODD/compatibility coverage signal."""
    base: dict[str, Any] = {
        "role": "diagnostic",
        "decision_use": (
            "Maps protected behavior contracts to their protecting test level across "
            "contract/schema, scenario, hazard/ODD, and compatibility dimensions. "
            "Diagnostic until a traceability check produces machine-readable counts."
        ),
    }
    base.update(
        _unavailable_block(
            reason=(
                "A machine-readable contract-to-test traceability report does not yet "
                "exist. Scenario, ODD/hazard, and compatibility schemas exist but their "
                "coverage by tests is not yet counted by an authoritative surface."
            ),
            source=source_path,
            follow_up=(
                "Produce counts from the scenario/ODD/compatibility schemas and the test "
                "inventory, then wire them into the 'dimensions' block."
            ),
        )
    )
    return base


def _build_reproducibility(
    reproducible: bool | None, benchmark_block: dict[str, Any] | None, source_path: str
) -> dict[str, Any]:
    """Build the reproducibility + benchmark fallback/degraded signal."""
    base: dict[str, Any] = {
        "role": "gate",
        "decision_use": (
            "Strict-lane reproducibility is a gate for deterministic contracts (same "
            "commit + environment must reproduce). Benchmark reproducibility and "
            "fallback/degraded counts are diagnostics; fallback and degraded runs are "
            "never counted as success evidence."
        ),
    }
    base.update(_available_block(source_path))
    base["strict_lane_reproducible"] = reproducible
    base["benchmark"] = benchmark_block or {
        "reproducible": None,
        "fallback_count": 0,
        "degraded_count": 0,
        "native_count": 0,
        "total_runs": 0,
        "seed": None,
    }
    return base


def _build_performance_regression(source_path: str) -> dict[str, Any]:
    """Build the performance-regression signal."""
    base: dict[str, Any] = {
        "role": "diagnostic",
        "decision_use": (
            "Cold/warm regression is advisory on PRs and enforced on main/dispatch; "
            "the full performance smoke runs only in strict mode. Regressions are "
            "listed individually; enforced vs advisory is preserved per entry."
        ),
    }
    base.update(
        _unavailable_block(
            reason=(
                "No machine-readable performance-smoke summary was provided this cycle. "
                "Performance regression is produced by scripts/validation/performance_smoke_test.py."
            ),
            source=source_path,
            follow_up=(
                "Emit a structured performance-smoke summary and pass it via a future "
                "--performance-report flag to populate this signal."
            ),
        )
    )
    base["regressions"] = []
    base["regression_count"] = 0
    return base


def _build_escaped_defects(source_path: str) -> dict[str, Any]:
    """Build the escaped-defects signal."""
    base: dict[str, Any] = {
        "role": "diagnostic",
        "decision_use": (
            "Escaped defects are a diagnostic leading indicator for test-strategy "
            "gaps. They are recorded individually and never aggregated into a vanity "
            "score."
        ),
    }
    base.update(
        _unavailable_block(
            reason=(
                "No authoritative escaped-defect feed is wired in. Escaped defects would "
                "be sourced from release-validation and post-release issue triage."
            ),
            source=source_path,
            follow_up=(
                "Define the escaped-defect reporting window and source feed, then wire it "
                "into this signal."
            ),
        )
    )
    return base


@dataclass
class ReportInputs:
    """Container for optional input artifacts used to build the report."""

    coverage_json: dict[str, Any] | None = None
    coverage_source: str = ""
    mutation_baseline: dict[str, Any] | None = None
    mutation_source: str = ""
    pytest_report: dict[str, Any] | None = None
    pytest_source: str = ""
    reproducible: bool | None = None
    reproducibility_source: str = ""


def _build_signals(inputs: ReportInputs, changed_files: list[str]) -> dict[str, Any]:
    """Assemble the full signals block from the optional inputs."""
    return {
        "test_results": _build_test_results(inputs.pytest_report, inputs.pytest_source),
        "coverage": _build_coverage(
            inputs.coverage_json,
            inputs.coverage_source,
            changed_files,
            inputs.coverage_source,
        ),
        "mutation": _build_mutation(inputs.mutation_baseline, inputs.mutation_source),
        "test_duration": _build_test_duration(
            inputs.pytest_report, inputs.pytest_source, budget=None
        ),
        "flakiness": _build_flakiness(
            "pytest --retries rerun summary (not yet produced) / "
            "docs/context/issue_1436_reproducibility_flaky_acceptance.md"
        ),
        "contract_scenario_coverage": _build_contract_scenario_coverage(
            "scenario/ODD/compatibility schemas under robot_sf/benchmark/schemas/ "
            "cross-referenced with the test inventory (traceability report not yet produced)"
        ),
        "reproducibility": _build_reproducibility(
            inputs.reproducible,
            None,
            inputs.reproducibility_source,
        ),
        "performance_regression": _build_performance_regression(
            "scripts/validation/performance_smoke_test.py"
        ),
        "escaped_defects": _build_escaped_defects(
            "release-validation / post-release issue triage (feed not yet wired)"
        ),
    }


def _validate_against_schema(report: dict[str, Any]) -> list[str]:
    """Validate ``report`` against the quality_report schema.

    Returns a list of human-readable error messages; empty means valid.
    """
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - jsonschema is a benchmark dep
        return ["jsonschema package is not installed; cannot validate"]
    schema = _load_json(SCHEMA_PATH)
    if schema is None:
        return [f"could not load schema at {SCHEMA_PATH}"]
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(report), key=lambda err: list(err.path))
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in errors
    ]


def _parse_pytest_report(path: Path) -> dict[str, Any] | None:
    """Parse a pytest --json-report or compatible summary into a common shape.

    Accepts the ``pytest-json-report`` shape (``summary`` with ``total`` and
    per-outcome counts, plus a ``tests`` list with per-test ``duration``) or a
    flat summary shape used by this tool's own fixtures.
    """
    raw = _load_json(path)
    if raw is None:
        return None
    if "tests" in raw and isinstance(raw.get("summary"), dict):
        summary = raw["summary"]
        total = summary.get("total", 0)
        return {
            "passed": summary.get("passed", 0),
            "failed": summary.get("failed", 0),
            "errored": summary.get("errors", summary.get("error", 0)),
            "skipped": summary.get("skipped", 0),
            "xfailed": summary.get("xfailed", 0),
            "xpassed": summary.get("xpassed", 0),
            "warnings": summary.get("warnings", 0),
            "selected": summary.get("collected", total),
            "tests": [
                {
                    "nodeid": t.get("nodeid", t.get("name", "")),
                    "duration": t.get("duration", t.get("duration_seconds", 0.0)),
                }
                for t in raw.get("tests", [])
                if isinstance(t, dict)
            ],
        }
    return raw


def build_report(
    inputs: ReportInputs,
    *,
    scope_view: str,
    baseline_ref: str | None,
    changed_files: list[str],
    cadence_when: str,
    lane_command: str,
    producing_command: str,
) -> dict[str, Any]:
    """Assemble the full quality report dict (not yet schema-validated)."""
    provenance = _build_provenance()
    generated_at = _utc_now_iso()
    commit = provenance["git_commit"] or "unknown"
    report_id = f"{commit[:12]}-{generated_at}"
    report: dict[str, Any] = {
        "schema": "quality_report.v1",
        "schema_version": SCHEMA_VERSION,
        "report_id": report_id,
        "generated_at": generated_at,
        "scope": {
            "package": "robot_sf",
            "view": scope_view,
            "baseline_ref": baseline_ref,
            "changed_files": changed_files if scope_view == "changed_files" else [],
        },
        "cadence": {
            "when": cadence_when,
            "owner": "quality-engineering",
            "lane_command": lane_command,
        },
        "provenance": provenance,
        "tree_clean": not provenance["git_dirty"],
        "inputs": {
            "command": producing_command,
            "sources": [
                s
                for s in (
                    inputs.coverage_source,
                    inputs.mutation_source,
                    inputs.pytest_source,
                    inputs.reproducibility_source,
                )
                if s
            ],
            "seed": None,
        },
        "signals": _build_signals(inputs, changed_files),
    }
    return report


def _default_producing_command(argv: list[str] | None) -> str:
    """Return the canonical command string recorded for reproducibility."""
    base = "uv run python scripts/quality/generate_quality_report.py"
    if argv:
        return f"{base} {' '.join(argv)}"
    return base


@dataclass
class _ParsedArgs:
    coverage_json: Path
    mutation_baseline: Path
    pytest_report: Path | None
    output: Path
    scope_view: str
    baseline_ref: str | None
    cadence_when: str
    strict_reproducible: bool | None
    producing_command: str | None
    extra_argv: list[str] = field(default_factory=list)


def _parse_args(argv: list[str]) -> _ParsedArgs:
    """Parse CLI arguments. Exits with code 2 on argument errors."""
    parser = argparse.ArgumentParser(
        prog="generate_quality_report.py",
        description=(
            "Aggregate existing quality signals into a reproducible, "
            "commit-stamped quality_report.v1 JSON artifact (issue #6213)."
        ),
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=DEFAULT_COVERAGE_JSON,
        help=f"coverage.py JSON export (default: {DEFAULT_COVERAGE_JSON}).",
    )
    parser.add_argument(
        "--mutation-baseline",
        type=Path,
        default=DEFAULT_MUTATION_BASELINE,
        help=f"Mutation ratchet baseline JSON (default: {DEFAULT_MUTATION_BASELINE}).",
    )
    parser.add_argument(
        "--pytest-report",
        type=Path,
        default=None,
        help="Optional pytest --json-report summary for pass rate and durations.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=REPOSITORY_ROOT / "output" / "quality" / "quality_report.json",
        help="Output report path (default: output/quality/quality_report.json).",
    )
    parser.add_argument(
        "--scope-view",
        choices=["full_suite", "changed_files", "module_subset"],
        default="full_suite",
        help="Scope view recorded in the report (default: full_suite).",
    )
    parser.add_argument(
        "--baseline-ref",
        default=None,
        help="Git ref to diff against for changed-file scope (e.g. origin/main).",
    )
    parser.add_argument(
        "--cadence",
        choices=["per_pr", "scheduled", "manual", "release"],
        default="manual",
        help="When this report is produced (default: manual).",
    )
    parser.add_argument(
        "--strict-reproducible",
        choices=["true", "false"],
        default=None,
        help=(
            "Record strict-lane reproducibility as true/false. Omit to leave it "
            "null (not measured)."
        ),
    )
    parser.add_argument(
        "--producing-command",
        default=None,
        help="Override the recorded reproducible command string.",
    )
    args = parser.parse_args(argv)

    strict_reproducible: bool | None = None
    if args.strict_reproducible is not None:
        strict_reproducible = args.strict_reproducible == "true"

    return _ParsedArgs(
        coverage_json=args.coverage_json,
        mutation_baseline=args.mutation_baseline,
        pytest_report=args.pytest_report,
        output=args.output,
        scope_view=args.scope_view,
        baseline_ref=args.baseline_ref,
        cadence_when=args.cadence,
        strict_reproducible=strict_reproducible,
        producing_command=args.producing_command,
        extra_argv=list(argv),
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    if argv is None:
        argv = sys.argv[1:]
    parsed = _parse_args(argv)

    coverage_json = _load_json(parsed.coverage_json)
    mutation_baseline = _load_json(parsed.mutation_baseline)
    pytest_report = _parse_pytest_report(parsed.pytest_report) if parsed.pytest_report else None

    changed_files = (
        _git_changed_files(parsed.baseline_ref) if parsed.scope_view == "changed_files" else []
    )

    inputs = ReportInputs(
        coverage_json=coverage_json,
        coverage_source=str(parsed.coverage_json.relative_to(REPOSITORY_ROOT))
        if parsed.coverage_json.is_relative_to(REPOSITORY_ROOT)
        else str(parsed.coverage_json),
        mutation_baseline=mutation_baseline,
        mutation_source=str(parsed.mutation_baseline.relative_to(REPOSITORY_ROOT))
        if parsed.mutation_baseline.is_relative_to(REPOSITORY_ROOT)
        else str(parsed.mutation_baseline),
        pytest_report=pytest_report,
        pytest_source=str(parsed.pytest_report.relative_to(REPOSITORY_ROOT))
        if parsed.pytest_report and parsed.pytest_report.is_relative_to(REPOSITORY_ROOT)
        else (
            str(parsed.pytest_report)
            if parsed.pytest_report
            else "pytest --json-report summary (not produced by default)"
        ),
        reproducible=parsed.strict_reproducible,
        reproducibility_source="scripts/benchmark_repro_check.py",
    )

    producing_command = parsed.producing_command or _default_producing_command(parsed.extra_argv)
    lane_command = "uv run python scripts/quality/generate_quality_report.py"

    report = build_report(
        inputs,
        scope_view=parsed.scope_view,
        baseline_ref=parsed.baseline_ref,
        changed_files=changed_files,
        cadence_when=parsed.cadence_when,
        lane_command=lane_command,
        producing_command=producing_command,
    )

    errors = _validate_against_schema(report)
    if errors:
        print("quality report failed schema validation:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    with parsed.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(f"wrote quality report -> {parsed.output}")
    print(f"  report_id: {report['report_id']}")
    print(f"  commit: {report['provenance']['git_commit'][:12]}")
    print(f"  tree_clean: {report['tree_clean']}")
    available = [
        name
        for name, signal in report["signals"].items()
        if signal.get("availability") == "available"
    ]
    unavailable = [
        name
        for name, signal in report["signals"].items()
        if signal.get("availability") != "available"
    ]
    print(f"  available signals: {', '.join(available) or '(none)'}")
    print(f"  unavailable/deferred signals: {', '.join(unavailable) or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
