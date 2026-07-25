#!/usr/bin/env python3
"""Generate a reproducible, commit/environment-stamped quality report (issue #6213).

The report deliberately distinguishes independent quality signals instead of
collapsing them into a single vanity score.  Every signal is read ONLY from an
existing authoritative surface; a signal that cannot be computed from an existing
surface is recorded as ``unavailable`` or ``deferred`` together with the source gap
that blocks it -- never invented.

The commit/environment stamp reuses ``scripts/dev/pr_ready_freshness.py`` so the
git/stamp logic is shared rather than re-derived here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_ready_freshness as freshness

SCHEMA_VERSION = "quality_report.v1"

# Mirror of scripts/tools/campaign_result_store.py::ROW_STATUS_VALUES (the canonical
# campaign row-status enum).  Kept as a local constant so this read-only report tool
# does not import the pandas/yaml-heavy result-store module just to read the enum.
ROW_STATUS_VALUES = (
    "native",
    "adapter",
    "diagnostic_only",
    "fallback",
    "degraded",
    "unavailable",
    "failed",
)


def _read_json(path: Path) -> Any | None:
    """Read a JSON artifact, degrading to ``None`` when missing or unreadable.

    Returns:
        Parsed JSON value, or ``None`` when the file is absent or cannot be parsed so
        collectors can degrade to ``unavailable`` instead of crashing.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _totals(repo_root: Path) -> dict[str, Any] | None:
    """Return the ``totals`` block from ``output/coverage/coverage.json``.

    Returns:
        Coverage ``totals`` mapping, or ``None`` when the artifact or block is absent.
    """
    data = _read_json(repo_root / "output/coverage/coverage.json")
    if isinstance(data, dict) and isinstance(data.get("totals"), dict):
        return data["totals"]
    return None


def _collect_coverage_line(repo_root: Path) -> dict[str, Any]:
    """Collect overall line coverage from the coverage.py JSON artifact.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    source = "scripts/coverage/compare_coverage.py"
    totals = _totals(repo_root)
    if totals is None or "percent_covered" not in totals:
        return {
            "status": "unavailable",
            "value": None,
            "unit": "percent",
            "source": source,
            "source_gap": (
                "output/coverage/coverage.json absent or has no totals.percent_covered; "
                "generate with pytest --cov-report=json then scripts/coverage/compare_coverage.py"
            ),
            "categories": None,
        }
    return {
        "status": "available",
        "value": totals.get("percent_covered"),
        "unit": "percent",
        "source": source,
        "source_gap": None,
        "categories": {
            "covered_lines": totals.get("covered_lines"),
            "missing_lines": totals.get("missing_lines"),
        },
    }


def _collect_coverage_branch(repo_root: Path) -> dict[str, Any]:
    """Collect overall branch coverage from the coverage.py JSON artifact.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    source = "scripts/dev/branch_coverage_report.py"
    totals = _totals(repo_root)
    if totals is None:
        return {
            "status": "unavailable",
            "value": None,
            "unit": "percent",
            "source": source,
            "source_gap": (
                "output/coverage/coverage.json absent; generate with "
                "pytest --cov-branch --cov-report=json (see scripts/dev/branch_coverage_report.py)"
            ),
            "categories": None,
        }
    covered = totals.get("covered_branches", 0)
    missing = totals.get("missing_branches", 0)
    total = covered + missing
    pct = (covered / total * 100) if total > 0 else 100.0
    return {
        "status": "available",
        "value": pct,
        "unit": "percent",
        "source": source,
        "source_gap": None,
        "categories": {"covered_branches": covered, "missing_branches": missing},
    }


def _collect_coverage_changed_file(repo_root: Path) -> dict[str, Any]:
    """Collect changed-file coverage status.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    source = "scripts/coverage/check_changed_files_coverage.py"
    return {
        "status": "unavailable",
        "value": None,
        "unit": "percent",
        "source": source,
        "source_gap": (
            "no persisted changed-file coverage artifact: check_changed_files_coverage.py "
            "computes per-changed-file coverage live from git diff + output/coverage/coverage.json "
            "(min 80 / goal 100) and emits a pass/fail verdict, not a stored metric"
        ),
        "categories": None,
    }


def _collect_test_pass_rate(repo_root: Path) -> dict[str, Any]:
    """Collect test pass-rate status.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    return {
        "status": "deferred",
        "value": None,
        "unit": "percent",
        "source": None,
        "source_gap": "no junitxml/pass-rate artifact owner; see epic #6205 child (issues #5753/#5108)",
        "categories": None,
    }


def _collect_collection_completeness(repo_root: Path) -> dict[str, Any]:
    """Collect test-collection completeness status.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    return {
        "status": "deferred",
        "value": None,
        "unit": None,
        "source": None,
        "source_gap": "no collected-test inventory owner (issues #5753/#5108)",
        "categories": None,
    }


def _collect_mutation(repo_root: Path) -> dict[str, Any]:
    """Collect mutation-testing summary categories from the ratchet baseline.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    source = "scripts/dev/mutation_ratchet.py --check"
    data = _read_json(repo_root / "scripts/validation/mutation_baseline.json")
    summary = data.get("summary") if isinstance(data, dict) else None
    if not isinstance(summary, dict):
        return {
            "status": "unavailable",
            "value": None,
            "unit": "mutants",
            "source": source,
            "source_gap": (
                "scripts/validation/mutation_baseline.json absent; generate with "
                "scripts/dev/mutation_ratchet.py --write-baseline"
            ),
            "categories": None,
        }
    categories = {
        "killed": summary.get("killed"),
        "survived": summary.get("survived"),
        "skipped": summary.get("skipped"),
        "suspicious": summary.get("suspicious"),
        "timeout": summary.get("timeout"),
        "total_mutants": summary.get("total_mutants"),
    }
    return {
        "status": "available",
        "value": summary.get("total_mutants"),
        "unit": "mutants",
        "source": source,
        "source_gap": (
            "equivalent/no_test classifications live in mutation_testing_triage.md / mutmut "
            "stats and are not part of this baseline; categories above stay separately visible"
        ),
        "categories": categories,
    }


def _collect_test_duration(repo_root: Path) -> dict[str, Any]:
    """Collect test-duration status.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    return {
        "status": "unavailable",
        "value": None,
        "unit": "seconds",
        "source": "scripts/dev/ci_timing_summary.py",
        "source_gap": (
            "ci_timing_summary.py parses CI step timing from `gh run view --log`; no local "
            "timing artifact is persisted to read, so the value is unavailable here"
        ),
        "categories": None,
    }


def _collect_timeout_budget_compliance(repo_root: Path) -> dict[str, Any]:
    """Collect timeout-budget compliance status.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    return {
        "status": "deferred",
        "value": None,
        "unit": None,
        "source": "scripts/dev/ci_timing_summary.py",
        "source_gap": "durations exist (ci_timing_summary.py) but no declared timeout-budget gate owner",
        "categories": None,
    }


def _collect_flaky_rerun_rate(repo_root: Path) -> dict[str, Any]:
    """Collect flaky-rerun rate status.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    return {
        "status": "unavailable",
        "value": None,
        "unit": "percent",
        "source": None,
        "source_gap": "no test-flakiness registry/quarantine owner (epic #6205 child 4)",
        "categories": None,
    }


def _collect_skip_xfail_age(repo_root: Path) -> dict[str, Any]:
    """Collect skip/xfail age status.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    return {
        "status": "unavailable",
        "value": None,
        "unit": "days",
        "source": None,
        "source_gap": "no skip/xfail-age surface (no xfail registry under scripts/)",
        "categories": None,
    }


def _collect_hazard_odd_coverage(repo_root: Path) -> dict[str, Any]:
    """Collect hazard/ODD coverage from a persisted rollup artifact when present.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    source = "scripts/tools/hazard_odd_coverage_rollup.py"
    data = _read_json(repo_root / "output/benchmarks/hazard_odd_coverage_rollup.json")
    if not isinstance(data, dict):
        return {
            "status": "unavailable",
            "value": None,
            "unit": None,
            "source": source,
            "source_gap": (
                "no persisted hazard/ODD coverage rollup artifact "
                "(output/benchmarks/hazard_odd_coverage_rollup.json); the rollup CLI emits to "
                "stdout and has no persisted-artifact owner yet"
            ),
            "categories": None,
        }
    return {
        "status": "available",
        "value": None,
        "unit": None,
        "source": source,
        "source_gap": None,
        "categories": data,
    }


def _collect_scenario_certification(repo_root: Path) -> dict[str, Any]:
    """Collect scenario-certification status from a persisted certificate artifact.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    source = "scripts/tools/certify_scenarios.py"
    data = _read_json(repo_root / "output/benchmarks/scenario_certification.json")
    if not isinstance(data, dict):
        return {
            "status": "unavailable",
            "value": None,
            "unit": None,
            "source": source,
            "source_gap": (
                "no persisted scenario-certification artifact "
                "(output/benchmarks/scenario_certification.json); certify_scenarios.py emits "
                "scenario_cert.v1 certificates to stdout (robot_sf/scenario_certification/)"
            ),
            "categories": None,
        }
    return {
        "status": "available",
        "value": None,
        "unit": None,
        "source": source,
        "source_gap": None,
        "categories": data,
    }


def _collect_contract_coverage(repo_root: Path) -> dict[str, Any]:
    """Collect contract-coverage status.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    return {
        "status": "deferred",
        "value": None,
        "unit": None,
        "source": None,
        "source_gap": "contract-coverage tests exist but no metric rollup owner",
        "categories": None,
    }


def _collect_compatibility_coverage(repo_root: Path) -> dict[str, Any]:
    """Collect compatibility-coverage status.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    return {
        "status": "deferred",
        "value": None,
        "unit": None,
        "source": None,
        "source_gap": "compat manifests exist but no single compat-coverage rollup owner",
        "categories": None,
    }


def _collect_reproducibility_status(repo_root: Path) -> dict[str, Any]:
    """Collect benchmark reproducibility status from the repro-check artifact.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    source = "scripts/benchmark_repro_check.py"
    data = _read_json(repo_root / "output/benchmarks/reproducibility_check.json")
    if not isinstance(data, dict):
        return {
            "status": "unavailable",
            "value": None,
            "unit": None,
            "source": source,
            "source_gap": (
                "output/benchmarks/reproducibility_check.json absent; generate with "
                "scripts/benchmark_repro_check.py"
            ),
            "categories": None,
        }
    return {
        "status": "available",
        "value": data.get("reproducible"),
        "unit": None,
        "source": source,
        "source_gap": None,
        "categories": {"status": data.get("status"), "reproducible": data.get("reproducible")},
    }


def _collect_fallback_degraded_counts(repo_root: Path) -> dict[str, Any]:
    """Aggregate campaign result-store row-status counts across stored campaigns.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    source = "scripts/tools/campaign_result_store.py + scripts/tools/generate_why_first_report.py"
    categories = dict.fromkeys(ROW_STATUS_VALUES, 0)
    stores_found = 0
    total_episodes = 0
    for summary_path in sorted(repo_root.glob("output/**/summary.json")):
        data = _read_json(summary_path)
        if not isinstance(data, dict):
            continue
        if data.get("schema_version") != "campaign-result-store.v1":
            continue
        counts = data.get("row_status_counts")
        if not isinstance(counts, dict):
            continue
        stores_found += 1
        for key, count in counts.items():
            if key in categories and isinstance(count, int):
                categories[key] += count
                total_episodes += count
    if stores_found == 0:
        return {
            "status": "unavailable",
            "value": None,
            "unit": "episodes",
            "source": source,
            "source_gap": (
                "no campaign result-store summary.json (schema campaign-result-store.v1) found "
                "under output/; generate via scripts/tools/campaign_result_store.py"
            ),
            "categories": None,
        }
    return {
        "status": "available",
        "value": total_episodes,
        "unit": "episodes",
        "source": source,
        "source_gap": None,
        "categories": categories,
    }


def _collect_performance_regression(repo_root: Path) -> dict[str, Any]:
    """Collect performance-regression status from the perf-trend latest artifact.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    source = "python -m robot_sf.benchmark.perf_trend"
    data = _read_json(repo_root / "output/benchmarks/perf/trend/latest.json")
    if not isinstance(data, dict):
        return {
            "status": "unavailable",
            "value": None,
            "unit": None,
            "source": source,
            "source_gap": (
                "output/benchmarks/perf/trend/latest.json absent; generate with "
                "python -m robot_sf.benchmark.perf_trend"
            ),
            "categories": None,
        }
    return {
        "status": "available",
        "value": None,
        "unit": None,
        "source": source,
        "source_gap": None,
        "categories": data,
    }


def _collect_escaped_defect(repo_root: Path) -> dict[str, Any]:
    """Collect escaped-defect status.

    Returns:
        Signal payload dict with status/value/unit/source/source_gap/categories.
    """
    return {
        "status": "unavailable",
        "value": None,
        "unit": None,
        "source": None,
        "source_gap": "no defect-escape tracking surface",
        "categories": None,
    }


SIGNAL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "key": "coverage_line",
        "label": "Line coverage",
        "formula": "covered_lines / (covered_lines + missing_lines) from coverage.py totals",
        "scope": "whole repository test run",
        "cadence": "per PR readiness run",
        "owner": "scripts/coverage/compare_coverage.py",
        "decision_use": "required regression floor; regressions block",
        "gate_class": "required",
        "collector": _collect_coverage_line,
    },
    {
        "key": "coverage_branch",
        "label": "Branch coverage",
        "formula": "covered_branches / (covered_branches + missing_branches) from coverage.py totals",
        "scope": "whole repository test run",
        "cadence": "per PR readiness run",
        "owner": "scripts/dev/branch_coverage_report.py",
        "decision_use": "diagnostic blind-spot signal; informs phased threshold schedule",
        "gate_class": "diagnostic",
        "collector": _collect_coverage_branch,
    },
    {
        "key": "coverage_changed_file",
        "label": "Changed-file coverage",
        "formula": "per-changed-file coverage from git diff + coverage.json (min 80 / goal 100)",
        "scope": "files changed vs base_ref",
        "cadence": "per PR readiness run",
        "owner": "scripts/coverage/check_changed_files_coverage.py",
        "decision_use": "required changed-file floor; computed live, not persisted",
        "gate_class": "required",
        "collector": _collect_coverage_changed_file,
    },
    {
        "key": "test_pass_rate",
        "label": "Test pass rate",
        "formula": "passed / (passed + failed + errored) from a junitxml artifact",
        "scope": "whole repository test run",
        "cadence": "per PR readiness run",
        "owner": "unowned (epic #6205)",
        "decision_use": "diagnostic health signal once an artifact owner exists",
        "gate_class": "diagnostic",
        "collector": _collect_test_pass_rate,
    },
    {
        "key": "collection_completeness",
        "label": "Test collection completeness",
        "formula": "collected_tests / expected_tests from a collected-test inventory",
        "scope": "whole repository test run",
        "cadence": "per PR readiness run",
        "owner": "unowned (issues #5753/#5108)",
        "decision_use": "diagnostic signal for silent test-collection drops",
        "gate_class": "diagnostic",
        "collector": _collect_collection_completeness,
    },
    {
        "key": "mutation",
        "label": "Mutation testing",
        "formula": "killed / survived / skipped / suspicious / timeout / total_mutants",
        "scope": "robot_sf/research/aggregation.py via tests/research/test_aggregation.py",
        "cadence": "scheduled diagnostic (never a required PR gate)",
        "owner": "scripts/dev/mutation_ratchet.py",
        "decision_use": "required ratchet on surviving-mutant set; new survivor fails",
        "gate_class": "required",
        "collector": _collect_mutation,
    },
    {
        "key": "test_duration",
        "label": "Test duration",
        "formula": "wall-clock test/step durations from CI timing logs",
        "scope": "CI test run",
        "cadence": "per CI run",
        "owner": "scripts/dev/ci_timing_summary.py",
        "decision_use": "diagnostic slowdown signal",
        "gate_class": "diagnostic",
        "collector": _collect_test_duration,
    },
    {
        "key": "timeout_budget_compliance",
        "label": "Timeout-budget compliance",
        "formula": "durations within declared per-step timeout budgets",
        "scope": "CI test run",
        "cadence": "per CI run",
        "owner": "unowned",
        "decision_use": "diagnostic budget signal once a budget gate owner exists",
        "gate_class": "diagnostic",
        "collector": _collect_timeout_budget_compliance,
    },
    {
        "key": "flaky_rerun_rate",
        "label": "Flaky rerun rate",
        "formula": "rerun_tests / total_tests from a flakiness registry",
        "scope": "CI test run",
        "cadence": "per CI run",
        "owner": "unowned (epic #6205 child 4)",
        "decision_use": "diagnostic flakiness signal",
        "gate_class": "diagnostic",
        "collector": _collect_flaky_rerun_rate,
    },
    {
        "key": "skip_xfail_age",
        "label": "Skip/xfail age",
        "formula": "age of skip/xfail markers from an xfail registry",
        "scope": "whole repository test suite",
        "cadence": "periodic",
        "owner": "unowned",
        "decision_use": "diagnostic stale-skip signal",
        "gate_class": "diagnostic",
        "collector": _collect_skip_xfail_age,
    },
    {
        "key": "hazard_odd_coverage",
        "label": "Hazard/ODD coverage",
        "formula": "hazard and ODD coverage rollup for benchmark campaign bundles",
        "scope": "benchmark campaign bundle",
        "cadence": "per campaign",
        "owner": "scripts/tools/hazard_odd_coverage_rollup.py",
        "decision_use": "diagnostic safety-coverage signal",
        "gate_class": "diagnostic",
        "collector": _collect_hazard_odd_coverage,
    },
    {
        "key": "scenario_certification",
        "label": "Scenario certification",
        "formula": "scenario_cert.v1 certificates per scenario manifest",
        "scope": "scenario manifests (robot_sf/scenario_certification/)",
        "cadence": "per scenario change",
        "owner": "scripts/tools/certify_scenarios.py",
        "decision_use": "diagnostic scenario-readiness signal",
        "gate_class": "diagnostic",
        "collector": _collect_scenario_certification,
    },
    {
        "key": "contract_coverage",
        "label": "Contract coverage",
        "formula": "contract-coverage test rollup",
        "scope": "contract tests",
        "cadence": "periodic",
        "owner": "unowned",
        "decision_use": "diagnostic contract-coverage signal once a rollup owner exists",
        "gate_class": "diagnostic",
        "collector": _collect_contract_coverage,
    },
    {
        "key": "compatibility_coverage",
        "label": "Compatibility coverage",
        "formula": "compat-coverage rollup across compat manifests",
        "scope": "compatibility manifests",
        "cadence": "periodic",
        "owner": "unowned",
        "decision_use": "diagnostic compatibility signal once a rollup owner exists",
        "gate_class": "diagnostic",
        "collector": _collect_compatibility_coverage,
    },
    {
        "key": "reproducibility_status",
        "label": "Benchmark reproducibility",
        "formula": "{status, reproducible} from the benchmark reproducibility check",
        "scope": "benchmark reproducibility check",
        "cadence": "per benchmark campaign",
        "owner": "scripts/benchmark_repro_check.py",
        "decision_use": "diagnostic reproducibility signal",
        "gate_class": "diagnostic",
        "collector": _collect_reproducibility_status,
    },
    {
        "key": "fallback_degraded_counts",
        "label": "Fallback/degraded row counts",
        "formula": "campaign row_status counts (native/adapter/diagnostic_only/fallback/degraded/unavailable/failed)",
        "scope": "campaign result stores under output/",
        "cadence": "per campaign",
        "owner": "scripts/tools/campaign_result_store.py",
        "decision_use": "required fail-closed signal; fallback/degraded never count as success",
        "gate_class": "required",
        "collector": _collect_fallback_degraded_counts,
    },
    {
        "key": "performance_regression",
        "label": "Performance regression",
        "formula": "perf-trend latest comparison against historical thresholds",
        "scope": "classic interaction scenarios",
        "cadence": "per perf-trend run",
        "owner": "robot_sf/benchmark/perf_trend.py",
        "decision_use": "diagnostic performance-regression signal",
        "gate_class": "diagnostic",
        "collector": _collect_performance_regression,
    },
    {
        "key": "escaped_defect",
        "label": "Escaped defects",
        "formula": "defects escaped to downstream use from a defect-escape tracker",
        "scope": "downstream defect tracking",
        "cadence": "periodic",
        "owner": "unowned",
        "decision_use": "diagnostic escaped-defect signal",
        "gate_class": "diagnostic",
        "collector": _collect_escaped_defect,
    },
]


def build_report(*, base_ref: str, require_clean_tree: bool, repo_root: Path) -> dict[str, Any]:
    """Assemble the stamped quality report from the configured signal collectors.

    Returns:
        Report dict with ``schema_version``, ``stamp``, and ``signals``; when
        ``require_clean_tree`` is set and the worktree is dirty, a fail-closed payload
        shaped like ``freshness._write_stamp`` (``ok=False``, ``reason=dirty_worktree``).
    """
    branch = freshness._current_branch()
    head_sha = freshness._head_sha()
    base_sha = freshness._resolve_base_sha(base_ref)
    tree_state = freshness._tree_state()

    if require_clean_tree and tree_state != "clean":
        return {
            "ok": False,
            "reason": "dirty_worktree",
            "branch": branch,
            "base_ref": base_ref,
            "base_sha": base_sha,
            "head_sha": head_sha,
            "tree_state": tree_state,
        }

    stamp = {
        "branch": branch,
        "base_ref": base_ref,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "tree_state": tree_state,
    }

    signals: dict[str, Any] = {}
    root = Path(repo_root)
    for definition in SIGNAL_DEFINITIONS:
        collected = definition["collector"](root)
        signals[definition["key"]] = {
            "label": definition["label"],
            "formula": definition["formula"],
            "scope": definition["scope"],
            "cadence": definition["cadence"],
            "owner": definition["owner"],
            "decision_use": definition["decision_use"],
            "gate_class": definition["gate_class"],
            "status": collected["status"],
            "value": collected.get("value"),
            "unit": collected.get("unit"),
            "source": collected.get("source"),
            "source_gap": collected.get("source_gap"),
            "categories": collected.get("categories"),
        }

    return {"schema_version": SCHEMA_VERSION, "stamp": stamp, "signals": signals}


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable Markdown summary of a report payload.

    Returns:
        Markdown text with the stamp and a per-signal table (or the fail-closed reason).
    """
    if "schema_version" not in report:
        return "\n".join(
            [
                "# Quality Report",
                "",
                f"- ok: {report.get('ok')}",
                f"- reason: {report.get('reason')}",
                f"- branch: {report.get('branch')}",
                f"- head_sha: {report.get('head_sha')}",
                f"- tree_state: {report.get('tree_state')}",
            ]
        )

    stamp = report["stamp"]
    lines = [
        "# Quality Report",
        "",
        f"- schema_version: {report['schema_version']}",
        f"- branch: {stamp['branch']}",
        f"- base_ref: {stamp['base_ref']}",
        f"- base_sha: {stamp['base_sha']}",
        f"- head_sha: {stamp['head_sha']}",
        f"- recorded_at_utc: {stamp['recorded_at_utc']}",
        f"- tree_state: {stamp['tree_state']}",
        "",
        "| signal | gate_class | status | value | source_gap |",
        "|---|---|---|---|---|",
    ]
    for key, signal in report["signals"].items():
        source_gap = signal.get("source_gap") or ""
        lines.append(
            f"| {key} | {signal['gate_class']} | {signal['status']} "
            f"| {signal.get('value')} | {source_gap} |"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build the quality-report CLI parser.

    Returns:
        Configured argument parser for the quality-report generator.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument(
        "--require-clean-tree",
        action="store_true",
        help="Fail closed (exit 2) instead of reporting from a dirty worktree.",
    )
    parser.add_argument("--output-dir", default="output/quality")
    parser.add_argument("--repo-root", default=".")
    return parser


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code: ``0`` on success, ``2`` when a clean tree is required but the
        worktree is dirty.
    """
    args = _build_parser().parse_args()
    repo_root = Path(args.repo_root)
    report = build_report(
        base_ref=args.base_ref,
        require_clean_tree=args.require_clean_tree,
        repo_root=repo_root,
    )

    payload = json.dumps(report, indent=2, sort_keys=True)
    if "schema_version" not in report:
        print(payload)
        return 2

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quality_report.json").write_text(payload + "\n", encoding="utf-8")
    (output_dir / "quality_report.md").write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
