"""Trend-oriented performance benchmark runner for classic interaction scenarios.

This module composes multiple ``perf_cold_warm`` scenario checks into a stable matrix run,
then optionally compares the current report against recent history snapshots.
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from loguru import logger

from robot_sf.benchmark import perf_cold_warm
from robot_sf.common.artifact_paths import ensure_canonical_tree, get_artifact_category_path

if TYPE_CHECKING:
    from collections.abc import Sequence


TREND_REPORT_SCHEMA_VERSION = "benchmark-perf-trend-report.v1"
TREND_MATRIX_SCHEMA_VERSION = "benchmark-perf-trend-matrix.v1"
TIME_METRICS = ("env_create_sec", "first_step_sec", "episode_sec")
ALL_METRICS = (*TIME_METRICS, "steps_per_sec")
STARTUP_METRICS = ("env_create_sec", "first_step_sec")
STEADY_METRICS = ("episode_sec", "steps_per_sec")


@dataclass(slots=True)
class ScenarioSpec:
    """Single scenario configuration for trend benchmark runs."""

    scenario_config: Path
    scenario_name: str
    episode_steps: int
    cold_runs: int
    warm_runs: int
    baseline: Path
    require_baseline: bool
    max_slowdown_pct: float
    max_throughput_drop_pct: float
    min_seconds_delta: float
    min_throughput_delta: float
    enforce_regression_gate: bool


@dataclass(slots=True)
class HistoryThresholds:
    """Regression thresholds for historical trend comparisons."""

    max_slowdown_pct: float = 0.35
    max_throughput_drop_pct: float = 0.30
    min_seconds_delta: float = 0.10
    min_throughput_delta: float = 0.75


@dataclass(slots=True)
class TrendFinding:
    """Regression finding versus historical medians."""

    scenario: str
    phase: str
    metric: str
    baseline: float
    current: float
    delta: float
    delta_pct: float
    is_regression: bool
    threshold_pct: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for matrix trend benchmarking.

    Args:
        argv: Optional argument vector override.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/benchmarks/perf_trend_matrix_classic_v1.yaml"),
        help="Scenario matrix YAML with perf_cold_warm settings per scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Base random seed used for deterministic matrix runs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=get_artifact_category_path("benchmarks") / "perf/trend/latest.json",
        help="Trend report JSON output path.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=get_artifact_category_path("benchmarks") / "perf/trend/latest.md",
        help="Trend report markdown output path.",
    )
    parser.add_argument(
        "--history-glob",
        type=str,
        default="",
        help="Optional glob for previous trend report JSON files.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=10,
        help="Maximum number of recent history reports to compare.",
    )
    parser.add_argument(
        "--max-history-slowdown-pct",
        type=float,
        default=0.35,
        help="Allowed slowdown for *_sec metrics versus history median.",
    )
    parser.add_argument(
        "--max-history-throughput-drop-pct",
        type=float,
        default=0.30,
        help="Allowed drop for steps_per_sec versus history median.",
    )
    parser.add_argument(
        "--min-history-seconds-delta",
        type=float,
        default=0.10,
        help="Absolute floor for time metric regression detection.",
    )
    parser.add_argument(
        "--min-history-throughput-delta",
        type=float,
        default=0.75,
        help="Absolute floor for throughput regression detection.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when any scenario-level perf_cold_warm run regresses.",
    )
    parser.add_argument(
        "--fail-on-history-regression",
        action="store_true",
        help="Exit non-zero when history comparison reports regressions.",
    )
    return parser.parse_args(argv)


def load_matrix(path: Path) -> tuple[str, list[ScenarioSpec]]:
    """Load and validate the matrix definition file.

    Args:
        path: Matrix YAML path.

    Returns:
        tuple[str, list[ScenarioSpec]]: Suite name and parsed scenario specs.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid matrix payload in {path}: expected object")
    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version != TREND_MATRIX_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported matrix schema_version {schema_version!r} in {path}; "
            f"expected {TREND_MATRIX_SCHEMA_VERSION!r}"
        )

    suite_name = str(payload.get("suite_name") or "").strip()
    if not suite_name:
        raise ValueError(f"Missing suite_name in {path}")

    scenario_payload = payload.get("scenarios")
    if not isinstance(scenario_payload, list) or not scenario_payload:
        raise ValueError(f"Matrix {path} must include a non-empty scenarios list")

    scenarios: list[ScenarioSpec] = []
    for idx, entry in enumerate(scenario_payload):
        if not isinstance(entry, dict):
            raise ValueError(f"Matrix scenario #{idx} in {path} is not an object")
        scenario_name = str(entry.get("scenario_name") or "").strip()
        if not scenario_name:
            raise ValueError(f"Matrix scenario #{idx} in {path} is missing scenario_name")
        scenario_config_raw = str(entry.get("scenario_config") or "").strip()
        if not scenario_config_raw:
            raise ValueError(f"Matrix scenario '{scenario_name}' is missing scenario_config")
        scenarios.append(
            ScenarioSpec(
                scenario_config=Path(scenario_config_raw),
                scenario_name=scenario_name,
                episode_steps=int(entry.get("episode_steps", 96)),
                cold_runs=int(entry.get("cold_runs", 1)),
                warm_runs=int(entry.get("warm_runs", 2)),
                baseline=Path(
                    str(
                        entry.get("baseline")
                        or "configs/benchmarks/perf_baseline_classic_cold_warm_v1.json"
                    )
                ),
                require_baseline=bool(entry.get("require_baseline", True)),
                max_slowdown_pct=float(entry.get("max_slowdown_pct", 0.60)),
                max_throughput_drop_pct=float(entry.get("max_throughput_drop_pct", 0.50)),
                min_seconds_delta=float(entry.get("min_seconds_delta", 0.15)),
                min_throughput_delta=float(entry.get("min_throughput_delta", 0.75)),
                enforce_regression_gate=bool(entry.get("enforce_regression_gate", True)),
            )
        )
    return suite_name, scenarios


def _run_scenario(
    *,
    spec: ScenarioSpec,
    seed: int,
    output_root: Path,
    fail_on_regression: bool,
) -> dict[str, Any]:
    """Run one ``perf_cold_warm`` scenario and load its JSON result.

    Args:
        spec: Matrix scenario configuration.
        seed: Scenario seed.
        output_root: Root output directory for scenario artifacts.
        fail_on_regression: Whether scenario-level regression should fail.

    Returns:
        dict[str, Any]: Scenario execution summary and parsed perf payload fields.
    """
    scenario_slug = spec.scenario_name.replace("/", "_")
    scenario_json = output_root / "runs" / f"{scenario_slug}.json"
    scenario_md = output_root / "runs" / f"{scenario_slug}.md"

    argv = [
        "--scenario-config",
        str(spec.scenario_config),
        "--scenario-name",
        spec.scenario_name,
        "--seed",
        str(seed),
        "--episode-steps",
        str(spec.episode_steps),
        "--cold-runs",
        str(spec.cold_runs),
        "--warm-runs",
        str(spec.warm_runs),
        "--baseline",
        str(spec.baseline),
        "--output-json",
        str(scenario_json),
        "--output-markdown",
        str(scenario_md),
        "--max-slowdown-pct",
        str(spec.max_slowdown_pct),
        "--max-throughput-drop-pct",
        str(spec.max_throughput_drop_pct),
        "--min-seconds-delta",
        str(spec.min_seconds_delta),
        "--min-throughput-delta",
        str(spec.min_throughput_delta),
    ]
    if spec.require_baseline:
        argv.append("--require-baseline")
    if fail_on_regression and spec.enforce_regression_gate:
        argv.append("--fail-on-regression")

    exit_code = perf_cold_warm.main(argv)
    payload: dict[str, Any] = {}
    if scenario_json.exists():
        try:
            loaded = json.loads(scenario_json.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except json.JSONDecodeError:
            payload = {}

    comparison = payload.get("comparison")
    comparison_status = (
        str(comparison.get("status") or "no-output")
        if isinstance(comparison, dict)
        else "no-output"
    )

    return {
        "scenario_name": spec.scenario_name,
        "scenario_config": str(spec.scenario_config),
        "seed": seed,
        "exit_code": int(exit_code),
        "enforce_regression_gate": bool(spec.enforce_regression_gate),
        "comparison_status": comparison_status,
        "output_json": str(scenario_json),
        "output_markdown": str(scenario_md),
        "cold_median": payload.get("cold", {}).get("median", {}),
        "warm_median": payload.get("warm", {}).get("median", {}),
        "raw": payload,
    }


def _load_history_reports(pattern: str, limit: int) -> list[dict[str, Any]]:
    """Load recent trend reports matching ``pattern``.

    Args:
        pattern: Glob pattern for report JSON files.
        limit: Maximum number of reports to load.

    Returns:
        list[dict[str, Any]]: Parsed report payloads with `_source_path`.
    """
    if not pattern.strip():
        return []
    candidates = sorted((Path(p) for p in glob.glob(pattern)), key=lambda p: p.stat().st_mtime)
    if limit > 0:
        candidates = candidates[-limit:]
    reports: list[dict[str, Any]] = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("schema_version") != TREND_REPORT_SCHEMA_VERSION:
            continue
        payload["_source_path"] = str(path)
        reports.append(payload)
    return reports


def _scenario_metric_values(
    reports: Sequence[dict[str, Any]],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Build ``scenario -> phase -> metric -> values`` from history reports.

    Args:
        reports: Parsed history report payloads.

    Returns:
        dict[str, dict[str, dict[str, list[float]]]]: Per-scenario phase metric history values.
    """
    values: dict[str, dict[str, dict[str, list[float]]]] = {}
    for report in reports:
        scenario_results = report.get("scenario_results")
        if not isinstance(scenario_results, list):
            continue
        for entry in scenario_results:
            if not isinstance(entry, dict):
                continue
            scenario_name = str(entry.get("scenario_name") or "").strip()
            if not scenario_name:
                continue
            scenario_bucket = values.setdefault(scenario_name, {"cold": {}, "warm": {}})
            for phase in ("cold", "warm"):
                phase_payload = entry.get(f"{phase}_median")
                if not isinstance(phase_payload, dict):
                    continue
                metric_bucket = scenario_bucket.setdefault(phase, {})
                for metric in ALL_METRICS:
                    value = phase_payload.get(metric)
                    if isinstance(value, (int, float)):
                        metric_bucket.setdefault(metric, []).append(float(value))
    return values


def compare_with_history(
    *,
    current_results: Sequence[dict[str, Any]],
    history_reports: Sequence[dict[str, Any]],
    thresholds: HistoryThresholds,
) -> dict[str, Any]:
    """Compare current matrix results with historical medians.

    Args:
        current_results: Current matrix scenario results.
        history_reports: Prior trend report payloads.
        thresholds: Regression thresholds for historical comparisons.

    Returns:
        dict[str, Any]: Status, sources, per-metric findings, and diagnostics.
    """
    if not history_reports:
        return {
            "status": "no-history",
            "findings": [],
            "diagnostics": ["No history reports found."],
        }

    history_values = _scenario_metric_values(history_reports)
    findings: list[TrendFinding] = []

    for current in current_results:
        scenario_name = str(current.get("scenario_name") or "")
        if not scenario_name:
            continue
        scenario_history = history_values.get(scenario_name)
        if scenario_history is None:
            continue
        for phase in ("cold", "warm"):
            current_phase = current.get(f"{phase}_median")
            if not isinstance(current_phase, dict):
                continue
            phase_history = scenario_history.get(phase, {})
            for metric in ALL_METRICS:
                finding = _build_history_finding(
                    scenario_name=scenario_name,
                    phase=phase,
                    metric=metric,
                    current_phase=current_phase,
                    phase_history=phase_history,
                    thresholds=thresholds,
                )
                if finding is not None:
                    findings.append(finding)

    status, diagnostics = _history_status_and_diagnostics(findings)

    return {
        "status": status,
        "history_sources": [str(report.get("_source_path", "")) for report in history_reports],
        "findings": [asdict(finding) for finding in findings],
        "diagnostics": diagnostics,
    }


def _build_history_finding(
    *,
    scenario_name: str,
    phase: str,
    metric: str,
    current_phase: dict[str, Any],
    phase_history: dict[str, list[float]],
    thresholds: HistoryThresholds,
) -> TrendFinding | None:
    """Build one historical comparison finding when inputs are valid.

    Returns:
        TrendFinding | None: Finding for this metric, else ``None``.
    """
    history_metric = phase_history.get(metric, [])
    if not history_metric:
        return None
    current_value = current_phase.get(metric)
    if not isinstance(current_value, (int, float)):
        return None
    baseline = float(np.median(history_metric))
    current_float = float(current_value)
    delta = current_float - baseline
    delta_pct = 0.0 if abs(baseline) < 1e-9 else (delta / baseline) * 100.0
    is_regression, threshold_pct = _is_history_regression(
        metric=metric,
        baseline=baseline,
        current=current_float,
        thresholds=thresholds,
    )
    return TrendFinding(
        scenario=scenario_name,
        phase=phase,
        metric=metric,
        baseline=baseline,
        current=current_float,
        delta=delta,
        delta_pct=delta_pct,
        is_regression=is_regression,
        threshold_pct=threshold_pct,
    )


def _is_history_regression(
    *,
    metric: str,
    baseline: float,
    current: float,
    thresholds: HistoryThresholds,
) -> tuple[bool, float]:
    """Return regression classification and threshold for one metric.

    Returns:
        tuple[bool, float]: ``(is_regression, threshold_pct)``.
    """
    if metric == "steps_per_sec":
        threshold_pct = thresholds.max_throughput_drop_pct * 100.0
        is_regression = (
            baseline > 0.0
            and current < baseline * (1.0 - thresholds.max_throughput_drop_pct)
            and (baseline - current) >= thresholds.min_throughput_delta
        )
        return is_regression, threshold_pct
    threshold_pct = thresholds.max_slowdown_pct * 100.0
    is_regression = (
        current > baseline * (1.0 + thresholds.max_slowdown_pct)
        and (current - baseline) >= thresholds.min_seconds_delta
    )
    return is_regression, threshold_pct


def _history_status_and_diagnostics(findings: Sequence[TrendFinding]) -> tuple[str, list[str]]:
    """Compute top-level history status and user-facing diagnostics.

    Returns:
        tuple[str, list[str]]: Status (`pass`/`fail`) and diagnostic lines.
    """
    regressed = [finding for finding in findings if finding.is_regression]
    if not regressed:
        return "pass", ["No historical regressions detected."]

    startup = [finding for finding in regressed if finding.metric in STARTUP_METRICS]
    steady = [finding for finding in regressed if finding.metric in STEADY_METRICS]
    diagnostics: list[str] = []
    if startup and not steady:
        diagnostics.append("Historical regression localized to startup overhead.")
    elif steady and not startup:
        diagnostics.append("Historical regression localized to steady-state stepping throughput.")
    else:
        diagnostics.append("Historical regression spans startup and steady-state behavior.")
    diagnostics.append(
        "Regressed metrics: "
        + ", ".join(f"{finding.scenario}.{finding.phase}.{finding.metric}" for finding in regressed)
    )
    return "fail", diagnostics


def render_markdown_report(
    *,
    suite_name: str,
    matrix_path: Path,
    scenario_results: Sequence[dict[str, Any]],
    history_comparison: dict[str, Any],
) -> str:
    """Render markdown summary for trend matrix execution.

    Returns:
        str: Markdown report content.
    """
    lines = [
        "# Performance Trend Benchmark Report",
        "",
        f"- Suite: `{suite_name}`",
        f"- Matrix: `{matrix_path}`",
        f"- Generated at (UTC): `{datetime.now(UTC).isoformat()}`",
        "",
        "## Scenario Results",
        "",
        "| Scenario | Exit | Comparison | cold.steps_per_sec | warm.steps_per_sec |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for result in scenario_results:
        cold = result.get("cold_median", {}) if isinstance(result.get("cold_median"), dict) else {}
        warm = result.get("warm_median", {}) if isinstance(result.get("warm_median"), dict) else {}
        lines.append(
            "| "
            f"{result.get('scenario_name')} | "
            f"{result.get('exit_code')} | "
            f"{result.get('comparison_status')} | "
            f"{float(cold.get('steps_per_sec', 0.0)):.3f} | "
            f"{float(warm.get('steps_per_sec', 0.0)):.3f} |"
        )

    lines.extend(["", "## History Comparison", ""])
    lines.append(f"Status: **{str(history_comparison.get('status', 'unknown')).upper()}**")
    for diag in history_comparison.get("diagnostics", []):
        lines.append(f"- {diag}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run matrix benchmark and optional history comparison.

    Args:
        argv: Optional argument vector override.

    Returns:
        int: Process exit code.
    """
    ensure_canonical_tree(categories=("benchmarks",))
    args = parse_args(argv)

    suite_name, scenarios = load_matrix(args.matrix)
    output_root = args.output_json.parent
    output_root.mkdir(parents=True, exist_ok=True)

    scenario_results: list[dict[str, Any]] = []
    exit_code = 0
    for idx, spec in enumerate(scenarios):
        scenario_seed = int(args.seed) + idx * 100
        result = _run_scenario(
            spec=spec,
            seed=scenario_seed,
            output_root=output_root,
            fail_on_regression=bool(args.fail_on_regression),
        )
        scenario_results.append(result)
        if result["exit_code"] != 0 and args.fail_on_regression:
            exit_code = 1

    history_reports = _load_history_reports(args.history_glob, max(0, int(args.history_limit)))
    history_comparison = compare_with_history(
        current_results=scenario_results,
        history_reports=history_reports,
        thresholds=HistoryThresholds(
            max_slowdown_pct=float(args.max_history_slowdown_pct),
            max_throughput_drop_pct=float(args.max_history_throughput_drop_pct),
            min_seconds_delta=float(args.min_history_seconds_delta),
            min_throughput_delta=float(args.min_history_throughput_delta),
        ),
    )

    if history_comparison.get("status") == "fail" and args.fail_on_history_regression:
        exit_code = 1

    payload = {
        "schema_version": TREND_REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "suite_name": suite_name,
        "matrix_path": str(args.matrix),
        "scenario_count": len(scenario_results),
        "scenario_results": scenario_results,
        "history_comparison": history_comparison,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    markdown = render_markdown_report(
        suite_name=suite_name,
        matrix_path=args.matrix,
        scenario_results=scenario_results,
        history_comparison=history_comparison,
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")

    logger.info("Trend JSON: {}", args.output_json)
    logger.info("Trend Markdown: {}", args.output_markdown)
    logger.info("History comparison status: {}", history_comparison.get("status", "unknown"))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
