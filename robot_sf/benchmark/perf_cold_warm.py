"""Cold/warm performance regression checks for classic interaction scenarios.

This module provides:
1. deterministic cold/warm benchmark sampling,
2. baseline comparison with relative thresholds, and
3. JSON/Markdown reporting for CI workflows.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from loguru import logger

from robot_sf.common.artifact_paths import ensure_canonical_tree, get_artifact_category_path
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

if TYPE_CHECKING:
    from collections.abc import Sequence

    from robot_sf.gym_env.unified_config import RobotSimulationConfig


_PHASES = ("cold", "warm")
_TIME_METRICS = ("env_create_sec", "first_step_sec", "episode_sec")
_STARTUP_METRICS = ("env_create_sec", "first_step_sec")
_STEADY_METRICS = ("episode_sec", "steps_per_sec")
_COLD_SUBPROCESS_TIMEOUT_SEC = 300


@dataclass(slots=True)
class PhaseMetrics:
    """Performance metrics for a single benchmark phase sample."""

    env_create_sec: float
    first_step_sec: float
    episode_sec: float
    steps_per_sec: float

    def to_dict(self) -> dict[str, float]:
        """Serialize metrics to a JSON-friendly mapping.

        Returns:
            dict[str, float]: Flat mapping of phase metric names to values.
        """
        return {
            "env_create_sec": float(self.env_create_sec),
            "first_step_sec": float(self.first_step_sec),
            "episode_sec": float(self.episode_sec),
            "steps_per_sec": float(self.steps_per_sec),
        }

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> PhaseMetrics:
        """Build metrics from a mapping.

        Args:
            payload: Mapping containing metric keys.

        Returns:
            PhaseMetrics: Parsed performance metrics.
        """
        return cls(
            env_create_sec=float(payload["env_create_sec"]),
            first_step_sec=float(payload["first_step_sec"]),
            episode_sec=float(payload["episode_sec"]),
            steps_per_sec=float(payload["steps_per_sec"]),
        )


@dataclass(slots=True)
class SuiteSnapshot:
    """Cold/warm metric summary snapshot."""

    cold: PhaseMetrics
    warm: PhaseMetrics

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Serialize the snapshot to a mapping.

        Returns:
            dict[str, dict[str, float]]: Serialized cold/warm phase metrics.
        """
        return {"cold": self.cold.to_dict(), "warm": self.warm.to_dict()}


@dataclass(slots=True)
class RegressionThresholds:
    """Relative and absolute guards for regression detection."""

    max_slowdown_pct: float = 0.60
    max_throughput_drop_pct: float = 0.50
    min_seconds_delta: float = 0.15
    min_throughput_delta: float = 0.75


@dataclass(slots=True)
class RegressionFinding:
    """Single metric comparison result."""

    phase: str
    metric: str
    baseline: float
    current: float
    delta: float
    delta_pct: float
    is_regression: bool
    threshold_pct: float


@dataclass(slots=True)
class RegressionReport:
    """Regression report summarizing metric comparisons."""

    status: str
    findings: tuple[RegressionFinding, ...]
    diagnostics: tuple[str, ...]

    @property
    def has_regression(self) -> bool:
        """Return whether at least one metric regressed.

        Returns:
            bool: ``True`` when any finding has ``is_regression=True``.
        """
        return any(finding.is_regression for finding in self.findings)

    @property
    def has_startup_regression(self) -> bool:
        """Return whether startup metrics contain any regression."""
        return any(
            finding.is_regression and finding.metric in _STARTUP_METRICS
            for finding in self.findings
        )

    @property
    def has_steady_regression(self) -> bool:
        """Return whether steady-state metrics contain any regression."""
        return any(
            finding.is_regression and finding.metric in _STEADY_METRICS for finding in self.findings
        )

    @property
    def has_blocking_regression(self) -> bool:
        """Return whether a regression should fail CI gating."""
        return self.has_steady_regression

    @property
    def failure_class(self) -> Literal["none", "startup_only", "steady"]:
        """Classify regression severity for gating/reporting."""
        if self.has_steady_regression:
            return "steady"
        if self.has_startup_regression:
            return "startup_only"
        return "none"


def _regression_status(findings: Sequence[RegressionFinding]) -> Literal["pass", "warn", "fail"]:
    """Map metric regressions to top-level status.

    Args:
        findings: Per-metric regression findings from snapshot comparison.

    Returns:
        Literal["pass", "warn", "fail"]: Aggregated status classification.
    """
    has_startup = any(f.is_regression and f.metric in _STARTUP_METRICS for f in findings)
    has_steady = any(f.is_regression and f.metric in _STEADY_METRICS for f in findings)
    if has_steady:
        return "fail"
    if has_startup:
        return "warn"
    return "pass"


def median_metrics(samples: Sequence[PhaseMetrics]) -> PhaseMetrics:
    """Compute median metrics over repeated samples.

    Args:
        samples: Repeated phase samples.

    Returns:
        PhaseMetrics: Median values per metric.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    return PhaseMetrics(
        env_create_sec=float(np.median([s.env_create_sec for s in samples])),
        first_step_sec=float(np.median([s.first_step_sec for s in samples])),
        episode_sec=float(np.median([s.episode_sec for s in samples])),
        steps_per_sec=float(np.median([s.steps_per_sec for s in samples])),
    )


def compare_snapshots(
    current: SuiteSnapshot,
    baseline: SuiteSnapshot,
    thresholds: RegressionThresholds,
) -> RegressionReport:
    """Compare current cold/warm metrics against baseline.

    Args:
        current: Current benchmark medians.
        baseline: Baseline benchmark medians.
        thresholds: Regression thresholds.

    Returns:
        RegressionReport: Detailed comparison findings and status.
    """
    findings: list[RegressionFinding] = []
    for phase in _PHASES:
        current_phase = getattr(current, phase)
        baseline_phase = getattr(baseline, phase)
        for metric in (*_TIME_METRICS, "steps_per_sec"):
            baseline_val = float(getattr(baseline_phase, metric))
            current_val = float(getattr(current_phase, metric))
            delta = current_val - baseline_val
            delta_pct = 0.0 if abs(baseline_val) < 1e-9 else (delta / baseline_val) * 100.0
            if metric == "steps_per_sec":
                threshold_pct = thresholds.max_throughput_drop_pct * 100.0
                is_regression = (
                    baseline_val > 0.0
                    and current_val < baseline_val * (1.0 - thresholds.max_throughput_drop_pct)
                    and (baseline_val - current_val) >= thresholds.min_throughput_delta
                )
            else:
                threshold_pct = thresholds.max_slowdown_pct * 100.0
                is_regression = (
                    current_val > baseline_val * (1.0 + thresholds.max_slowdown_pct)
                    and (current_val - baseline_val) >= thresholds.min_seconds_delta
                )
            findings.append(
                RegressionFinding(
                    phase=phase,
                    metric=metric,
                    baseline=baseline_val,
                    current=current_val,
                    delta=delta,
                    delta_pct=delta_pct,
                    is_regression=is_regression,
                    threshold_pct=threshold_pct,
                )
            )

    status = _regression_status(findings)
    diagnostics = _build_diagnostics(findings)
    return RegressionReport(status=status, findings=tuple(findings), diagnostics=diagnostics)


def load_snapshot(path: Path) -> SuiteSnapshot | None:
    """Load a baseline snapshot from disk.

    Supports either:
    - direct ``{"cold": {...}, "warm": {...}}`` metric shape, or
    - nested ``{"cold": {"median": {...}}, "warm": {"median": {...}}}``.

    Args:
        path: Snapshot file path.

    Returns:
        SuiteSnapshot | None: Parsed snapshot, or ``None`` if not available/invalid.
    """
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    def _extract_phase(phase: str) -> dict[str, Any]:
        phase_payload = payload.get(phase, {})
        if isinstance(phase_payload, dict) and isinstance(phase_payload.get("median"), dict):
            return phase_payload["median"]
        if isinstance(phase_payload, dict):
            return phase_payload
        raise TypeError(f"Invalid phase payload for '{phase}'")

    try:
        return SuiteSnapshot(
            cold=PhaseMetrics.from_mapping(_extract_phase("cold")),
            warm=PhaseMetrics.from_mapping(_extract_phase("warm")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def render_markdown_report(
    *,
    scenario_label: str,
    episode_steps: int,
    cold_runs: int,
    warm_runs: int,
    current: SuiteSnapshot,
    baseline: SuiteSnapshot | None,
    report: RegressionReport | None,
) -> str:
    """Build a concise markdown summary.

    Args:
        scenario_label: Human-readable benchmark scenario label.
        episode_steps: Steps measured per sample.
        cold_runs: Number of cold samples.
        warm_runs: Number of warm samples.
        current: Current benchmark medians.
        baseline: Optional baseline snapshot.
        report: Optional regression report.

    Returns:
        str: Markdown report suitable for CI artifacts.
    """
    lines = [
        "# Cold/Warm Performance Regression Report",
        "",
        f"- Scenario: `{scenario_label}`",
        f"- Episode steps: `{episode_steps}`",
        f"- Cold samples: `{cold_runs}`",
        f"- Warm samples: `{warm_runs}`",
        "",
        "## Current Medians",
        "",
        "| Phase | env_create_sec | first_step_sec | episode_sec | steps_per_sec |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for phase in _PHASES:
        metrics = getattr(current, phase)
        lines.append(
            "| "
            f"{phase} | {metrics.env_create_sec:.3f} | {metrics.first_step_sec:.3f} | "
            f"{metrics.episode_sec:.3f} | {metrics.steps_per_sec:.3f} |"
        )

    if baseline is None:
        lines.extend(["", "## Baseline Comparison", "", "No baseline snapshot available."])
        return "\n".join(lines)

    lines.extend(["", "## Baseline Comparison", ""])
    lines.append(
        f"Status: **{report.status.upper()}**" if report is not None else "Status: **UNKNOWN**"
    )
    if report is not None and report.failure_class == "startup_only":
        lines.append("Startup-only regression detected; steady-state gate not violated.")
    lines.extend(
        [
            "",
            "| Phase | Metric | Baseline | Current | Delta | Delta % | Regression |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if report is not None:
        for finding in report.findings:
            lines.append(
                "| "
                f"{finding.phase} | {finding.metric} | {finding.baseline:.3f} | "
                f"{finding.current:.3f} | {finding.delta:+.3f} | {finding.delta_pct:+.1f}% | "
                f"{'yes' if finding.is_regression else 'no'} |"
            )
        if report.diagnostics:
            lines.extend(["", "### Diagnostics"])
            for diag in report.diagnostics:
                lines.append(f"- {diag}")
    return "\n".join(lines)


def measure_once(
    *,
    config: RobotSimulationConfig,
    seed: int,
    episode_steps: int,
) -> PhaseMetrics:
    """Measure one in-process benchmark sample.

    Args:
        config: Simulation configuration used to construct the environment.
        seed: Seed passed to reset calls.
        episode_steps: Number of steps to execute.

    Returns:
        PhaseMetrics: Measured performance sample.
    """
    if episode_steps <= 0:
        raise ValueError("episode_steps must be > 0")

    create_started = time.perf_counter()
    env = make_robot_env(config=copy.deepcopy(config), debug=False)
    env_create_sec = time.perf_counter() - create_started

    action = (0.0, 0.0)
    steps_executed = 0
    try:
        episode_started = time.perf_counter()
        _, _ = env.reset(seed=int(seed))

        first_step_started = time.perf_counter()
        _, _, terminated, truncated, _ = env.step(action)
        first_step_sec = time.perf_counter() - first_step_started
        steps_executed = 1

        while steps_executed < episode_steps:
            if terminated or truncated:
                _, _ = env.reset(seed=int(seed + steps_executed))
            _, _, terminated, truncated, _ = env.step(action)
            steps_executed += 1

        episode_sec = time.perf_counter() - episode_started
    finally:
        env.close()

    steps_per_sec = float(steps_executed) / float(episode_sec) if episode_sec > 1e-9 else 0.0
    return PhaseMetrics(
        env_create_sec=env_create_sec,
        first_step_sec=first_step_sec,
        episode_sec=episode_sec,
        steps_per_sec=steps_per_sec,
    )


def run_suite(
    *,
    config: RobotSimulationConfig,
    script_path: Path,
    scenario_config: Path,
    scenario_name: str,
    seed: int,
    episode_steps: int,
    cold_runs: int,
    warm_runs: int,
) -> tuple[list[PhaseMetrics], list[PhaseMetrics]]:
    """Execute cold/warm sample collection.

    Args:
        config: Resolved scenario configuration.
        script_path: Path to this module file for subprocess cold runs.
        scenario_config: Scenario YAML path.
        scenario_name: Scenario identifier.
        seed: Base random seed.
        episode_steps: Number of steps per sample.
        cold_runs: Number of cold subprocess samples.
        warm_runs: Number of warm in-process samples.

    Returns:
        tuple[list[PhaseMetrics], list[PhaseMetrics]]: Cold and warm sample lists.
    """
    cold_samples = [
        _measure_cold_subprocess(
            script_path=script_path,
            scenario_config=scenario_config,
            scenario_name=scenario_name,
            seed=seed + idx,
            episode_steps=episode_steps,
        )
        for idx in range(cold_runs)
    ]

    # Warm path: pay initialization once in-process, then measure repeated runs.
    _ = measure_once(config=config, seed=seed + 1000, episode_steps=episode_steps)
    warm_samples = [
        measure_once(config=config, seed=seed + 1100 + idx, episode_steps=episode_steps)
        for idx in range(warm_runs)
    ]
    return cold_samples, warm_samples


def _positive_int(value: str) -> int:
    """Argparse validator for positive integer values.

    Args:
        value: Raw CLI value.

    Returns:
        int: Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If the value is not a positive integer.
    """
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be > 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for cold/warm perf checks.

    Args:
        argv: Optional argument sequence override.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario-config",
        type=Path,
        default=Path("configs/scenarios/archetypes/classic_crossing.yaml"),
        help="Scenario YAML used for benchmark config loading.",
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        default="classic_crossing_low",
        help="Scenario name inside the scenario YAML.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Base random seed used for repeated samples.",
    )
    parser.add_argument(
        "--episode-steps",
        type=_positive_int,
        default=64,
        help="Number of steps executed per sample.",
    )
    parser.add_argument(
        "--cold-runs",
        type=_positive_int,
        default=1,
        help="Number of cold subprocess runs (fresh interpreter each run).",
    )
    parser.add_argument(
        "--warm-runs",
        type=_positive_int,
        default=2,
        help="Number of warm in-process runs after warmup.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("configs/benchmarks/perf_baseline_classic_cold_warm_v1.json"),
        help="Baseline snapshot path used for relative regression checks.",
    )
    parser.add_argument(
        "--max-slowdown-pct",
        type=float,
        default=0.60,
        help="Allowed slowdown for *_sec metrics before regression is flagged.",
    )
    parser.add_argument(
        "--max-throughput-drop-pct",
        type=float,
        default=0.50,
        help="Allowed drop for steps_per_sec before regression is flagged.",
    )
    parser.add_argument(
        "--min-seconds-delta",
        type=float,
        default=0.15,
        help="Minimum absolute seconds delta to treat as meaningful regression.",
    )
    parser.add_argument(
        "--min-throughput-delta",
        type=float,
        default=0.75,
        help="Minimum absolute steps/sec drop to treat as meaningful regression.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=get_artifact_category_path("benchmarks") / "perf/cold_warm_perf_check.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=get_artifact_category_path("benchmarks") / "perf/cold_warm_perf_check.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when regressions are detected.",
    )
    parser.add_argument(
        "--require-baseline",
        action="store_true",
        help="Exit non-zero when the baseline file is missing or invalid.",
    )
    parser.add_argument(
        "--internal-measure-once",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run cold/warm performance regression checks.

    Args:
        argv: Optional CLI argument overrides.

    Returns:
        int: Process exit code.
    """
    ensure_canonical_tree(categories=("benchmarks",))
    args = parse_args(argv)

    if args.internal_measure_once:
        config, _ = _load_scenario_config(
            args.scenario_config,
            args.scenario_name,
            args.episode_steps,
        )
        sample = measure_once(config=config, seed=args.seed, episode_steps=args.episode_steps)
        sys.stdout.write(json.dumps(sample.to_dict()) + "\n")
        return 0

    config, scenario_label = _load_scenario_config(
        args.scenario_config,
        args.scenario_name,
        args.episode_steps,
    )
    cold_samples, warm_samples = run_suite(
        config=config,
        script_path=Path(__file__),
        scenario_config=args.scenario_config,
        scenario_name=args.scenario_name,
        seed=args.seed,
        episode_steps=args.episode_steps,
        cold_runs=args.cold_runs,
        warm_runs=args.warm_runs,
    )
    current = SuiteSnapshot(
        cold=median_metrics(cold_samples),
        warm=median_metrics(warm_samples),
    )

    baseline = load_snapshot(args.baseline)
    thresholds = RegressionThresholds(
        max_slowdown_pct=args.max_slowdown_pct,
        max_throughput_drop_pct=args.max_throughput_drop_pct,
        min_seconds_delta=args.min_seconds_delta,
        min_throughput_delta=args.min_throughput_delta,
    )
    report = compare_snapshots(current, baseline, thresholds) if baseline is not None else None

    payload = {
        "schema_version": "1.0",
        "suite": "classic-interactions-cold-warm-v1",
        "scenario": scenario_label,
        "episode_steps": int(args.episode_steps),
        "seed": int(args.seed),
        "cold": {
            "runs": int(args.cold_runs),
            "samples": [sample.to_dict() for sample in cold_samples],
            "median": current.cold.to_dict(),
        },
        "warm": {
            "runs": int(args.warm_runs),
            "samples": [sample.to_dict() for sample in warm_samples],
            "median": current.warm.to_dict(),
        },
        "baseline_path": str(args.baseline),
        "baseline_loaded": baseline is not None,
        "thresholds": asdict(thresholds),
        "comparison": _report_to_dict(report),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    markdown = render_markdown_report(
        scenario_label=scenario_label,
        episode_steps=args.episode_steps,
        cold_runs=args.cold_runs,
        warm_runs=args.warm_runs,
        current=current,
        baseline=baseline,
        report=report,
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")

    logger.info("Cold/warm perf JSON: {}", args.output_json)
    logger.info("Cold/warm perf Markdown: {}", args.output_markdown)
    if report is not None:
        logger.info("Comparison status: {}", report.status.upper())
        for diag in report.diagnostics:
            logger.info("Diagnostic: {}", diag)
    else:
        logger.info("Comparison status: NO_BASELINE")

    if baseline is None and args.require_baseline:
        logger.error("Baseline missing or invalid: {}", args.baseline)
        return 2
    if report is not None and report.has_blocking_regression and args.fail_on_regression:
        return 1
    return 0


def _report_to_dict(report: RegressionReport | None) -> dict[str, Any]:
    """Serialize a regression report for JSON output.

    Args:
        report: Optional regression report.

    Returns:
        dict[str, Any]: JSON-friendly report payload.
    """
    if report is None:
        return {
            "status": "no-baseline",
            "failure_class": "none",
            "has_startup_regression": False,
            "has_steady_regression": False,
            "findings": [],
            "diagnostics": [],
        }
    return {
        "status": report.status,
        "failure_class": report.failure_class,
        "has_startup_regression": report.has_startup_regression,
        "has_steady_regression": report.has_steady_regression,
        "findings": [
            {
                "phase": finding.phase,
                "metric": finding.metric,
                "baseline": finding.baseline,
                "current": finding.current,
                "delta": finding.delta,
                "delta_pct": finding.delta_pct,
                "is_regression": finding.is_regression,
                "threshold_pct": finding.threshold_pct,
            }
            for finding in report.findings
        ],
        "diagnostics": list(report.diagnostics),
    }


def _build_diagnostics(findings: Sequence[RegressionFinding]) -> tuple[str, ...]:
    """Generate high-level diagnostics from metric findings.

    Args:
        findings: Flat metric findings across phases.

    Returns:
        tuple[str, ...]: Human-readable diagnostics.
    """
    regressions = [finding for finding in findings if finding.is_regression]
    if not regressions:
        return ("No meaningful regressions detected.",)

    startup = [f for f in regressions if f.metric in _STARTUP_METRICS]
    steady = [f for f in regressions if f.metric in _STEADY_METRICS]
    notes: list[str] = []
    if startup and not steady:
        notes.append("Regression localized to startup overhead (env creation / first step).")
    elif steady and not startup:
        notes.append("Regression localized to steady-state stepping throughput.")
    else:
        notes.append("Regression spans startup and steady-state behavior.")

    notes.append(f"Regressed metrics: {', '.join(f'{f.phase}.{f.metric}' for f in regressions)}")
    return tuple(notes)


def _load_scenario_config(
    scenario_config: Path,
    scenario_name: str,
    episode_steps: int,
) -> tuple[RobotSimulationConfig, str]:
    """Load and normalize a benchmark scenario config.

    Args:
        scenario_config: Scenario YAML path.
        scenario_name: Scenario name to load.
        episode_steps: Overrides max steps in the runtime config.

    Returns:
        tuple[RobotSimulationConfig, str]: Runtime config and resolved scenario label.
    """
    scenarios = load_scenarios(scenario_config)
    selected = None
    for scenario in scenarios:
        name = str(scenario.get("name") or scenario.get("scenario_id") or "").strip()
        if name == scenario_name:
            selected = scenario
            break
    if selected is None:
        msg = f"Scenario '{scenario_name}' not found in {scenario_config}"
        raise ValueError(msg)

    config = build_robot_config_from_scenario(selected, scenario_path=scenario_config)
    config.sim_config.sim_time_in_secs = (
        max(1, int(episode_steps)) * config.sim_config.time_per_step_in_secs
    )
    label = str(selected.get("name") or selected.get("scenario_id") or scenario_name)
    return config, label


def _measure_cold_subprocess(
    *,
    script_path: Path,
    scenario_config: Path,
    scenario_name: str,
    seed: int,
    episode_steps: int,
) -> PhaseMetrics:
    """Measure one cold sample in a fresh subprocess.

    Args:
        script_path: Path to this module file.
        scenario_config: Scenario YAML path.
        scenario_name: Scenario identifier.
        seed: Random seed for deterministic replay.
        episode_steps: Number of simulation steps.

    Returns:
        PhaseMetrics: Parsed cold-run sample.
    """
    with tempfile.TemporaryDirectory(prefix="robot_sf_perf_cold_") as tmpdir:
        json_out = Path(tmpdir) / "internal_cold.json"
        md_out = Path(tmpdir) / "internal_cold.md"
        command = [
            sys.executable,
            str(script_path),
            "--scenario-config",
            str(scenario_config),
            "--scenario-name",
            scenario_name,
            "--seed",
            str(seed),
            "--episode-steps",
            str(episode_steps),
            "--output-json",
            str(json_out),
            "--output-markdown",
            str(md_out),
            "--internal-measure-once",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=_COLD_SUBPROCESS_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired as exc:
            cmd = " ".join(command)
            msg = (
                "Cold subprocess measurement timed out after "
                f"{_COLD_SUBPROCESS_TIMEOUT_SEC}s: {cmd}"
            )
            raise RuntimeError(msg) from exc
        if completed.returncode != 0:
            msg = (
                "Cold subprocess measurement failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )
            raise RuntimeError(msg)
        for line in reversed(completed.stdout.splitlines()):
            raw = line.strip()
            if raw.startswith("{") and raw.endswith("}"):
                return PhaseMetrics.from_mapping(json.loads(raw))
        msg = f"Cold subprocess did not return JSON payload: {completed.stdout!r}"
        raise RuntimeError(msg)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
