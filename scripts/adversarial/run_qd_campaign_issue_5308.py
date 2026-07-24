#!/usr/bin/env python3
"""Run the bounded doorway-family MAP-Elites / CMA-ME QD campaign (issue #5308).

Wires the existing quality-diversity capability
(``robot_sf.adversarial.qd.run_map_elites`` + ``production_qd_evaluator`` +
``robot_sf.adversarial.cma_me.CMaMeEmitter``) to the real production adversarial
pipeline and runs a bounded (<=4h CPU) MAP-Elites search over the
``(distance_to_human_min, time_to_collision_min)`` behavior grid for the doorway
family.

Emits two archive artifacts under ``--output-dir``:

* ``archive.json`` (``adversarial_qd_archive.v1``) - the populated MAP-Elites grid
  with coverage, QD score, and distinct certified failure mechanisms;
* ``comparison.json`` - the equal-budget MAP-Elites vs single-objective diversity
  comparison (filled cells, coverage, distinct failure modes).

Capability-not-evidence boundary: these are archive artifact paths, not
camera-ready benchmark findings. See the issue #5308 contract.

Usage::

    uv run python scripts/adversarial/run_qd_campaign_issue_5308.py \\
        --config configs/adversarial/issue_5308_qd_doorway.yaml \\
        --output-dir output/issue_5308_qd_doorway

For a fast capability smoke without the simulator, pass ``--smoke-budget N`` to run
the same wiring against an injected evaluator (no benchmark runner).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.adversarial.cma_me import CMaMeEmitter
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    RangeConfig,
    SearchSpaceConfig,
)
from robot_sf.adversarial.qd import (
    QD_ARCHIVE_SCHEMA_VERSION,
    GridSpec,
    QDArchive,
    QDSearchConfig,
    QDSearchResult,
    compare_qd_vs_single_objective,
    production_qd_evaluator,
    run_map_elites,
    write_qd_archive,
)
from robot_sf.adversarial.samplers import (
    CoordinateRefinementSampler,
    RandomCandidateSampler,
)
from robot_sf.adversarial.warm_start import extract_warm_starts, load_flip_report

if TYPE_CHECKING:
    from collections.abc import Callable

CLAIM_BOUNDARY = (
    "capability_not_evidence: archive.json and comparison.json for issue #5308 are "
    "archive artifact paths produced by the MAP-Elites + CMA-ME + production evaluator "
    "capability wiring. They are not camera-ready benchmark findings, planner-performance "
    "evidence, or a validated QD-superiority claim without the durable execution artifacts "
    "and a held-out comparator."
)

# Hard wall-clock guard so a runaway simulator cannot exceed the issue #5308 budget.
MAX_WALL_CLOCK_SECONDS = 4 * 60 * 60
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/adversarial/issue_5308_qd_doorway.yaml"

logger = logging.getLogger("issue_5308_qd_campaign")


def _load_campaign_config(path: Path) -> dict[str, Any]:
    """Load the campaign YAML and validate its top-level structure."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Campaign config must be a mapping: {path}")
    for required in ("search_space", "grid", "scenario_template", "objective", "budget"):
        if required not in payload:
            raise ValueError(f"Campaign config missing required key {required!r}: {path}")
    return payload


def _build_search_space(payload: dict[str, Any]) -> SearchSpaceConfig:
    """Build the search space from the campaign `search_space` section."""
    section = payload.get("search_space", {})
    # Support both the flat dataclass-shape and the `variables`/`constraints` shape.
    if "variables" in section or "constraints" in section:
        return SearchSpaceConfig.from_mapping(section)
    variables: dict[str, Any] = {}
    for name in (
        "start_x",
        "start_y",
        "goal_x",
        "goal_y",
        "spawn_time_s",
        "pedestrian_speed_mps",
        "pedestrian_delay_s",
        "scenario_seed",
    ):
        if name not in section:
            raise ValueError(f"search_space.{name} is required")
        raw_value = section[name]
        if not isinstance(raw_value, dict):
            raise ValueError(f"search_space.{name} must be a mapping with min/max")
        variables[name] = RangeConfig.from_mapping(raw_value, name=name)
    constraints = section.get("constraints", {}) or {}
    return SearchSpaceConfig(
        start_x=variables["start_x"],
        start_y=variables["start_y"],
        goal_x=variables["goal_x"],
        goal_y=variables["goal_y"],
        spawn_time_s=variables["spawn_time_s"],
        pedestrian_speed_mps=variables["pedestrian_speed_mps"],
        pedestrian_delay_s=variables["pedestrian_delay_s"],
        scenario_seed=variables["scenario_seed"],
        min_start_goal_distance_m=float(constraints.get("min_start_goal_distance_m", 0.25)),
    )


def _build_grid(payload: dict[str, Any]) -> GridSpec:
    """Build the behavior grid from the campaign `grid` section."""
    grid = payload["grid"]
    return GridSpec(
        x_min=float(grid["x_min"]),
        x_max=float(grid["x_max"]),
        y_min=float(grid["y_min"]),
        y_max=float(grid["y_max"]),
        bins=int(grid.get("bins", 8)),
    )


def _write_scenario_template(payload: dict[str, Any], output_dir: Path) -> Path:
    """Materialize the campaign scenario template into a benchmark-runner YAML."""
    template_path = output_dir / "scenario_template.yaml"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(
        yaml.safe_dump(payload["scenario_template"], sort_keys=False), encoding="utf-8"
    )
    return template_path


def _load_warm_starts(
    payload: dict[str, Any], search_space: SearchSpaceConfig, config_dir: Path
) -> tuple[CandidateSpec, ...]:
    """Extract warm-start candidates from the knife-edge flip report.

    Returns an empty tuple when no flip report is configured so the campaign still
    runs with cold-start emitters.
    """
    warm = payload.get("warm_start") or {}
    report_name = warm.get("flip_report")
    if not report_name:
        return ()
    report_path = Path(report_name)
    if not report_path.is_absolute():
        report_path = config_dir / report_path
    if not report_path.exists():
        logger.warning("warm-start flip report not found, cold-starting: %s", report_path)
        return ()
    report = load_flip_report(report_path)
    extraction = extract_warm_starts(
        report,
        search_space=search_space,
        margin_threshold=float(warm.get("margin_threshold", 0.5)),
        source=report_path.as_posix(),
    )
    logger.info(
        "warm-start extraction: %d selected, %d near-boundary, %d rejected",
        extraction.num_selected,
        extraction.num_near_boundary,
        len(extraction.rejected),
    )
    return tuple(warm.candidate for warm in extraction.warm_starts)


def _build_emitters(
    payload: dict[str, Any],
    search_space: SearchSpaceConfig,
    archive: QDArchive,
    warm_starts: tuple[CandidateSpec, ...],
    *,
    seed: int,
) -> list[Any]:
    """Build the configured emitter mix with warm-start seeds."""
    names = payload.get("emitters", ["warm_start_random", "warm_start_coordinate_refinement"])
    # ``WarmStartCandidate`` is the sampler contract; wrap bare candidates so the
    # existing samplers replay them first.
    from robot_sf.adversarial.warm_start import WarmStartCandidate

    warm = tuple(
        WarmStartCandidate(
            candidate=candidate, scenario="doorway_blocker", planner="goal", outcome_margin=0.0
        )
        for candidate in warm_starts
    )
    emitters: list[Any] = []
    for offset, name in enumerate(names):
        if name in {"random", "warm_start_random"}:
            emitters.append(
                RandomCandidateSampler(search_space, seed=seed + offset, warm_start=warm)
            )
        elif name in {"coordinate_refinement", "warm_start_coordinate_refinement"}:
            emitters.append(
                CoordinateRefinementSampler(search_space, seed=seed + offset + 100, warm_start=warm)
            )
        elif name in {"cma_me"}:
            emitters.append(CMaMeEmitter(search_space, archive, seed=seed + offset + 200))
        else:
            raise ValueError(f"unknown emitter {name!r} in campaign config")
    if not emitters:
        emitters = [RandomCandidateSampler(search_space, seed=seed)]
    return emitters


def _smoke_evaluator_factory(*, seed: int) -> Callable[[Any, CandidateSpec], CandidateEvaluation]:
    """Return an injected evaluator for the fast capability smoke (no simulator).

    Derives a deterministic (distance, ttc) descriptor and objective from the
    candidate so distinct proposals land in distinct grid cells. Collisions and
    timeouts are synthesized so the archive admits certified failure mechanisms.
    """
    import tempfile

    from robot_sf.adversarial.attribution import attribution_from_episode_record
    from robot_sf.adversarial.certification import passed_status

    counter = {"n": 0}
    tmp_root = Path(tempfile.mkdtemp(prefix="qd_smoke_"))

    def _evaluate(_config: QDSearchConfig, candidate: CandidateSpec) -> CandidateEvaluation:
        index = counter["n"]
        counter["n"] += 1
        rng_local = (candidate.scenario_seed * 31 + index) % 1000 / 1000.0
        distance = 5.0 * rng_local
        critical = 5.0 * ((1.0 - rng_local) + 0.01)
        failure = ["collision", "timeout", "incomplete"][index % 3]
        outcome = {"route_complete": False}
        if failure == "collision":
            outcome["collision"] = True
            termination = "collision"
        elif failure == "timeout":
            outcome["timeout"] = True
            termination = "timeout"
        else:
            termination = "incomplete"
        record = {
            "status": "completed",
            "termination_reason": termination,
            "outcome": outcome,
            "metrics": {"distance_to_human_min": distance, "time_to_collision_min": critical},
        }
        bundle = tmp_root / f"candidate_{index:04d}"
        bundle.mkdir(parents=True, exist_ok=True)
        episode_path = bundle / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        attribution = attribution_from_episode_record(record)
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status("smoke adapter"),
            objective_value=float(index),
            failure_attribution=attribution,
            episode_record_path=episode_path,
            trajectory_csv_path=None,
            scenario_yaml_path=bundle / "scenario.yaml",
            bundle_path=bundle,
        )

    return _evaluate


def _run_single_objective_baseline(
    *,
    qd_config: QDSearchConfig,
    search_config: Any,
    evaluator: Callable[[Any, CandidateSpec], CandidateEvaluation],
    warm_starts: tuple[CandidateSpec, ...],
    budget: int,
    seed: int,
) -> list[CandidateEvaluation]:
    """Run an equal-budget single-objective search for the comparison row.

    Uses a RandomCandidateSampler (warm-started) that converges on the best objective,
    mirroring the single-objective baseline of ``compare_qd_vs_single_objective``.
    """
    from robot_sf.adversarial.objectives import get_objective
    from robot_sf.adversarial.warm_start import WarmStartCandidate

    warm = tuple(
        WarmStartCandidate(
            candidate=candidate, scenario="doorway_blocker", planner="goal", outcome_margin=0.0
        )
        for candidate in warm_starts
    )
    sampler = RandomCandidateSampler(qd_config.search_space, seed=seed + 999, warm_start=warm)
    objective_fn = get_objective(qd_config.objective)
    evaluations: list[CandidateEvaluation] = []
    for _ in range(budget):
        candidate = sampler.sample()
        evaluation = evaluator(search_config if search_config is not None else qd_config, candidate)
        if evaluation.objective_value is None:
            evaluation = evaluation.with_objective(objective_fn(evaluation))
        evaluations.append(evaluation)
    return evaluations


def run_campaign(
    config_path: Path,
    output_dir: Path,
    *,
    budget_override: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the bounded doorway QD campaign and emit archive + comparison artifacts.

    Returns the archive summary dict (also written to ``archive.json``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_campaign_config(config_path)
    budget = int(budget_override if budget_override is not None else payload["budget"])
    if budget < 1:
        raise ValueError("budget must be >= 1")
    seed = int(payload.get("seed", 0))

    search_space = _build_search_space(payload)
    grid = _build_grid(payload)
    template_path = _write_scenario_template(payload, output_dir)
    warm_starts = _load_warm_starts(payload, search_space, config_path.parent)

    qd_config = QDSearchConfig(
        search_space=search_space,
        objective=str(payload["objective"]),
        grid=grid,
        budget=budget,
        seed=seed,
        require_certification=bool(payload.get("require_certification", False)),
    )
    shared_archive = QDArchive(grid=grid, require_certification=qd_config.require_certification)
    emitters = _build_emitters(payload, search_space, shared_archive, warm_starts, seed=seed)

    if smoke:
        evaluator = _smoke_evaluator_factory(seed=seed)
        search_config: Any = None
    else:
        search_config = qd_config.to_search_config(
            policy=str(payload.get("policy", "goal")),
            scenario_template=template_path,
            output_dir=output_dir / "qd_candidates",
            horizon=int(payload.get("horizon", 60)) or None,
            dt=float(payload.get("dt", 0.1)) or None,
            workers=int(payload.get("workers", 1)),
        )
        evaluator = production_qd_evaluator(search_config)

    logger.info(
        "issue #5308 QD campaign: budget=%d grid=%dx%d smoke=%s emitters=%d warm_starts=%d",
        budget,
        grid.bins,
        grid.bins,
        smoke,
        len(emitters),
        len(warm_starts),
    )
    start = time.time()
    result = _run_with_budget_guard(
        qd_config, evaluator=evaluator, emitters=emitters, archive=shared_archive
    )
    elapsed = time.time() - start

    archive_path = write_qd_archive(result, output_dir / "archive.json")
    summary = _finalize_summary(result, elapsed=elapsed, smoke=smoke)
    archive_summary = _write_summary_and_comparison(
        _ComparisonContext(
            qd_config=qd_config,
            search_config=search_config,
            evaluator=evaluator,
            warm_starts=warm_starts,
            budget=budget,
            seed=seed,
        ),
        result=result,
        summary=summary,
        output_dir=output_dir,
    )
    _emit_run_manifest(
        output_dir=output_dir,
        config_path=config_path,
        budget=budget,
        smoke=smoke,
        elapsed=elapsed,
        archive_path=archive_path,
        archive_summary=archive_summary,
    )
    logger.info(
        "issue #5308 QD campaign done: filled_cells=%d distinct_modes=%s elapsed=%.1fs",
        result.archive.filled_cell_count(),
        sorted(result.archive.distinct_failure_modes()),
        elapsed,
    )
    return archive_summary


def _run_with_budget_guard(
    qd_config: QDSearchConfig,
    *,
    evaluator: Callable[[Any, CandidateSpec], CandidateEvaluation],
    emitters: list[Any],
    archive: QDArchive,
) -> QDSearchResult:
    """Run MAP-Elites, enforcing the issue #5308 4h wall-clock stop condition."""
    if MAX_WALL_CLOCK_SECONDS <= 0:
        return run_map_elites(qd_config, evaluator=evaluator, emitters=emitters, archive=archive)
    deadline = time.monotonic() + MAX_WALL_CLOCK_SECONDS

    class _DeadlineEvaluator:
        """Wrap the evaluator so the run fails closed past the wall-clock budget."""

        def __init__(self, inner: Callable[[Any, CandidateSpec], CandidateEvaluation]) -> None:
            self._inner = inner

        def __call__(self, config: QDSearchConfig, candidate: CandidateSpec) -> CandidateEvaluation:
            if time.monotonic() > deadline:
                raise TimeoutError("issue #5308 QD campaign exceeded the 4h wall-clock CPU budget")
            return self._inner(config, candidate)

    return run_map_elites(
        qd_config, evaluator=_DeadlineEvaluator(evaluator), emitters=emitters, archive=archive
    )


def _finalize_summary(result: QDSearchResult, *, elapsed: float, smoke: bool) -> dict[str, Any]:
    """Build the campaign summary provenance block."""
    return {
        "filled_cell_count": result.archive.filled_cell_count(),
        "coverage_fraction": round(result.archive.coverage_fraction(), 6),
        "qd_score": round(result.archive.qd_score(), 6),
        "distinct_failure_modes": sorted(result.archive.distinct_failure_modes()),
        "num_evaluated": result.num_evaluated,
        "num_admitted": result.num_admitted,
        "wall_clock_seconds": round(elapsed, 3),
        "smoke": bool(smoke),
        "claim_boundary": CLAIM_BOUNDARY,
    }


@dataclass(frozen=True)
class _ComparisonContext:
    """Bundle of inputs for the equal-budget comparison artifact (issue #5308)."""

    qd_config: QDSearchConfig
    search_config: Any
    evaluator: Callable[[Any, CandidateSpec], CandidateEvaluation]
    warm_starts: tuple[CandidateSpec, ...]
    budget: int
    seed: int


def _write_summary_and_comparison(
    ctx: _ComparisonContext,
    *,
    result: QDSearchResult,
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Write the campaign summary and the equal-budget comparison artifact."""
    summary_path = output_dir / "campaign_summary.json"
    summary_payload = {
        "schema_version": QD_ARCHIVE_SCHEMA_VERSION,
        "issue": 5308,
        "summary": summary,
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    baseline = _run_single_objective_baseline(
        qd_config=ctx.qd_config,
        search_config=ctx.search_config,
        evaluator=ctx.evaluator,
        warm_starts=ctx.warm_starts,
        budget=ctx.budget,
        seed=ctx.seed,
    )
    report = compare_qd_vs_single_objective(
        qd_result=result,
        single_objective_evaluations=baseline,
        budget=ctx.budget,
        grid=ctx.qd_config.grid,
        require_certification=ctx.qd_config.require_certification,
    )
    comparison_path = output_dir / "comparison.json"
    comparison_path.write_text(
        json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _emit_run_manifest(
    *,
    output_dir: Path,
    config_path: Path,
    budget: int,
    smoke: bool,
    elapsed: float,
    archive_path: Path,
    archive_summary: dict[str, Any],
) -> None:
    """Write a compact run manifest with provenance for the archive artifact."""
    import subprocess

    def _git_head() -> str:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    manifest = {
        "schema_version": "adversarial-qd-campaign-run.v1",
        "issue": 5308,
        "claim_boundary": CLAIM_BOUNDARY,
        "config": config_path.as_posix(),
        "budget": int(budget),
        "smoke": bool(smoke),
        "wall_clock_seconds": round(elapsed, 3),
        "git_head": _git_head(),
        "archive_artifact": archive_path.as_posix(),
        "archive_summary": archive_summary,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args and run the bounded QD campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Campaign config YAML path."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for emitted artifacts."
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Override the campaign budget (number of candidates).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the fast injected-evaluator capability smoke (no simulator).",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default INFO).")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    summary = run_campaign(
        args.config,
        args.output_dir,
        budget_override=args.budget,
        smoke=args.smoke,
    )
    print(
        "issue #5308 QD campaign: filled_cells="
        f"{summary['filled_cell_count']} distinct_modes={summary['distinct_failure_modes']}"
    )
    if summary["filled_cell_count"] == 0 or len(summary["distinct_failure_modes"]) < 2:
        # Surface the stop condition without raising so artifacts remain inspectable.
        print(
            "WARNING: issue #5308 stop condition not met "
            "(filled_cell_count > 0 and >= 2 distinct certified failure modes required).",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
