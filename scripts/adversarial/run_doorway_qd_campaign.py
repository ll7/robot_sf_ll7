#!/usr/bin/env python3
"""Bounded doorway-family MAP-Elites QD campaign (issue #5308).

Usage:
    uv run python scripts/adversarial/run_doorway_qd_campaign.py \\
        --config configs/adversarial/doorway_qd_campaign_v1.yaml \\
        --output-dir output/adversarial/doorway_qd_v1 \\
        [--warm-start path/to/flip_report.json]

Populates a (min_pedestrian_distance, time_to_collision) behavior grid over
the classic_doorway scenario family using Random + CoordinateRefinement +
CMaMeEmitter into the MAP-Elites archive.  Produces an equal-budget comparison
report vs single-objective search.

This is capability plumbing only (design #1433).  The archive artifact is an
adversarial output path, not a camera-ready finding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from robot_sf.adversarial.cma_me import CMaMeEmitter
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    SearchConfig,
    SearchSpaceConfig,
)
from robot_sf.adversarial.qd import (
    GridSpec,
    QDArchive,
    QDSearchConfig,
    compare_qd_vs_single_objective,
    production_qd_evaluator,
    run_map_elites,
    write_qd_archive,
)
from robot_sf.adversarial.samplers import (
    CoordinateRefinementSampler,
    RandomCandidateSampler,
)
from robot_sf.adversarial.warm_start import (
    extract_warm_starts,
    load_flip_report,
)
from robot_sf.common.artifact_paths import get_artifact_root


class _CapturingEvaluator:
    """Wraps a QDEvaluator and records every evaluation for later comparison."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.evaluations: list[CandidateEvaluation] = []

    def __call__(self, qd_config: QDSearchConfig, candidate: Any) -> CandidateEvaluation:
        evaluation = self._inner(qd_config, candidate)
        self.evaluations.append(evaluation)
        return evaluation


def _load_campaign_config(path: Path) -> dict:
    """Load and validate a QD campaign YAML config."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Campaign config must be a mapping: {path}")
    schema = raw.get("schema_version")
    if schema != "adversarial-qd-campaign.v1":
        raise ValueError(
            f"Unsupported campaign schema {schema!r}; expected adversarial-qd-campaign.v1"
        )
    required_keys = {"campaign", "search_space", "production"}
    missing = required_keys - set(raw)
    if missing:
        raise ValueError(f"Campaign config missing required sections: {missing}")
    return raw


def build_grid(cfg: dict) -> GridSpec:
    """Build the behavior-grid spec from the campaign config."""
    grid_cfg = cfg["campaign"]["behavior_grid"]
    return GridSpec(
        x_min=grid_cfg["x_min"],
        x_max=grid_cfg["x_max"],
        y_min=grid_cfg["y_min"],
        y_max=grid_cfg["y_max"],
        bins=grid_cfg["bins"],
    )


def build_search_space(cfg: dict) -> SearchSpaceConfig:
    """Build the search space from the campaign config."""
    return SearchSpaceConfig.from_mapping(cfg["search_space"])


def build_emitters(
    search_space: SearchSpaceConfig,
    archive: QDArchive,
    *,
    cfg: dict,
    seed: int,
    warm_starts: tuple | None = None,
) -> list:
    """Build emitters listed in the campaign config."""
    warm_start_tuple = warm_starts or ()
    emitter_names = cfg["campaign"]["emitters"]
    emitters: list = []
    for name in emitter_names:
        if name == "random":
            emitters.append(
                RandomCandidateSampler(search_space, seed=seed, warm_start=warm_start_tuple)
            )
        elif name == "coordinate_refinement":
            emitters.append(
                CoordinateRefinementSampler(
                    search_space, seed=seed + 1, warm_start=warm_start_tuple
                )
            )
        elif name == "cma_me":
            cma_cfg = cfg["campaign"].get("cma_me", {})
            emitters.append(
                CMaMeEmitter(
                    search_space,
                    archive,
                    seed=seed + 2,
                    sigma_fraction=cma_cfg.get("sigma_fraction", 0.25),
                    popsize=cma_cfg.get("popsize"),
                )
            )
        else:
            raise ValueError(f"Unknown emitter: {name!r}")
    return emitters


def _resolve_artifact_path(output_dir: str | Path) -> Path:
    """Resolve the output directory under the artifact root."""
    path = Path(output_dir)
    if not path.is_absolute():
        artifact_root = get_artifact_root()
        path = artifact_root / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_warm_starts(
    warm_start_path: Path | None,
    search_space: SearchSpaceConfig,
) -> tuple:
    """Load warm-start candidates from a knife-edge flip report if available."""
    if warm_start_path is None or not warm_start_path.exists():
        return ()
    report = load_flip_report(warm_start_path)
    extraction = extract_warm_starts(report, search_space=search_space)
    return extraction.warm_starts


def _build_production_search_config(
    qd_config: QDSearchConfig,
    prod_cfg: dict,
    output_dir: Path,
    suffix: str,
) -> SearchConfig:
    """Build a SearchConfig from the QD config and production settings."""
    return qd_config.to_search_config(
        policy=prod_cfg["policy"],
        scenario_template=Path(prod_cfg["scenario_template"]),
        output_dir=output_dir / suffix,
        horizon=prod_cfg.get("horizon"),
        dt=prod_cfg.get("dt"),
        workers=prod_cfg.get("workers", 1),
        benchmark_profile=prod_cfg.get("benchmark_profile", "baseline-safe"),
    )


def run_campaign(
    config_path: Path,
    output_dir: Path,
    *,
    warm_start_path: Path | None = None,
    budget: int | None = None,
    seed: int | None = None,
    inject_evaluator: Any | None = None,
    inject_certifier: Any | None = None,
) -> dict[str, Any]:
    """Run the bounded doorway QD campaign and return the result summary.

    Args:
        config_path: Path to the YAML campaign configuration.
        output_dir: Output directory (resolved under artifact root if relative).
        warm_start_path: Optional path to a knife-edge flip report for warm starts.
        budget: Override the campaign budget from config.
        seed: Override the campaign seed from config.
        inject_evaluator: Optional injected production evaluator for tests.
        inject_certifier: Optional injected certifier for tests.

    Returns:
        dict with campaign summary keys.
    """
    cfg = _load_campaign_config(config_path)
    campaign_cfg = cfg["campaign"]
    prod_cfg = cfg["production"]
    search_space = build_search_space(cfg)
    grid = build_grid(cfg)
    output_dir = _resolve_artifact_path(output_dir)

    effective_budget = budget if budget is not None else campaign_cfg["budget"]
    effective_seed = seed if seed is not None else campaign_cfg["seed"]

    warm_starts = _load_warm_starts(warm_start_path, search_space)
    warm_start_count = len(warm_starts)

    archive = QDArchive(
        grid=grid,
        require_certification=campaign_cfg.get("require_certification", False),
    )

    emitters = build_emitters(
        search_space,
        archive,
        cfg=cfg,
        seed=effective_seed,
        warm_starts=warm_starts if warm_starts else None,
    )

    qd_config = QDSearchConfig(
        search_space=search_space,
        objective=campaign_cfg["objective"],
        grid=grid,
        budget=effective_budget,
        seed=effective_seed,
        require_certification=campaign_cfg.get("require_certification", False),
    )

    qd_search_config = _build_production_search_config(qd_config, prod_cfg, output_dir, "qd_run")

    qd_evaluator = production_qd_evaluator(
        qd_search_config,
        candidate_evaluator=inject_evaluator,
        certifier=inject_certifier,
    )

    result = run_map_elites(
        qd_config,
        evaluator=qd_evaluator,
        emitters=emitters,
        archive=archive,
    )

    archive_path = output_dir / "qd_archive.json"
    write_qd_archive(result, archive_path)

    so_config = QDSearchConfig(
        search_space=search_space,
        objective=campaign_cfg["objective"],
        grid=grid,
        budget=effective_budget,
        seed=effective_seed + 100,
        require_certification=campaign_cfg.get("require_certification", False),
    )
    so_search_config = _build_production_search_config(
        so_config, prod_cfg, output_dir, "single_objective"
    )

    so_evaluator = _CapturingEvaluator(
        production_qd_evaluator(
            so_search_config,
            candidate_evaluator=inject_evaluator,
            certifier=inject_certifier,
        )
    )

    from robot_sf.adversarial.qd import _default_emitters

    so_result = run_map_elites(
        so_config,
        evaluator=so_evaluator,
        emitters=_default_emitters(search_space, seed=effective_seed + 100),
    )

    comparison = compare_qd_vs_single_objective(
        qd_result=result,
        single_objective_evaluations=so_evaluator.evaluations,
        budget=effective_budget,
        grid=grid,
        require_certification=campaign_cfg.get("require_certification", False),
    )

    comparison_path = output_dir / "qd_vs_single_objective_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison.to_json(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "schema_version": "adversarial_qd_campaign_result.v1",
        "archive_path": archive_path.as_posix(),
        "comparison_path": comparison_path.as_posix(),
        "filled_cell_count": result.archive.filled_cell_count(),
        "distinct_failure_modes": sorted(result.archive.distinct_failure_modes()),
        "qd_score": round(result.archive.qd_score(), 6),
        "coverage_fraction": round(result.archive.coverage_fraction(), 6),
        "num_evaluated": result.num_evaluated,
        "num_admitted": result.num_admitted,
        "warm_starts_loaded": warm_start_count,
        "budget": effective_budget,
        "seed": effective_seed,
        "grid_bins": grid.bins,
        "grid_cell_count": grid.cell_count,
        "objective": campaign_cfg["objective"],
        "so_num_evaluated": so_result.num_evaluated,
        "so_filled_cells_retrospective": comparison.single_objective.filled_cells,
        "so_distinct_failure_modes_retrospective": comparison.single_objective.distinct_failure_modes,
    }

    summary_path = output_dir / "campaign_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the doorway QD campaign."""
    parser = argparse.ArgumentParser(
        description="Run a bounded doorway-family MAP-Elites QD campaign (issue #5308)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/adversarial/doorway_qd_campaign_v1.yaml"),
        help="Path to campaign YAML config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory (resolved under artifact root if relative)",
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help="Optional path to knife-edge flip report JSON for warm-start seeds",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Override campaign budget from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override campaign seed from config",
    )
    args = parser.parse_args(argv)

    summary = run_campaign(
        config_path=args.config,
        output_dir=args.output_dir,
        warm_start_path=args.warm_start,
        budget=args.budget,
        seed=args.seed,
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
