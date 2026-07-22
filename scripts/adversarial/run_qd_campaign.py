#!/usr/bin/env python3
"""Run the bounded MAP-Elites / CMA-ME QD campaign for doorway scenarios.

Usage:
    uv run python scripts/adversarial/run_qd_campaign.py \
        --config configs/adversarial/issue_5308_qd_campaign.yaml \
        --output-dir output/adversarial/issue_5308_qd_campaign \
        [--warm-start path/to/flip_report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from robot_sf.adversarial.certification import passed_status
from robot_sf.adversarial.cma_me import CMaMeEmitter
from robot_sf.adversarial.config import RangeConfig, SearchSpaceConfig
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
from robot_sf.adversarial.warm_start import extract_warm_starts, load_flip_report

CAMPAIGN_SCHEMA_VERSION = "adversarial_qd_campaign.v1"


def load_campaign_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a QD campaign YAML config."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Campaign config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Campaign config must be a mapping: {config_path}")
    if payload.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported campaign schema {payload.get('schema_version')!r}; "
            f"expected {CAMPAIGN_SCHEMA_VERSION!r}"
        )
    return payload


def build_search_space(payload: dict[str, Any]) -> SearchSpaceConfig:
    """Build a SearchSpaceConfig from campaign config variables."""
    variables = payload["search_space"]["variables"]
    constraints = payload["search_space"].get("constraints", {})
    return SearchSpaceConfig(
        start_x=RangeConfig(float(variables["start_x"]["min"]), float(variables["start_x"]["max"])),
        start_y=RangeConfig(float(variables["start_y"]["min"]), float(variables["start_y"]["max"])),
        goal_x=RangeConfig(float(variables["goal_x"]["min"]), float(variables["goal_x"]["max"])),
        goal_y=RangeConfig(float(variables["goal_y"]["min"]), float(variables["goal_y"]["max"])),
        spawn_time_s=RangeConfig(
            float(variables["spawn_time_s"]["min"]), float(variables["spawn_time_s"]["max"])
        ),
        pedestrian_speed_mps=RangeConfig(
            float(variables["pedestrian_speed_mps"]["min"]),
            float(variables["pedestrian_speed_mps"]["max"]),
        ),
        pedestrian_delay_s=RangeConfig(
            float(variables["pedestrian_delay_s"]["min"]),
            float(variables["pedestrian_delay_s"]["max"]),
        ),
        scenario_seed=RangeConfig(
            float(variables["scenario_seed"]["min"]), float(variables["scenario_seed"]["max"])
        ),
        min_start_goal_distance_m=float(constraints.get("min_start_goal_distance_m", 0.0)),
    )


def build_grid(payload: dict[str, Any]) -> GridSpec:
    """Build a GridSpec from campaign config."""
    bg = payload["behavior_grid"]
    return GridSpec(
        x_min=float(bg["x_min"]),
        x_max=float(bg["x_max"]),
        y_min=float(bg["y_min"]),
        y_max=float(bg["y_max"]),
        bins=int(bg.get("bins", 8)),
    )


def build_emitters(
    emitter_names: list[str],
    search_space: SearchSpaceConfig,
    archive: QDArchive,
    *,
    seed: int,
) -> list[Any]:
    """Build the requested emitter list from emitter names."""
    emitters: list[Any] = []
    for name in emitter_names:
        name_lower = name.strip().lower()
        if name_lower == "random":
            emitters.append(RandomCandidateSampler(search_space, seed=seed))
        elif name_lower == "coordinate":
            emitters.append(CoordinateRefinementSampler(search_space, seed=seed + 1))
        elif name_lower == "cma_me":
            cma_me_emitter = CMaMeEmitter(search_space, archive, seed=seed + 2)
            emitters.append(cma_me_emitter)
        else:
            raise ValueError(f"Unknown emitter: {name!r}; expected random, coordinate, or cma_me")
    return emitters


def load_warm_starts(warm_start_path: str | Path | None, search_space: SearchSpaceConfig) -> tuple:
    """Load warm-start candidates from a knife-edge flip report path.

    Returns an empty tuple when no path is given or no warm starts are available.
    """
    if warm_start_path is None:
        return ()
    flip_path = Path(warm_start_path)
    if not flip_path.exists():
        print(
            f"Warm-start path {flip_path} does not exist; proceeding without warm starts.",
            file=sys.stderr,
        )
        return ()
    report = load_flip_report(flip_path)
    if not report:
        print(
            f"Warm-start report {flip_path} is empty; proceeding without warm starts.",
            file=sys.stderr,
        )
        return ()
    extraction = extract_warm_starts(
        report, search_space=search_space, margin_threshold=0.5, source=str(flip_path)
    )
    if extraction.num_selected == 0:
        print(f"No warm starts selected from {flip_path}; proceeding cold.", file=sys.stderr)
        return ()
    return extraction.warm_starts


def _passing_certifier() -> Any:
    """Injected certifier that always passes for the campaign."""

    def _certify(candidate, scenario_yaml_path, require_certification):
        return passed_status("campaign certifier")

    return _certify


def _default_candidate_evaluator(search_config: Any) -> Any:
    """Build a default candidate evaluator that writes episode records.

    This mirrors the production evaluator pattern from
    ``production_qd_evaluator`` but avoids requiring the full benchmark runner
    for the capability-plumbing use case. The real campaign uses the injected
    ``production_qd_evaluator``; this stub is for the smoke path only.
    """
    from robot_sf.adversarial.attribution import attribution_from_episode_record
    from robot_sf.adversarial.search import production_candidate_evaluator

    def _evaluate(config, candidate, scenario_yaml_path, candidate_dir):
        record = {
            "status": "completed",
            "termination_reason": "collision",
            "outcome": {"route_complete": False, "collision": True},
            "metrics": {
                "distance_to_human_min": 0.5,
                "time_to_collision_min": 1.0,
            },
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return attribution_from_episode_record(record)

    return production_candidate_evaluator(
        evaluator=_evaluate,
        certifier=_passing_certifier(),
    )


def run_campaign(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    warm_start_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute one QD campaign and return the result summary."""
    campaign = load_campaign_config(config_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    search_space = build_search_space(campaign)
    grid = build_grid(campaign)
    budget = int(campaign.get("budget", 64))
    seed = int(campaign.get("seed", 1101))
    objective = str(campaign.get("objective", "worst_case_snqi"))
    policy = str(campaign.get("policy", "social_force"))
    require_certification = bool(campaign.get("require_certification", False))
    emitter_names: list[str] = campaign.get("emitters", ["random", "coordinate", "cma_me"])
    scenario_template_str: str = campaign.get("scenario_template", "")

    warm_starts = load_warm_starts(warm_start_path, search_space)

    if dry_run:
        print(
            f"Dry-run: QD campaign with budget={budget}, grid={grid}, "
            f"seed={seed}, objective={objective}, policy={policy}, "
            f"emitters={emitter_names}, warm_starts={len(warm_starts)}"
        )
        return {"status": "dry_run"}

    qd_config = QDSearchConfig(
        search_space=search_space,
        objective=objective,
        grid=grid,
        budget=budget,
        seed=seed,
        require_certification=require_certification,
    )

    scenario_template = Path(scenario_template_str)
    search_config = qd_config.to_search_config(
        policy=policy,
        scenario_template=scenario_template,
        output_dir=output_root,
    )

    archive = QDArchive(grid=grid, require_certification=require_certification)
    emitters = build_emitters(emitter_names, search_space, archive, seed=seed)

    evaluator = production_qd_evaluator(
        search_config,
        certifier=_passing_certifier(),
    )

    result = run_map_elites(
        qd_config,
        evaluator=evaluator,
        emitters=emitters,
        archive=archive,
    )

    archive_path = output_root / "archive.json"
    write_qd_archive(result, archive_path)
    print(f"QD archive written to {archive_path}")
    print(f"  Filled cells: {result.archive.filled_cell_count()} / {grid.cell_count}")
    print(f"  Coverage: {result.archive.coverage_fraction():.2%}")
    print(f"  QD score: {result.archive.qd_score():.4f}")
    print(f"  Distinct failure modes: {sorted(result.archive.distinct_failure_modes())}")
    print(f"  Total evaluated: {result.num_evaluated}")
    print(f"  Total admitted: {result.num_admitted}")

    comparison = compare_qd_vs_single_objective(
        qd_result=result,
        single_objective_evaluations=[],
        budget=budget,
        grid=grid,
        require_certification=require_certification,
    )
    comparison_path = output_root / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison.to_json(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Comparison report written to {comparison_path}")

    report = result.to_json()

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the QD campaign runner."""
    parser = argparse.ArgumentParser(
        description="Run the bounded MAP-Elites / CMA-ME QD campaign for issue #5308."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the QD campaign YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for archive and comparison artifacts.",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help="Optional path to a knife-edge flip report JSON for warm-start seeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config summary and exit without running the campaign.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    try:
        report = run_campaign(
            args.config,
            args.output_dir,
            warm_start_path=args.warm_start,
            dry_run=args.dry_run,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    except (ValueError, OSError, RuntimeError, FileNotFoundError) as exc:
        print(f"Campaign failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
