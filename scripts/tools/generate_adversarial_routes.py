#!/usr/bin/env python3
"""Generate adversarial robot/pedestrian route overrides with Bayesian optimization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from robot_sf.common.logging import configure_logging
from robot_sf.nav.adversarial_route_generation import (
    AdversarialRouteGenerationConfig,
    optimize_route_set,
    write_route_override_artifact,
)
from robot_sf.planner import ClassicGlobalPlanner, ClassicPlannerConfig
from robot_sf.training.scenario_loader import (
    load_scenarios,
    resolve_map_definition,
    select_scenario,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/adversarial_routes/default.yaml"),
        help="Path to adversarial-route config YAML.",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        default=None,
        help="Override scenario id from config.",
    )
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at config root: {path}")
    return data


def _dict_section(data: dict[str, Any], key: str) -> dict[str, Any]:
    section = data.get(key, {})
    if not isinstance(section, dict):
        raise ValueError(f"'{key}' must be a mapping")
    return section


def _resolve_map_file(scenario_file: Path, scenario_entry: dict[str, Any]) -> str:
    map_file = scenario_entry.get("map_file")
    if not isinstance(map_file, str) or not map_file.strip():
        raise ValueError("Selected scenario has no map_file")
    candidate = Path(map_file)
    if not candidate.is_absolute():
        candidate = (scenario_file.parent / candidate).resolve()
    return str(candidate)


def main() -> int:
    """Run the adversarial route generation workflow from config.

    Returns:
        int: Process exit code (0 on success).
    """
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(verbose=False)

    config = _load_yaml(args.config)
    scenario_cfg = _dict_section(config, "scenario")
    planner_cfg = _dict_section(config, "planner")
    opt_cfg = _dict_section(config, "optimization")
    output_cfg = _dict_section(config, "output")

    config_path = args.config.resolve()
    config_dir = config_path.parent
    scenario_file_raw = str(scenario_cfg.get("scenario_file", "")).strip()
    if not scenario_file_raw:
        raise ValueError("scenario.scenario_file must be set")
    scenario_file = Path(scenario_file_raw)
    if not scenario_file.is_absolute():
        scenario_file = config_dir / scenario_file
    scenario_file = scenario_file.resolve()
    if not scenario_file.is_file():
        raise ValueError(f"scenario_file does not exist: {scenario_file}")
    scenario_id = args.scenario_id or str(scenario_cfg.get("scenario_id", "")).strip()
    if not scenario_id:
        raise ValueError("scenario.scenario_id must be set")

    scenarios = load_scenarios(scenario_file)
    selected = dict(select_scenario(scenarios, scenario_id))
    map_file = _resolve_map_file(scenario_file, selected)
    map_def = resolve_map_definition(map_file, scenario_path=scenario_file)
    if map_def is None:
        raise ValueError(f"Unable to load map definition: {map_file}")

    planner = ClassicGlobalPlanner(
        map_def=map_def,
        config=ClassicPlannerConfig(
            cells_per_meter=float(planner_cfg.get("cells_per_meter", 2.0)),
            inflate_radius_meters=float(planner_cfg.get("inflate_radius_meters", 0.0)),
            add_boundary_obstacles=bool(planner_cfg.get("add_boundary_obstacles", True)),
            algorithm=str(planner_cfg.get("algorithm", "theta_star_v2")),
        ),
    )

    generation_config = AdversarialRouteGenerationConfig(
        scenario_id=scenario_id,
        map_file=map_file,
        objective_mode=str(opt_cfg.get("objective_mode", "composite")),  # type: ignore[arg-type]
        trial_count=int(opt_cfg.get("trial_count", 20)),
        seed=int(opt_cfg.get("seed", 123)),
        robot_route_count=int(opt_cfg.get("robot_route_count", 1)),
        ped_route_count=int(opt_cfg.get("ped_route_count", 2)),
        allow_inflation_fallback=bool(opt_cfg.get("allow_inflation_fallback", False)),
        feasibility_filter=bool(opt_cfg.get("feasibility_filter", True)),
        top_k=int(opt_cfg.get("top_k", 5)),
        min_valid_trial_ratio=float(opt_cfg.get("min_valid_trial_ratio", 0.1)),
        near_miss_threshold_m=float(opt_cfg.get("near_miss_threshold_m", 1.5)),
        clearance_threshold_m=float(opt_cfg.get("clearance_threshold_m", 0.75)),
        failure_weight=float(opt_cfg.get("failure_weight", 0.45)),
        delay_weight=float(opt_cfg.get("delay_weight", 0.25)),
        inefficiency_weight=float(opt_cfg.get("inefficiency_weight", 0.15)),
        near_miss_weight=float(opt_cfg.get("near_miss_weight", 0.15)),
    )

    result = optimize_route_set(map_def, planner, generation_config)
    output_root = Path(str(output_cfg.get("root", "output/adversarial_routes")))
    artifacts = write_route_override_artifact(result, output_root=output_root)

    logger.info("Adversarial route generation completed.")
    logger.info("Route override artifact: {}", artifacts["artifact_path"])
    logger.info("JSON summary: {}", artifacts["json_summary_path"])
    logger.info("Markdown report: {}", artifacts["report_path"])
    logger.info("Trajectory overlay: {}", artifacts["overlay_plot_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
