"""Resolve successor campaign identity inside its detached source checkout.

This file is launched with ``python -I``. The checkout path is installed before
any project import, so production helpers come from the named source commit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch


def _assert_pinned_modules(checkout: Path) -> None:
    for name in (
        "robot_sf.benchmark.camera_ready._util",
        "robot_sf.benchmark.camera_ready._config",
        "robot_sf.benchmark.camera_ready._preflight",
        "robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution",
        "robot_sf.benchmark.utils",
        "pysocialforce",
    ):
        module = sys.modules[name]
        if not Path(module.__file__).resolve().is_relative_to(checkout):
            raise ValueError(f"runtime module {name} did not load from pinned successor source")


def main() -> None:  # noqa: C901, PLR0912, PLR0915 - pinned resolution stays together
    checkout = Path.cwd().resolve()
    sys.path.insert(0, str(checkout))
    sys.path.insert(1, str(checkout / "fast-pysf"))
    for package in ("robot_sf", "fast-pysf/pysocialforce"):
        if not (checkout / package / "__init__.py").is_file():
            raise ValueError(f"pinned successor source lacks runtime package {package}")

    import pysocialforce  # noqa: F401 - assert its origin with the project modules below

    from robot_sf.benchmark.camera_ready import _util
    from robot_sf.benchmark.camera_ready._config import (
        _load_campaign_scenarios,
        load_campaign_config,
    )
    from robot_sf.benchmark.camera_ready._preflight import _scenario_matrix_hash
    from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (
        _apply_planner_selector_v2_context,
        _apply_scenario_uncertainty_envelope_config,
        _parse_algo_config,
        _resolve_config_path,
        _resolve_policy_search_candidate_runtime,
    )
    from robot_sf.benchmark.utils import _config_hash

    _assert_pinned_modules(checkout)

    request = json.load(sys.stdin)
    with patch.object(_util, "get_repository_root", return_value=checkout):
        cfg = load_campaign_config(checkout / request["config_path"], repository_root=checkout)
        for path in (cfg.source_config_path, cfg.scenario_matrix_path):
            if path is None or not path.resolve().is_relative_to(checkout):
                raise ValueError("successor config source escapes pinned checkout")
        scenarios = _load_campaign_scenarios(cfg, repository_root=checkout)
        config_hash = _config_hash(_util._config_hash_payload(cfg))
        scenario_hash = _scenario_matrix_hash(scenarios)

    scenario_by_id = {}
    for scenario in scenarios:
        identity = scenario.get("name") or scenario.get("scenario_id") or scenario.get("id")
        if not isinstance(identity, str) or not identity or identity in scenario_by_id:
            raise ValueError(f"invalid or duplicate pinned scenario identity: {identity!r}")
        scenario_by_id[identity] = _util._jsonable_repo_relative(scenario)

    planners = {}
    for planner in cfg.planners:
        path = planner.algo_config_path
        if path is not None and not path.resolve().is_relative_to(checkout):
            raise ValueError(f"successor planner config escapes pinned checkout: {planner.key}")
        raw = _parse_algo_config(str(path)) if path else {}
        if not isinstance(raw, dict):
            raise ValueError(f"successor planner config must be a mapping: {planner.key}")
        if path is not None:
            references = [raw.get("base_config_path")]
            overrides = raw.get("scenario_algo_overrides")
            if isinstance(overrides, dict):
                references.extend(
                    entry.get("base_config_path")
                    for entry in overrides.values()
                    if isinstance(entry, dict)
                )
            for reference in references:
                if reference is None:
                    continue
                resolved = _resolve_config_path(path.parent, reference)
                if (
                    resolved is None
                    or not resolved.is_file()
                    or not resolved.is_relative_to(checkout)
                ):
                    raise ValueError(
                        f"successor planner referenced config escapes pinned checkout: {planner.key}"
                    )
        planners[planner.key] = {
            "algo": planner.algo,
            "config": raw,
            "path": path.relative_to(checkout).as_posix() if path else None,
            "absolute_path": str(path) if path else None,
        }

    runtime_rows = []
    for item in request["rows"]:
        slot = item["slot"]
        planner = planners.get(slot[0])
        if planner is None:
            raise ValueError(f"0.0.8 row planner is absent from verified successor config: {slot}")
        matrix_scenario = scenario_by_id.get(slot[2])
        if matrix_scenario is None:
            raise ValueError(f"0.0.8 row scenario is absent from pinned matrix: {slot}")
        seeds = matrix_scenario.get("seeds")
        if isinstance(seeds, list) and slot[3] not in seeds:
            raise ValueError(f"0.0.8 row seed is absent from pinned scenario: {slot}")
        scenario = item["scenario_params"]
        if not isinstance(scenario, dict):
            raise ValueError(f"0.0.8 row lacks scenario provenance at {slot}")
        algo, effective = _resolve_policy_search_candidate_runtime(
            default_algo=planner["algo"],
            algo_config_path=planner["absolute_path"],
            scenario=matrix_scenario,
            algo_config=planner["config"],
        )
        effective = _apply_planner_selector_v2_context(
            algo, effective, scenario=matrix_scenario, seed=slot[3]
        )
        effective = _apply_scenario_uncertainty_envelope_config(algo, effective, matrix_scenario)
        runtime_rows.append(
            {
                "slot": slot,
                "algo": algo,
                "config": effective,
                "config_hash": _config_hash(effective),
                "scenario_config_hash": _config_hash(scenario),
                "scenario": matrix_scenario,
                "path": planner["path"],
            }
        )
    json.dump(
        {"config_hash": config_hash, "scenario_hash": scenario_hash, "rows": runtime_rows},
        sys.stdout,
    )


if __name__ == "__main__":
    main()
