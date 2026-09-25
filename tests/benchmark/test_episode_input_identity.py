"""Pure provenance checks for exact benchmark scenario input identities."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from robot_sf.benchmark.episode_input_identity import capture_episode_input_identity

if TYPE_CHECKING:
    from pathlib import Path


def test_episode_input_identity_binds_scenario_route_and_map_bytes(tmp_path: Path) -> None:
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("scenario fixture\n", encoding="utf-8")
    route_path = tmp_path / "route.yaml"
    route_path.write_text("route: [1, 2]\n", encoding="utf-8")
    map_path = tmp_path / "map.svg"
    map_path.write_text("<svg id='map-a'/>", encoding="utf-8")
    scenario = {
        "name": "case-a",
        "map_file": "map.svg",
        "route_overrides_file": "route.yaml",
        "seeds": [17],
    }

    identity = capture_episode_input_identity(
        scenario, scenario_path=scenario_path, seed=17, run_id="run-a"
    )

    assert identity["status"] == "bound"
    assert identity["run_id"] == "run-a"
    assert identity["scenario_semantic_sha256"]
    assert identity["route_overrides_sha256"] == hashlib.sha256(route_path.read_bytes()).hexdigest()
    assert identity["map_assets"] == [
        {"role": "map", "sha256": hashlib.sha256(map_path.read_bytes()).hexdigest()}
    ]
    assert identity["reason_codes"] == []

    route_path.write_text("route: [1, 3]\n", encoding="utf-8")
    updated = capture_episode_input_identity(
        scenario, scenario_path=scenario_path, seed=17, run_id="run-b"
    )
    assert updated["status"] == "bound"
    assert updated["route_overrides_sha256"] != identity["route_overrides_sha256"]
    assert updated["scenario_semantic_sha256"] == identity["scenario_semantic_sha256"]

    changed_scenario = {
        **scenario,
        "simulation_config": {"max_episode_steps": 40},
    }
    changed_semantics = capture_episode_input_identity(
        changed_scenario, scenario_path=scenario_path, seed=17, run_id="run-c"
    )
    assert changed_semantics["scenario_semantic_sha256"] != identity["scenario_semantic_sha256"]

    map_path.write_text("<svg id='map-a'/><!-- changed -->", encoding="utf-8")
    changed_map = capture_episode_input_identity(
        scenario, scenario_path=scenario_path, seed=17, run_id="run-d"
    )
    assert changed_map["map_assets"] != identity["map_assets"]
