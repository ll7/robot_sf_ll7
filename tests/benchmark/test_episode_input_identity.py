"""Pure provenance checks for exact benchmark scenario input identities."""

from __future__ import annotations

import hashlib
from pathlib import Path

from robot_sf.benchmark.episode_input_identity import (
    capture_episode_input_identity,
    reconcile_consumed_map_identity,
    reconcile_consumed_route_identity,
)
from robot_sf.training import scenario_loader
from robot_sf.training.scenario_loader import build_robot_config_from_scenario


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


def test_replaced_map_bytes_cannot_reuse_path_cached_geometry(tmp_path: Path) -> None:
    """The parsed map digest follows the exact bytes consumed across replacements."""
    repository_root = Path(__file__).resolve().parents[2]
    classic_map = repository_root / "maps/svg_maps/classic_crossing.svg"
    narrow_map = repository_root / "maps/svg_maps/narrow_corridor.svg"
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("scenario fixture\n", encoding="utf-8")
    route_path = tmp_path / "route.yaml"
    route_path.write_text("route_payload: {}\n", encoding="utf-8")
    map_path = tmp_path / "map.svg"
    classic_bytes = classic_map.read_bytes()
    narrow_bytes = narrow_map.read_bytes()
    map_path.write_bytes(classic_bytes)
    scenario = {
        "name": "replace-map",
        "map_file": "map.svg",
        "route_overrides_file": "route.yaml",
    }

    scenario_loader._load_map_definition.cache_clear()
    try:
        initial_identity = capture_episode_input_identity(
            scenario, scenario_path=scenario_path, seed=17, run_id="run-before"
        )
        first_config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
        (first_map,) = first_config.map_pool.map_defs.values()
        assert initial_identity["status"] == "bound"
        assert first_map._consumed_map_sha256 == hashlib.sha256(classic_bytes).hexdigest()

        map_path.write_bytes(narrow_bytes)
        replacement_identity = capture_episode_input_identity(
            scenario, scenario_path=scenario_path, seed=17, run_id="run-after"
        )
        second_config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
        (second_map,) = second_config.map_pool.map_defs.values()

        assert replacement_identity["status"] == "bound"
        assert second_map is not first_map
        assert second_map.width != first_map.width
        assert second_map.obstacles != first_map.obstacles
        assert second_map._consumed_map_sha256 == hashlib.sha256(narrow_bytes).hexdigest()
        assert (
            reconcile_consumed_map_identity(
                replacement_identity,
                consumed_map_sha256=second_map._consumed_map_sha256,
            )["status"]
            == "bound"
        )

        # A replacement after parsing can restore the captured path bytes (an ABA
        # race); the identity must still become unavailable because the consumed
        # snapshot differs from the bytes present at the capture boundary.
        map_path.write_bytes(classic_bytes)
        restored_identity = capture_episode_input_identity(
            scenario, scenario_path=scenario_path, seed=17, run_id="run-restored"
        )
        raced_identity = reconcile_consumed_map_identity(
            restored_identity, consumed_map_sha256=second_map._consumed_map_sha256
        )
        assert raced_identity["status"] == "unavailable"
        assert "parsed_map_bytes_differ_from_captured_map_asset" in raced_identity["reason_codes"]
    finally:
        scenario_loader._load_map_definition.cache_clear()


def test_route_override_aba_replacement_parses_and_hashes_one_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    """A route replacement during config construction cannot change parsed snapshot bytes."""
    repository_root = Path(__file__).resolve().parents[2]
    scenario_path = repository_root / "configs/scenarios/classic_interactions.yaml"
    map_path = repository_root / "maps/svg_maps/classic_overtaking.svg"
    route_path = tmp_path / "route.yaml"
    route_a = (
        b"robot_routes:\n"
        b"  - spawn_id: 0\n"
        b"    goal_id: 0\n"
        b"    waypoints:\n"
        b"      - [4.0, 4.0]\n"
        b"      - [8.0, 8.0]\n"
    )
    route_b = (
        b"robot_routes:\n"
        b"  - spawn_id: 0\n"
        b"    goal_id: 0\n"
        b"    waypoints:\n"
        b"      - [10.0, 10.0]\n"
        b"      - [14.0, 14.0]\n"
    )
    route_path.write_bytes(route_a)
    scenario = {
        "name": "route-aba",
        "map_file": str(map_path),
        "route_overrides_file": str(route_path),
    }

    snapshot = scenario_loader.capture_route_override_snapshot(
        scenario, scenario_path=scenario_path
    )
    assert snapshot.source_bytes == route_a
    assert snapshot.sha256 == hashlib.sha256(route_a).hexdigest()

    # Reproduce the ABA schedule at config construction: the path holds B when
    # the loader would reopen it, then is restored to A while parsing returns.
    route_path.write_bytes(route_b)
    original_safe_load = scenario_loader.yaml.safe_load
    parsed_route_inputs: list[str] = []

    def restore_during_parse(stream: object) -> object:
        if isinstance(stream, str) and "robot_routes:" in stream:
            parsed_route_inputs.append(stream)
            route_path.write_bytes(route_a)
        return original_safe_load(stream)

    monkeypatch.setattr(scenario_loader.yaml, "safe_load", restore_during_parse)
    config = build_robot_config_from_scenario(
        scenario,
        scenario_path=scenario_path,
        route_override_snapshot=snapshot,
    )
    ((_map_name, map_definition),) = config.map_pool.map_defs.items()
    assert parsed_route_inputs == [route_a.decode("utf-8")]
    assert map_definition.robot_routes[0].waypoints == [(4.0, 4.0), (8.0, 8.0)]
    assert config._consumed_route_overrides_sha256 == hashlib.sha256(route_a).hexdigest()

    identity = capture_episode_input_identity(
        scenario,
        scenario_path=scenario_path,
        seed=17,
        run_id="run-route-aba",
        route_override_snapshot=snapshot,
    )
    assert identity["status"] == "bound"
    assert identity["route_overrides_sha256"] == hashlib.sha256(route_a).hexdigest()
    assert (
        reconcile_consumed_route_identity(
            identity,
            consumed_route_overrides_sha256=config._consumed_route_overrides_sha256,
        )["status"]
        == "bound"
    )

    mismatch = reconcile_consumed_route_identity(
        identity, consumed_route_overrides_sha256=hashlib.sha256(route_b).hexdigest()
    )
    assert mismatch["status"] == "unavailable"
    assert "parsed_route_bytes_differ_from_captured_route_asset" in mismatch["reason_codes"]
