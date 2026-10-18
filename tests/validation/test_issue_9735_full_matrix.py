"""Seed-free refusal and custody controls for the VV-5 compactor.

Packets below are reconstructed controls, not the original raw receipts.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PRODUCERS = "404ad70b26f62199fb3a035a54e14ff6003291f0"
SOURCE = "03f96dbf4907622ea0ec7f865db4adf1630ede7a"
SCRIPT = "scripts/validation/build_issue_9735_full_matrix.py"
ROSTERS = {
    "first": (
        "scripts/validation/run_issue_9735_fault_injection.py",
        "robot_sf/sim/spawn_validation.py",
        "robot_sf/benchmark/spawn_validity.py",
        "robot_sf/analysis_workbench/release_row_anomalies.py",
        "maps/svg_maps/classic_head_on_corridor.svg",
    ),
    "geometry_radius": (
        "scripts/validation/run_issue_9735_geometry_radius_diagnostic.py",
        "robot_sf/benchmark/spawn_preflight.py",
        "robot_sf/benchmark/camera_ready/_route_clearance.py",
        "robot_sf/scenario_certification/feasibility_oracle.py",
        "robot_sf/benchmark/map_runner/map_runner.py",
        "robot_sf/planner/socnav_prediction.py",
        "robot_sf/analysis_workbench/release_row_anomalies.py",
        "tests/metamorphic/test_planner_unit_consistency.py",
        "configs/algos/prediction_planner_camera_ready.yaml",
        "configs/scenarios/single/francis2023_narrow_doorway.yaml",
        "maps/svg_maps/francis2023/francis2023_narrow_doorway.svg",
    ),
    "force": (
        "robot_sf/planner/socnav_social_force.py",
        "robot_sf/planner/socnav_base.py",
        "robot_sf/sim/simulator.py",
        "robot_sf/ped_npc/ped_robot_force.py",
        "robot_sf/benchmark/metrics.py",
    ),
    "remaining": (
        "configs/algos/hybrid_rule_v3_teb_like_rollout.yaml",
        "configs/algos/hybrid_rule_v4_clearance_braking.yaml",
        "robot_sf/planner/hybrid_rule_local_planner.py",
        "robot_sf/planner/socnav_social_force.py",
        "robot_sf/benchmark/map_runner/map_runner.py",
        "robot_sf/analysis_workbench/release_row_anomalies.py",
        "robot_sf/robot/differential_drive.py",
    ),
}


@pytest.fixture
def compactor():
    """Load the actual script, keeping absent-main behavior explicit."""
    assert (ROOT / SCRIPT).is_file(), "main has no ten-class compactor"
    spec = importlib.util.spec_from_file_location("vv5_matrix", ROOT / SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def packets():
    """Reconstruct only the compactor input fields from preserved diagnostic data."""
    artifact = json.loads(
        (ROOT / "docs/context/evidence/issue_9735_vv5_full_matrix_2026-09-28.json").read_text()
    )
    hashes = artifact["source_sha256"]
    result = {
        name: {"source_sha256": {p: hashes[p] for p in paths}} for name, paths in ROSTERS.items()
    }
    result["first"] = {"source": {"sha256": result["first"]["source_sha256"]}, "results": []}
    for name in ("force", "remaining"):
        result[name]["cases"] = []
        result[name]["head"] = artifact["packet_execution_heads"][name]
    result["force"]["fixture_script_sha256"] = hashes[
        "scripts/validation/run_issue_9735_force_diagnostic.py"
    ]
    result["remaining"]["script_sha256"] = hashes[
        "scripts/validation/run_issue_9735_braking_goal_diagnostic.py"
    ]
    result["geometry_radius"].update(
        cases={}, source_commit=artifact["packet_execution_heads"]["geometry_radius"]
    )
    for item in copy.deepcopy(artifact["matrix"]):
        name = item.pop("packet")
        if name == "first":
            item["control"] = item.pop("clean")
            result[name]["results"].append(item)
        elif name == "geometry_radius":
            fault = item["fault"]
            item["control"] = item.pop("clean")
            if fault == "infeasible_gap":
                for key in ("control", "mutant"):
                    case = item[key]
                    case["preflight"] = {"overall_status": case.pop("preflight_status")}
                    case["oracle"] = {
                        "status": case.pop("oracle_full_status"),
                        "geometric": {
                            "route_geometrically_feasible": case.pop("oracle_geometric_feasible")
                        },
                    }
                item["detected_by"] = {"route_clearance": True, "spawn_matrix_preflight": True}
                item["oracle_boundary"] = item["boundary"]
            else:
                item["detected_by"] = {"planner_unit_audit_rule": True}
            result[name]["cases"][fault] = item
        else:
            if item["fault"] == "per_cell_wall_force_sum":
                for key in ("clean", "mutant"):
                    item[key]["invariance_pass"] = item[key].pop("pass")
            result[name]["cases"].append(item)
    return result


def verify(compactor, packets, source=SOURCE, producer=PRODUCERS):
    """Exercise historical verification on both old and explicit-reference APIs."""
    kwargs = (
        {"producer_commit": producer}
        if "producer_commit" in inspect.signature(compactor._verified_sources).parameters
        else {}
    )
    return compactor._verified_sources(packets, source, **kwargs)


@pytest.mark.parametrize("packet", ROSTERS)
def test_missing_packet_source_roster_rejected(compactor, packets, packet):
    """A packet cannot borrow another packet's manifest to hide missing provenance."""
    sources = (
        packets[packet]["source"]["sha256"]
        if packet == "first"
        else packets[packet]["source_sha256"]
    )
    sources.clear()
    # Use working producer hashes so rejection cannot come from historical typing drift.
    for name, paths in ROSTERS.items():
        mapping = (
            packets[name]["source"]["sha256"] if name == "first" else packets[name]["source_sha256"]
        )
        for path in paths:
            if path.startswith("scripts/") and path in mapping:
                mapping[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    packets["force"]["fixture_script_sha256"] = hashlib.sha256(
        (ROOT / "scripts/validation/run_issue_9735_force_diagnostic.py").read_bytes()
    ).hexdigest()
    packets["remaining"]["script_sha256"] = hashlib.sha256(
        (ROOT / "scripts/validation/run_issue_9735_braking_goal_diagnostic.py").read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="required source roster"):
        compactor._verified_sources(packets, SOURCE)


def test_nonexistent_source_rejected_before_empty_manifests(compactor, packets):
    """An empty manifest cannot make an unresolvable source identity acceptable."""
    packets["first"]["source"]["sha256"].clear()
    for name in ("geometry_radius", "force", "remaining"):
        packets[name]["source_sha256"].clear()
    for name, field, filename in (
        ("force", "fixture_script_sha256", "force"),
        ("remaining", "script_sha256", "braking_goal"),
    ):
        packets[name][field] = hashlib.sha256(
            (ROOT / f"scripts/validation/run_issue_9735_{filename}_diagnostic.py").read_bytes()
        ).hexdigest()
    with pytest.raises(ValueError, match="resolvable immutable source commit"):
        compactor._verified_sources(packets, "a" * 40)


@pytest.mark.parametrize("flag", ["false", 1, None])
def test_non_boolean_detection_rejected(compactor, packets, flag):
    """JSON strings and numbers must not be promoted to successful detections."""
    packets["first"]["results"][0]["detected"] = flag
    with pytest.raises(ValueError, match="boolean"):
        compactor._gather_cases(packets)


@pytest.mark.parametrize("packet", ["first", "geometry_radius", "force", "remaining"])
def test_reversed_clean_mutant_rejected(compactor, packets, packet):
    """A true flag must agree with a clean-pass and mutant-reject result pair."""
    case = (
        packets[packet]["cases"]["infeasible_gap"]
        if packet == "geometry_radius"
        else packets[packet]["results" if packet == "first" else "cases"][0]
    )
    control = "control" if packet in ("first", "geometry_radius") else "clean"
    case[control], case["mutant"] = case["mutant"], case[control]
    with pytest.raises(ValueError, match="outcome"):
        compactor._gather_cases(packets)


def test_historical_producer_bindings_recompact(compactor, packets):
    """Historical producer bytes remain verifiable after typing/development updates."""
    verified = verify(compactor, packets)
    assert len(verified) == 26
    assert (
        verified["scripts/validation/run_issue_9735_braking_goal_diagnostic.py"]
        == "29e026803dc22e0dc7c5d1fd64f25869d19183a90bf1c021ef6168206a2a37bd"
    )
    cases = compactor._gather_cases(packets)
    assert len(cases) == 10
    assert sum(c["detected"] for c in cases.values()) == 8


def test_new_producers_pass_seed_holdout_gate():
    """Executable producer additions must obey the actual seed diff gate."""
    from scripts.validation.check_seed_holdout_diff import check_diff

    paths = [
        f"scripts/validation/run_issue_9735_{name}_diagnostic.py"
        for name in ("geometry_radius", "braking_goal")
    ]
    diff = "\n".join(
        f"diff --git a/{p} b/{p}\n+++ b/{p}\n@@ -0,0 +1,999 @@\n"
        + "\n".join("+" + line for line in (ROOT / p).read_text().splitlines())
        for p in paths
    )
    assert check_diff(diff, ROOT) == []


@pytest.mark.parametrize(
    "fault,field",
    [
        ("infeasible_gap", "route_clearance"),
        ("infeasible_gap", "spawn_matrix_preflight"),
        ("planner_only_radius_halved", "planner_unit_audit_rule"),
    ],
)
def test_non_boolean_geometry_flags_rejected(compactor, packets, fault, field):
    """Geometry detection flags also require JSON booleans."""
    packets["geometry_radius"]["cases"][fault]["detected_by"][field] = "false"
    with pytest.raises(ValueError, match="boolean"):
        compactor._gather_cases(packets)


def test_historical_cli_roundtrip(compactor, packets, tmp_path, monkeypatch):
    """The real CLI binds the explicit producer pin and preserves historical results."""
    import sys

    args = ["vv5", "--source-commit", SOURCE, "--producer-commit", PRODUCERS]
    for name, packet in packets.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(packet))
        args.extend(["--" + name.replace("_", "-"), str(path)])
    output = tmp_path / "matrix.json"
    args.extend(["--output", str(output)])
    monkeypatch.setattr(sys, "argv", args)
    compactor.main()
    result = json.loads(output.read_text())
    assert result["producer_commit"] == PRODUCERS
    assert result["sensitivity"] == {"detected": 8, "injected": 10, "fraction": 0.8}
    assert result["release_admitted"] is False
    assert len(result["source_sha256"]) == 26
