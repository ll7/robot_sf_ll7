"""Parity guards for diagnostic typing repairs, without running a contact campaign."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pysocialforce import Simulator

SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "scripts/validation/issue_10017_wall_contact"


def _definitions(script):
    """Load actual definitions, excluding the scripts' campaign startup side effects."""
    path = SCRIPT_ROOT / script
    tree = ast.parse(path.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))
    ]
    namespace = {"__name__": "wall_contact_test"}
    _execute(definitions, str(path), namespace)
    return tree, namespace


def _execute(nodes, filename, namespace):
    """Execute only trusted repository AST, never external campaign inputs."""
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)  # noqa: S102


@pytest.mark.parametrize("scale", [0.0, 1.0, 10.0])
def test_far_doorway_probe_stays_outside_finite_force_range(scale):
    """The 1.2 m clear doorway probe stays force-free with the minimal geometry provider."""
    _, namespace = _definitions("analyze_variants.py")
    assert namespace["doorway_braking_force"]("body_edge_exponential_v3", scale) == 0.0


@pytest.mark.parametrize("missing_width", [False, True])
def test_width_crossing_keeps_geometry_and_missing_attribute_error(tmp_path, missing_width):
    """The crossing result and float(None) TypeError survive required-field narrowing."""
    _, namespace = _definitions("analyze_variants.py")
    width = "" if missing_width else 'width="1"'
    (tmp_path / "map.svg").write_text(
        f'<svg><rect x="15" y="0" height="2" {width}/><rect x="15" y="4" height="2" {width}/></svg>'
    )
    (tmp_path / "scenario.yaml").write_text(
        "scenarios:\n  - name: doorway\n    map_file: map.svg\n"
    )
    directory = tmp_path / "measurements/v/0"
    directory.mkdir(parents=True)
    trajectory = directory / "trajectory.npz"
    np.savez(
        trajectory,
        positions=np.array([[[18.0, 3.0]], [[16.0, 3.0]], [[14.0, 3.0]]]),
        velocities=np.array([[[1.0, 0.0]], [[0.0, 0.0]], [[1.0, 0.0]]]),
    )
    namespace.update(
        root=tmp_path,
        plan={
            "tasks": [{"manifest": "scenario.yaml", "scenario": "doorway"}],
            "roots": {"v": str(tmp_path)},
        },
        indexed={
            ("v", 0): {
                "measurement": {
                    "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                    "physical_radius_m": 0.35,
                }
            }
        },
    )
    if missing_width:
        with pytest.raises(TypeError, match="not 'NoneType'"):
            namespace["width_crossing"]("v", 0)
    else:
        assert namespace["width_crossing"]("v", 0) == (True, 0.1)


def test_measurement_controls_keep_component_selection_and_timestep():
    """Dynamic timestep and name-selected native force factors keep their old values."""
    tree, namespace = _definitions("measure_pedestrians.py")
    simulation = Simulator(state=np.array([[2.0, 3.0, 0.1, 0.0, 6.0, 3.0, 0.5]]))
    timed = SimpleNamespace(time_step_s=0.1)
    untimed = SimpleNamespace()
    sim = SimpleNamespace(peds_behaviors=[timed, untimed], pysf_sim=simulation)
    namespace.update(sim=sim, dt=0.05, controls={"disable_social": True, "disable_groups": True})
    loops = [node for node in tree.body if isinstance(node, ast.For)]
    selected = [loops[0], next(node for node in loops if node.target.id == "component")]
    _execute(selected, "controls", namespace)
    assert timed.time_step_s == 0.05
    assert not hasattr(untimed, "time_step_s")
    for component in simulation.forces:
        name = type(component).__name__
        if name == "SocialForce" or name.startswith("Group"):
            assert component.config.factor == 0.0
        else:
            assert component.config.factor != 0.0


def test_trace_callback_stays_unbound_and_evaluates_each_component_once():
    """Installing the closure must not inject self or double-evaluate native forces."""
    tree, namespace = _definitions("measure_pedestrians.py")
    simulation = Simulator(state=np.array([[2.0, 3.0, 0.1, 0.0, 6.0, 3.0, 0.5]]))
    calls = []

    def component():
        calls.append(1)
        return np.array([[4.0, 6.0]])

    simulation.forces = [component]
    sim = SimpleNamespace(
        pysf_sim=simulation,
        pysf_state=SimpleNamespace(
            pysf_states=lambda: simulation.peds.state,
        ),
    )
    namespace.update(
        sim=sim,
        trace_enabled=True,
        law="body_edge_exponential_v3",
        wall_component=SimpleNamespace(config=simulation.config.obstacle_force_config),
        raw_edges=np.empty((0, 6)),
        component_names=[],
        force_states=[],
        force_components=[],
        segment_forces=[],
        closest_points=[],
        integration=[],
    )
    trace = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "trace_enabled"
    )
    _execute([trace], "trace", namespace)
    np.testing.assert_array_equal(simulation.compute_forces(), [[4.0, 6.0]])
    assert calls == [1]
    assert len(namespace["integration"]) == len(namespace["force_states"]) == 1
    np.testing.assert_array_equal(namespace["force_components"][0], [[[4.0, 6.0]]])
