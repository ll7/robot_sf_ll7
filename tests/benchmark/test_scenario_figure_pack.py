"""Composition, export, and refusal tests using small synthetic recorded traces.

No fixture here is benchmark evidence or an author-approved source package.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import matplotlib
import numpy as np
import pytest

from robot_sf.benchmark.figures import export
from robot_sf.benchmark.figures import scenario_pack as pack


@pytest.fixture
def case():
    """Build a synthetic trace with known coordinates, radii and applied controls."""
    steps = []
    for index in range(9):
        steps.append(
            {
                "time_s": index * 0.5,
                "robot": {"actor_id": "robot", "position": [index * 0.4, 0.0], "radius_m": 0.3},
                "pedestrians": [
                    {
                        "actor_id": "pedestrian-7",
                        "position": [1.6, 1.5 - index * 0.25],
                        "radius_m": 0.25,
                    }
                ],
                "controls": {"applied": {"linear_m_s": 0.8, "turn_rate_rad_s": 0.0}},
            }
        )
    return {
        "case_id": "synthetic-crossing",
        "scenario_id": "synthetic-crossing",
        "planner": "fixture",
        "seed": 7,
        "role": "diagnostic-fixture",
        "trace": {
            "schema_version": "analysis-trace.v1",
            "coordinate_frame": "world",
            "units": {"position": "m", "time": "s"},
            "steps": steps,
            "events": [],
        },
    }


@pytest.fixture
def source(tmp_path, case):
    """Write a checksum-consistent fixture, with no source trust or admission."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "proposal.json").write_text(json.dumps({"portfolio": [case]}), encoding="utf-8")
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    checksums = [
        hashlib.sha256(p.read_bytes()).hexdigest() + "  " + p.name for p in sorted(root.iterdir())
    ]
    (root / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    return root


def diagnostic(**kwargs):
    """Keep tests small without weakening the default admitted mode."""
    return pack.PackConfig(mode="diagnostic", formats=("svg",), views=("trajectory",), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "automatic"},
        {"size": "tiny"},
        {"max_cases": True},
        {"max_cases": 0},
        {"max_frames": 1_000_001},
        {"max_actors": -1},
        {"formats": ()},
        {"formats": ("svg", "svg")},
        {"formats": ("png", "html")},
        {"views": ("unknown",)},
    ],
)
def test_configuration_refuses_invalid_requests(kwargs):
    with pytest.raises(ValueError):
        pack.PackConfig(**kwargs)


def test_selection_retains_order_and_accounts_for_every_omission(case):
    cases = [{**case, "case_id": name} for name in ("c", "a", "b")]
    selected, omitted = pack.select_cases({"portfolio": cases}, diagnostic(max_cases=1), ("b", "a"))
    assert [item["case_id"] for item in selected] == ["a"]
    assert omitted == [
        {"case_id": "c", "reason": "not_requested"},
        {"case_id": "b", "reason": "presentation_budget"},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        pack.select_cases({"portfolio": [case, case]}, diagnostic())
    with pytest.raises(ValueError, match="absent"):
        pack.select_cases({"portfolio": cases}, diagnostic(), ("missing",))


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), -1, 0])
def test_bad_time_refused_without_compacting_frames(case, value):
    case["trace"]["steps"][1]["time_s"] = value
    with pytest.raises(ValueError, match="time"):
        pack.prepare_case(case, diagnostic())


def test_actor_disappearance_is_a_gap_not_a_join(case):
    case["trace"]["steps"][3]["pedestrians"] = []
    result = pack.prepare_case(case, diagnostic())
    assert len(result["actors"]["pedestrian-7"]) == 9
    assert all(np.isnan(result["actors"]["pedestrian-7"][3]))
    assert np.isnan(result["series"]["clearance"][3])
    assert result["series"]["clearance"][0] == pytest.approx(np.hypot(1.6, 1.5) - 0.55)


def test_partial_radius_coverage_cannot_be_a_complete_minimum(case):
    step = case["trace"]["steps"][4]
    step["pedestrians"].append({"actor_id": "unknown-radius", "position": [1.6, 0.1]})
    result = pack.prepare_case(case, diagnostic())
    assert np.isnan(result["series"]["clearance"][4])
    assert result["critical_index"] != 4


@pytest.mark.parametrize(
    "field,value", [("coordinate_frame", "ego"), ("units", {}), ("schema_version", "legacy")]
)
def test_explicit_trace_contract_required(case, field, value):
    case["trace"][field] = value
    with pytest.raises(ValueError):
        pack.prepare_case(case, diagnostic())


def test_duplicate_actor_and_resource_budgets_refused(case):
    with pytest.raises(ValueError, match="max_frames"):
        pack.prepare_case(case, diagnostic(max_frames=2))
    case["trace"]["steps"][0]["pedestrians"] *= 2
    with pytest.raises(ValueError, match="unique stable"):
        pack.prepare_case(case, diagnostic())


@pytest.mark.parametrize(
    "controls", [False, [], {"applied": []}, {"applied": {"linear_m_s": True}}]
)
def test_malformed_controls_are_not_missing_controls(case, controls):
    case["trace"]["steps"][0]["controls"] = controls
    with pytest.raises(ValueError, match="control"):
        pack.prepare_case(case, diagnostic())


def test_event_snapshot_uses_recorded_frame_without_interpolation(case):
    case["trace"]["events"] = [{"time_s": 0.6}]
    result = pack.prepare_case(case, diagnostic())
    assert result["critical_index"] == 1
    assert result["event_times"] == [0.6]
    assert "nearest recorded frame" in result["snapshot_reason"]
    case["trace"]["events"] = [{"time_s": 20.0}]
    with pytest.raises(ValueError, match="outside"):
        pack.prepare_case(case, diagnostic())


@pytest.mark.parametrize("view", pack.VIEWS)
def test_every_view_is_single_axis_and_visibly_diagnostic(case, view):
    figure, receipt = pack.render_view(pack.prepare_case(case, diagnostic()), view, diagnostic())
    figure.canvas.draw()
    assert len(figure.axes) == 1
    assert "DIAGNOSTIC ONLY" in figure.axes[0].get_title()
    assert receipt["view"] == view
    figure.clear()


def test_missing_controls_display_unavailable_not_zero(case):
    for step in case["trace"]["steps"]:
        step.pop("controls")
    prepared = pack.prepare_case(case, diagnostic())
    figure, receipt = pack.render_view(prepared, "speed", diagnostic())
    assert receipt["status"] == "unavailable"
    assert receipt["finite_samples"] == 0
    assert not figure.axes[0].lines
    assert any("Not zero" in text.get_text() for text in figure.axes[0].texts)
    figure.clear()


def test_export_saves_supplied_figure_not_pyplot_current(tmp_path):
    import matplotlib.pyplot as plt

    with matplotlib.rc_context({"svg.fonttype": "none", "text.usetex": False}):
        requested = plt.figure()
        requested.text(0.5, 0.5, "REQUESTED_FIGURE")
        current = plt.figure()
        current.text(0.5, 0.5, "WRONG_FIGURE")
        try:
            export.save_publication_figure(
                requested, tmp_path / "figure", formats=("svg",), provenance={}
            )
            result = (tmp_path / "figure.svg").read_text(encoding="utf-8")
            assert "REQUESTED_FIGURE" in result
            assert "WRONG_FIGURE" not in result
        finally:
            plt.close(requested)
            plt.close(current)


def test_invalid_format_cannot_leave_partial_outputs(tmp_path):
    from matplotlib.figure import Figure

    with pytest.raises(ValueError):
        export.save_publication_figure(
            Figure(), tmp_path / "new" / "figure", formats=("svg", "bad")
        )
    assert not (tmp_path / "new").exists()


def test_bundle_roundtrip_inventory_and_no_raw_trace(source, tmp_path):
    output = tmp_path / "pack"
    config = replace(diagnostic(), views=pack.VIEWS, formats=("pdf", "svg", "png"))
    result = pack.build_pack(source, output, config=config)
    assert pack.PackConfig.from_file(output / "config.json") == config
    assert len(result["cases"][0]["views"]) == 5
    assert result["selection"]["selected_count"] == 1
    assert '"steps"' not in (output / "manifest.json").read_text()
    for artifact in result["artifacts"]:
        assert pack._sha(output / artifact["path"]) == artifact["sha256"]
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == {
        artifact["path"] for artifact in result["artifacts"]
    } | {"manifest.json"}
    assert not (output / ".INCOMPLETE").exists()


def test_admission_gate_default_and_integrity_in_both_modes(source, tmp_path, monkeypatch):
    called = []
    owner = pack._owner()

    def refuse(_path):
        called.append("admission")
        raise ValueError("source approval unavailable")

    monkeypatch.setattr(owner, "_verify_publication_gate", refuse)
    with pytest.raises(ValueError, match="approval"):
        pack.build_pack(source, tmp_path / "admitted")
    assert called == ["admission"]
    pack.build_pack(source, tmp_path / "diagnostic", config=diagnostic())
    assert called == ["admission"]
    (source / "proposal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        pack.build_pack(source, tmp_path / "tampered", config=diagnostic())
    assert not (tmp_path / "tampered").exists()


@pytest.mark.parametrize("failure", ["export", "source_change", "output_race"])
def test_failed_export_preserves_source_or_other_writer_and_releases_lock(
    source, tmp_path, monkeypatch, failure
):
    output = tmp_path / "pack"
    original = export.save_publication_figure
    initial_rc = dict(matplotlib.rcParams)

    def interrupted(*args, **kwargs):
        if failure == "export":
            raise OSError("synthetic export failure")
        result = original(*args, **kwargs)
        if failure == "source_change":
            (source / "manifest.json").write_text('{"changed": true}', encoding="utf-8")
        else:
            output.mkdir()
            (output / "other-writer").write_text("preserve", encoding="utf-8")
        return result

    monkeypatch.setattr(export, "save_publication_figure", interrupted)
    with pytest.raises((ValueError, OSError)):
        pack.build_pack(source, output, config=diagnostic())
    assert not (tmp_path / ".pack.scenario-pack.lock").exists()
    assert not list(tmp_path.glob(".pack.*"))
    assert matplotlib.rcParams["text.usetex"] == initial_rc["text.usetex"]
    if failure == "output_race":
        assert (output / "other-writer").read_text() == "preserve"
    else:
        assert not output.exists()


def test_existing_output_overlap_and_symlink_refused(source, tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(ValueError, match="exists"):
        pack.build_pack(source, target, config=diagnostic())
    with pytest.raises(ValueError, match="overlap"):
        pack.build_pack(source, source / "inside", config=diagnostic())
    (source / "link").symlink_to(source / "proposal.json")
    with pytest.raises(ValueError, match="symlink"):
        pack.build_pack(source, tmp_path / "linked", config=diagnostic())


def test_total_point_budget_refuses_instead_of_downsampling(source, tmp_path):
    with pytest.raises(ValueError, match="max_points"):
        pack.build_pack(source, tmp_path / "too-many", config=diagnostic(max_points=8))
    assert not (tmp_path / "too-many").exists()


@pytest.mark.parametrize("size", ["single", "double"])
def test_world_legend_does_not_overlap_axis_labels(case, size):
    config = diagnostic(size=size)
    figure, _receipt = pack.render_view(pack.prepare_case(case, config), "trajectory", config)
    figure.canvas.draw()
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis = figure.axes[0]
    legend_box = axis.get_legend().get_window_extent(renderer)
    label_box = axis.xaxis.label.get_window_extent(renderer)
    assert not legend_box.overlaps(label_box)
    assert legend_box.x0 >= 0
    assert legend_box.x1 <= figure.bbox.x1
    figure.clear()


def test_large_recorded_footprints_are_not_clipped(case):
    case["trace"]["steps"][0]["robot"]["radius_m"] = 2.0
    prepared = pack.prepare_case(case, diagnostic())
    figure, receipt = pack.render_view(prepared, "snapshot", diagnostic())
    assert receipt["world_limits"][0][0] <= -2.0
    assert receipt["world_limits"][1][0] <= -2.0
    figure.clear()
