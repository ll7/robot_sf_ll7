"""Tests for the MuJoCo AMV micro-backend diagnostic helper."""

from __future__ import annotations

import csv
import json
import types
from pathlib import Path

import pytest

from scripts.tools import mujoco_amv_micro_backend as micro

_TRACE_FIXTURE = Path(
    "tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json"
)


class _FakeModel:
    @classmethod
    def from_xml_string(cls, xml: str) -> _FakeModel:
        """Build a fake model while preserving the XML for assertions."""
        model = cls()
        model.xml = xml
        return model


class _FakeData:
    def __init__(self, model: _FakeModel) -> None:
        """Store the model reference."""
        self.model = model
        self.steps = 0


def _fake_mujoco() -> types.SimpleNamespace:
    """Return a tiny MuJoCo-like module object."""

    def mj_step(_model: _FakeModel, data: _FakeData) -> None:
        data.steps += 1

    return types.SimpleNamespace(
        __version__="fake-3.9.0",
        MjModel=_FakeModel,
        MjData=_FakeData,
        mj_step=mj_step,
    )


def test_replay_commands_emits_diagnostic_payload() -> None:
    """The replay should emit bounded diagnostic values and explicit claim boundaries."""
    payload = micro.replay_commands(
        [
            micro.CommandSegment(duration_s=0.2, v_m_s=2.0, omega_rad_s=0.0),
            micro.CommandSegment(duration_s=0.2, v_m_s=2.0, omega_rad_s=2.0),
            micro.CommandSegment(duration_s=0.2, v_m_s=0.0, omega_rad_s=0.0),
        ],
        micro.ReplayConfig(
            timestep_s=0.1,
            max_linear_accel_m_s2=1.0,
            max_linear_decel_m_s2=1.5,
            max_angular_accel_rad_s2=2.0,
            max_yaw_rate_rad_s=0.5,
            latency_steps=1,
        ),
        mujoco=_fake_mujoco(),
    )

    assert payload["schema_version"] == micro.SCHEMA_VERSION
    assert payload["status"] == "completed"
    assert payload["runtime"]["mujoco_version"] == "fake-3.9.0"
    assert payload["runtime"]["routine_dependency"] is False
    assert "not calibrated AMV hardware evidence" in payload["claim_boundary"]
    assert "social-navigation benchmark outcomes" in payload["unsupported_semantics"]
    assert payload["summary"]["steps"] == 6
    assert payload["summary"]["command_clip_fraction"] > 0
    assert payload["summary"]["max_abs_yaw_rate_rad_s"] <= 0.5
    assert payload["rows"][0]["applied_v_m_s"] == 0.0
    assert payload["rows"][1]["linear_rate_clipped"] is True


def test_cli_writes_json_and_markdown_with_csv_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The CLI should write reviewable JSON and Markdown outputs."""
    monkeypatch.setattr(micro, "_load_mujoco", _fake_mujoco)
    commands = tmp_path / "commands.csv"
    with commands.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["duration_s", "v_m_s", "omega_rad_s"])
        writer.writeheader()
        writer.writerow({"duration_s": "0.2", "v_m_s": "1.0", "omega_rad_s": "0.0"})

    output_json = tmp_path / "diagnostic.json"
    output_md = tmp_path / "diagnostic.md"
    exit_code = micro.main(
        [
            "--commands",
            str(commands),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--timestep",
            "0.1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == micro.SCHEMA_VERSION
    markdown = output_md.read_text(encoding="utf-8")
    assert "MuJoCo AMV Micro-Backend Diagnostic" in markdown
    assert "Unsupported Semantics" in markdown


def test_cli_accepts_simulation_trace_export_fixture(tmp_path: Path, monkeypatch) -> None:
    """The trace mode should preserve source trace metadata and selected actions."""
    monkeypatch.setattr(micro, "_load_mujoco", _fake_mujoco)
    output_json = tmp_path / "diagnostic.json"

    exit_code = micro.main(
        [
            "--trace",
            str(_TRACE_FIXTURE),
            "--output-json",
            str(output_json),
            "--timestep",
            "0.1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["source_trace"]["schema_version"] == "simulation_trace_export.v1"
    assert payload["source_trace"]["trace_id"] == "fixture_trace_001"
    assert payload["source_trace"]["source"]["planner_id"] == "hybrid_rule_v0_minimal"
    assert payload["command_contract"]["action_space"] == "unicycle_vw"
    assert payload["rows"][0]["source_step"] == 0
    assert payload["rows"][0]["commanded_v_m_s"] == pytest.approx(0.1)
    assert payload["rows"][1]["source_step"] == 1


def test_cli_fails_closed_when_mujoco_missing(tmp_path: Path, monkeypatch) -> None:
    """Missing MuJoCo should return a non-zero exit instead of faking success."""

    def _raise_missing() -> types.SimpleNamespace:
        raise RuntimeError("MuJoCo is not available")

    monkeypatch.setattr(micro, "_load_mujoco", _raise_missing)
    exit_code = micro.main(
        [
            "--demo-fixture",
            "--output-json",
            str(tmp_path / "diagnostic.json"),
        ]
    )

    assert exit_code == 2


def test_cli_fails_closed_for_invalid_trace(tmp_path: Path, monkeypatch) -> None:
    """Invalid trace payloads should not be silently replayed."""
    monkeypatch.setattr(micro, "_load_mujoco", _fake_mujoco)
    invalid_trace = tmp_path / "invalid_trace.json"
    invalid_trace.write_text(
        json.dumps(
            {
                "schema_version": "simulation_trace_export.v1",
                "trace_id": "bad",
                "source": {
                    "scenario_id": "scenario",
                    "seed": 1,
                    "planner_id": "planner",
                    "episode_id": "episode",
                    "generated_by": "test",
                },
                "evidence_boundary": "analysis_workbench_only",
                "coordinate_frame": "world",
                "units": {
                    "position": "m",
                    "heading": "rad",
                    "time": "s",
                    "velocity": "m/s",
                },
                "frames": [
                    {
                        "step": 0,
                        "time_s": 0.0,
                        "robot": {
                            "position": [0.0, 0.0],
                            "heading": 0.0,
                            "velocity": [0.0, 0.0],
                        },
                        "pedestrians": [],
                        "planner": {"selected_action": {"linear_velocity": 0.0}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = micro.main(
        [
            "--trace",
            str(invalid_trace),
            "--output-json",
            str(tmp_path / "diagnostic.json"),
        ]
    )

    assert exit_code == 2
