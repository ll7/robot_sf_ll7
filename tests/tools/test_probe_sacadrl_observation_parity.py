"""Tests for the SACADRL observation parity probe."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools import probe_sacadrl_observation_parity as probe

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_fake_repo(repo_root: Path) -> None:
    _write(repo_root / "README.md", "# stub\n")
    _write(
        repo_root / "gym_collision_avoidance" / "experiments" / "src" / "example.py",
        "print('example')\n",
    )


def test_run_native_roundtrip_case_reproduces_live_contract() -> None:
    """A live upstream-style native state should round-trip through the adapter exactly."""
    native_state = {
        "num_other_agents": 1,
        "dist_to_goal": 8.48528137423857,
        "heading_ego_frame": -0.7853981633974483,
        "pref_speed": 1.0,
        "radius": 0.5,
        "other_agents_states": [
            [8.485281374238571, 0.0, 0.0, 0.0, 0.5, 1.0, 7.4852813742385695],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    }
    case = probe._run_native_roundtrip_case("live", native_state)
    assert case.verdict == "parity reproduced"
    assert case.max_abs_diff == pytest.approx(0.0, abs=1e-6)


def test_controlled_rotated_case_reproduces_parity() -> None:
    """The controlled rotated multi-agent case should preserve the full CADRL input."""
    case = probe._run_controlled_rotated_case()
    assert case.verdict == "parity reproduced"
    assert case.max_abs_diff == pytest.approx(0.0, abs=1e-6)


def test_socnav_fusion_case_confirms_velocity_frame_contract() -> None:
    """The SocNav observation builder and adapter should agree on velocity-frame handling."""
    case = probe._run_socnav_fusion_case()
    assert case.verdict == "parity reproduced"
    assert case.component_max_abs_diff["other_agents_states"] == pytest.approx(0.0, abs=1e-6)


def test_run_probe_reports_reproduced_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A successful live payload should yield the reproduced observation-mapping verdict."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    side_env_python = tmp_path / "side" / ".venv" / "bin" / "python"
    side_env_python.parent.mkdir(parents=True)
    side_env_python.write_text("", encoding="utf-8")

    payload = {
        "native_state": {
            "num_other_agents": 1,
            "dist_to_goal": 8.48528137423857,
            "heading_ego_frame": -0.7853981633974483,
            "pref_speed": 1.0,
            "radius": 0.5,
            "other_agents_states": [
                [8.485281374238571, 0.0, 0.0, 0.0, 0.5, 1.0, 7.4852813742385695],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        }
    }
    result = probe.CommandResult(
        name="upstream_live_native_state",
        command=["python"],
        returncode=0,
        failure_summary=None,
        stdout_tail=f"banner\n{json.dumps(payload)}\n",
        stderr_tail="",
    )
    monkeypatch.setattr(probe, "_run_command", lambda *args, **kwargs: result)

    report = probe.run_probe(repo_root, side_env_python, timeout_seconds=10)

    assert report.verdict == "adapter observation mapping reproduced in controlled cases"
    assert len(report.cases) == 3
    assert max(case.max_abs_diff for case in report.cases) == pytest.approx(0.0, abs=1e-6)


def test_run_probe_blocks_on_malformed_upstream_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Malformed upstream JSON should produce a blocked report instead of crashing."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    side_env_python = tmp_path / "side" / ".venv" / "bin" / "python"
    side_env_python.parent.mkdir(parents=True)
    side_env_python.write_text("", encoding="utf-8")

    result = probe.CommandResult(
        name="upstream_live_native_state",
        command=["python"],
        returncode=0,
        failure_summary=None,
        stdout_tail="banner\nnot-json\n",
        stderr_tail="",
    )
    monkeypatch.setattr(probe, "_run_command", lambda *args, **kwargs: result)

    report = probe.run_probe(repo_root, side_env_python, timeout_seconds=10)

    assert report.verdict == "parity blocked"
    assert report.failure_stage == "upstream_live_native_state"


def test_upstream_payload_sets_config_before_env_import() -> None:
    """The upstream payload should set GYM_CONFIG_CLASS before importing gym_collision_avoidance.envs."""
    script = probe._upstream_live_payload_script()
    assert script.index("os.environ['GYM_CONFIG_CLASS'] = 'Example'") < script.index(
        "from gym_collision_avoidance.envs import test_cases as tc"
    )


def test_render_markdown_recommends_benchmarkable_proxy(tmp_path: Path) -> None:
    """Successful markdown should frame SACADRL as benchmarkable evidence, not a strong planner."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    report = probe.ProbeReport(
        issue=663,
        repo_root=str(repo_root),
        repo_remote_url="https://github.com/mit-acl/gym-collision-avoidance",
        side_env_python=str(tmp_path / "python"),
        verdict="adapter observation mapping reproduced in controlled cases",
        failure_stage=None,
        failure_summary=None,
        cases=[
            probe.CaseResult(
                name="case",
                verdict="parity reproduced",
                max_abs_diff=0.0,
                component_max_abs_diff={"dist_to_goal": 0.0},
                notes=[],
            )
        ],
        source_contract=probe._extract_source_contract(),
        commands=[],
    )
    markdown = probe._render_markdown(report)
    assert "benchmarkable CADRL-family evidence" in markdown
