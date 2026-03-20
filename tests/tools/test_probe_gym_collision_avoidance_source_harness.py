"""Tests for gym-collision-avoidance source-harness probing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.tools import probe_gym_collision_avoidance_source_harness as probe
from scripts.tools.probe_gym_collision_avoidance_source_harness import _render_markdown, run_probe

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_fake_repo(repo_root: Path) -> None:
    _write(repo_root / "README.md", "# stub\n")
    _write(
        repo_root / "gym_collision_avoidance" / "__init__.py",
        "from gym.envs.registration import register\n",
    )
    _write(
        repo_root / "gym_collision_avoidance" / "envs" / "config.py",
        """
class Config:
    def __init__(self):
        self.DT = 0.2
        self.MAX_NUM_OTHER_AGENTS_OBSERVED = 3
        self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal']
        self.STATES_NOT_USED_IN_POLICY = ['is_learning']
""",
    )
    _write(
        repo_root / "gym_collision_avoidance" / "envs" / "test_cases.py",
        """
def get_testcase_two_agents(policies=['learning', 'GA3C_CADRL']):
    return policies
""",
    )
    _write(
        repo_root / "gym_collision_avoidance" / "experiments" / "src" / "example.py",
        "print('example')\n",
    )
    _write(
        repo_root / "gym_collision_avoidance" / "tests" / "test_collision_avoidance.py",
        "def test_example_script():\n    assert True\n",
    )
    _write(
        repo_root / "gym_collision_avoidance" / "envs" / "policies" / "GA3CCADRLPolicy.py",
        """
import numpy as np
class GA3CCADRLPolicy:
    def initialize_network(self):
        return True
    def find_next_action(self, obs, agents, i):
        pref_speed = obs['pref_speed']
        raw_action = np.array([1.0, 0.0])
        action = np.array([pref_speed*raw_action[0], raw_action[1]])
        return action
""",
    )
    _write(
        repo_root / "gym_collision_avoidance" / "envs" / "policies" / "GA3C_CADRL" / "network.py",
        "# Define 11 choices of actions\n",
    )
    _write(
        repo_root
        / "gym_collision_avoidance"
        / "envs"
        / "policies"
        / "GA3C_CADRL"
        / "checkpoints"
        / "IROS18"
        / "network_01900000.meta",
        "meta\n",
    )
    _write(
        repo_root
        / "gym_collision_avoidance"
        / "envs"
        / "policies"
        / "GA3C_CADRL"
        / "checkpoints"
        / "IROS18"
        / "network_01900000.index",
        "index\n",
    )
    _write(
        repo_root
        / "gym_collision_avoidance"
        / "envs"
        / "policies"
        / "GA3C_CADRL"
        / "checkpoints"
        / "IROS18"
        / "network_01900000.data-00000-of-00001",
        "data\n",
    )


def test_run_probe_blocks_on_missing_assets(tmp_path: Path) -> None:
    """Missing source files should fail fast with a concrete missing-assets error."""
    with pytest.raises(FileNotFoundError, match="README.md"):
        run_probe(tmp_path / "missing_repo", timeout_seconds=1)


def test_run_probe_captures_missing_dependency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A blocked upstream run should surface the first missing dependency and contract."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)

    def fake_run(
        name: str, command: list[str], cwd: Path, timeout_seconds: int
    ) -> probe.CommandResult:
        return probe.CommandResult(
            name=name,
            command=command,
            returncode=1,
            failure_summary="missing python dependency: gym",
            stdout_tail="",
            stderr_tail="ModuleNotFoundError: No module named 'gym'",
        )

    monkeypatch.setattr(probe, "_run_command", fake_run)

    report = run_probe(repo_root, timeout_seconds=5)

    assert report.verdict == "source harness blocked"
    assert report.failure_stage == "upstream_example"
    assert report.failure_summary == "missing python dependency: gym"
    assert report.source_contract["learned_policy"] == "GA3C_CADRL"
    assert report.source_contract["action_space"] == "speed_delta_heading"
    assert report.source_contract["discrete_action_count"] == 11


def test_run_probe_marks_success_when_all_commands_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The probe should mark the source harness reproducible when every command passes."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)

    def fake_run(
        name: str, command: list[str], cwd: Path, timeout_seconds: int
    ) -> probe.CommandResult:
        return probe.CommandResult(
            name=name,
            command=command,
            returncode=0,
            failure_summary=None,
            stdout_tail="ok",
            stderr_tail="",
        )

    monkeypatch.setattr(probe, "_run_command", fake_run)

    report = run_probe(repo_root, timeout_seconds=5)

    assert report.verdict == "source harness reproducible"
    assert report.failure_stage is None
    assert report.failure_summary is None


def test_run_command_reports_timeout(tmp_path: Path) -> None:
    """Timeouts should become structured blocked results rather than exceptions."""
    result = probe._run_command(
        "slow",
        [probe.sys.executable, "-c", "import time; time.sleep(2)"],
        cwd=tmp_path,
        timeout_seconds=1,
    )
    assert result.returncode is None
    assert result.failure_summary == "command exceeded timeout (1s)"


def test_render_markdown_mentions_wrapper_recommendation(tmp_path: Path) -> None:
    """Markdown output should end with a wrapper recommendation tied to the verdict."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)

    report = probe.ProbeReport(
        issue=639,
        repo_remote_url="https://github.com/mit-acl/gym-collision-avoidance",
        repo_root=str(repo_root),
        verdict="source harness blocked",
        failure_stage="upstream_example",
        failure_summary="missing python dependency: gym",
        timeout_seconds=30,
        required_files={"readme": str(repo_root / "README.md")},
        source_contract={
            "example_default_policies": ["learning", "GA3C_CADRL"],
            "observation_states_in_obs": ["dist_to_goal"],
            "observation_states_not_used_in_policy": [],
            "observation_encoding": "flattened_dict_obs_excluding_states_not_used_in_policy",
            "dt_seconds": 0.2,
            "max_num_other_agents_observed": 3,
            "learned_policy": "GA3C_CADRL",
            "action_space": "speed_delta_heading",
            "discrete_action_count": 11,
            "checkpoint_family": "GA3C_CADRL/checkpoints/IROS18/network_01900000",
            "kinematics": "unicycle_like_speed_plus_delta_heading",
        },
        commands=[
            probe.CommandResult(
                name="upstream_example",
                command=["python", "gym_collision_avoidance/experiments/src/example.py"],
                returncode=1,
                failure_summary="missing python dependency: gym",
                stdout_tail="",
                stderr_tail="ModuleNotFoundError",
            )
        ],
    )

    markdown = _render_markdown(report)
    assert "gym-collision-avoidance Source Harness Probe" in markdown
    assert "Verdict: `source harness blocked`" in markdown
    assert "Wrapper work is not yet justified" in markdown
