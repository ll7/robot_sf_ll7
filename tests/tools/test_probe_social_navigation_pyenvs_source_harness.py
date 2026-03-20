"""Tests for Social-Navigation-PyEnvs source-harness probing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.tools import probe_social_navigation_pyenvs_source_harness as probe
from scripts.tools.probe_social_navigation_pyenvs_source_harness import _render_markdown, run_probe

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_fake_repo(repo_root: Path) -> None:
    _write(repo_root / "README.md", "# stub\n")
    _write(repo_root / "requirements.txt", "gymnasium==0.29.1\nnumpy==1.26.1\ntorch==2.1.1\n")
    _write(repo_root / "setup.py", "from setuptools import setup\n")
    _write(
        repo_root / "social_gym" / "__init__.py",
        "from gymnasium.envs.registration import register\n",
    )
    _write(repo_root / "social_gym" / "social_nav_gym.py", "class SocialNavGym: ...\n")
    _write(repo_root / "social_gym" / "social_nav_sim.py", "class SocialNavSim: ...\n")
    _write(
        repo_root / "social_gym" / "src" / "robot_agent.py",
        """
from social_gym.src.actuators import DifferentialDrive
from crowd_nav.utils.action import ActionXY, ActionRot
class RobotAgent:
    def check_validity(self, action):
        if self.kinematics == 'holonomic': assert isinstance(action, ActionXY)
        else: assert isinstance(action, ActionRot)
""",
    )
    _write(
        repo_root / "social_gym" / "src" / "motion_model_manager.py",
        """
class MotionModelManager:
    def __init__(self):
        self.robot_motion_model_title = None
        if policy_name == "orca":
            pass
        self.goals[:,:,:] = np.NaN
        self.obstacles[:,:,:,:] = np.NaN
""",
    )
    _write(
        repo_root / "crowd_nav" / "policy_no_train" / "policy_factory.py",
        """
policy_factory = dict()
policy_factory['none'] = object
policy_factory['bp'] = object
policy_factory['orca'] = object
policy_factory['socialforce'] = object
""",
    )
    _write(repo_root / "crowd_nav" / "train.py", "print('train')\n")


def test_run_probe_requires_checkout_assets(tmp_path: Path) -> None:
    """The probe should fail fast when the upstream checkout is incomplete."""
    with pytest.raises(FileNotFoundError, match="README.md"):
        run_probe(tmp_path / "missing", timeout_seconds=1)


def test_run_probe_marks_partial_reproducibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Successful simulator and ORCA runs with later env failure should be partial reproducibility."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    monkeypatch.setattr(probe, "_uv_command", lambda: "uv")
    monkeypatch.setattr(
        probe, "_extract_remote_url", lambda _repo_root: "https://example.com/repo.git"
    )
    monkeypatch.setattr(probe, "_packaged_weights_present", lambda _repo_root: False)

    results = {
        "package_import": probe.CommandResult("package_import", ["uv"], 0, None, "import_ok", ""),
        "env_make_main_runtime": probe.CommandResult(
            "env_make_main_runtime",
            ["uv"],
            1,
            "missing python dependency: socialforce",
            "",
            "ModuleNotFoundError: No module named 'socialforce'",
        ),
        "simulator_core_with_socialforce": probe.CommandResult(
            "simulator_core_with_socialforce", ["uv"], 0, None, "sim_ok 3 True", ""
        ),
        "env_make_with_socialforce": probe.CommandResult(
            "env_make_with_socialforce",
            ["uv"],
            1,
            "upstream NumPy 2 incompatibility: np.NaN",
            "",
            "AttributeError: np.NaN was removed in the NumPy 2.0 release.",
        ),
        "policy_registry_with_socialforce": probe.CommandResult(
            "policy_registry_with_socialforce", ["uv"], 0, None, "['bp', 'orca']", ""
        ),
        "robot_orca_policy_with_socialforce": probe.CommandResult(
            "robot_orca_policy_with_socialforce",
            ["uv"],
            0,
            None,
            "robot_motion_model orca crowdnav False",
            "",
        ),
        "pinned_requirements_probe": probe.CommandResult(
            "pinned_requirements_probe",
            ["uv"],
            1,
            "pinned requirements incompatible with current Python ABI",
            "",
            "requirements are unsatisfiable",
        ),
        "shimmed_orca_reset_step": probe.CommandResult(
            "shimmed_orca_reset_step",
            ["uv"],
            0,
            None,
            "shim_ok 5 ActionXY 5 False False",
            "",
        ),
    }

    def fake_run(
        name: str, command: list[str], cwd: Path, timeout_seconds: int
    ) -> probe.CommandResult:
        return results[name]

    monkeypatch.setattr(probe, "_run_command", fake_run)

    report = run_probe(repo_root, timeout_seconds=5)

    assert report.verdict == "source harness partially reproducible"
    assert report.failure_stage == "env_make_with_socialforce"
    assert report.failure_summary == "upstream NumPy 2 incompatibility: np.NaN"
    assert report.packaged_weights_present is False
    assert report.source_contract["robot_actuation"] == "differential_drive"
    assert report.source_contract["gymnasium_version"] == "0.29.1"
    assert report.source_contract["non_trainable_policies"] == ["bp", "none", "orca", "socialforce"]
    assert (
        "install socialforce extra dependency"
        in report.source_contract["minimal_local_compatibility_shims"]
    )


def test_run_probe_marks_blocked_when_simulator_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A simulator-core failure should keep the verdict in the blocked state."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    monkeypatch.setattr(probe, "_uv_command", lambda: "uv")
    monkeypatch.setattr(
        probe, "_extract_remote_url", lambda _repo_root: "https://example.com/repo.git"
    )
    monkeypatch.setattr(probe, "_packaged_weights_present", lambda _repo_root: False)

    def fake_run(
        name: str, command: list[str], cwd: Path, timeout_seconds: int
    ) -> probe.CommandResult:
        if name == "simulator_core_with_socialforce":
            return probe.CommandResult(
                name, command, 1, "missing python dependency: socialforce", "", ""
            )
        return probe.CommandResult(name, command, 0, None, "ok", "")

    monkeypatch.setattr(probe, "_run_command", fake_run)
    report = run_probe(repo_root, timeout_seconds=5)

    assert report.verdict == "source harness blocked"
    assert report.failure_stage == "simulator_core_with_socialforce"


def test_detect_failure_summary_covers_numpy_and_abi() -> None:
    """Known runtime blockers should map to stable failure summaries."""
    assert (
        probe._detect_failure_summary(
            "", "AttributeError: np.NaN was removed in the NumPy 2.0 release."
        )
        == "upstream NumPy 2 incompatibility: np.NaN"
    )
    assert (
        probe._detect_failure_summary("", "requirements are unsatisfiable")
        == "pinned requirements incompatible with current Python ABI"
    )


def test_render_markdown_mentions_partial_reproducibility(tmp_path: Path) -> None:
    """Markdown should explain the near-runnable verdict and remaining blockers."""
    report = probe.ProbeReport(
        issue=642,
        repo_remote_url="https://github.com/example/Social-Navigation-PyEnvs",
        repo_root=str(tmp_path / "repo"),
        verdict="source harness partially reproducible",
        failure_stage="env_make_with_socialforce",
        failure_summary="upstream NumPy 2 incompatibility: np.NaN",
        timeout_seconds=30,
        required_files={"readme": str(tmp_path / "repo" / "README.md")},
        source_contract={
            "gymnasium_version": "0.29.1",
            "numpy_version": "1.26.1",
            "torch_version": "2.1.1",
            "non_trainable_policies": ["orca"],
            "robot_actuation": "differential_drive",
            "robot_policy_accepts_holonomic_actions": True,
            "robot_policy_accepts_nonholonomic_actions": True,
            "orca_available_as_robot_motion_model": True,
            "orca_preferred_velocity_semantics": "goal_vector_pref_velocity",
            "runtime_bug_signature": "np.NaN removed in NumPy 2",
            "runtime_bug_locations": ["social_gym/src/motion_model_manager.py:264"],
            "minimal_local_compatibility_shims": ["install socialforce extra dependency"],
            "notes": "stub",
        },
        commands=[
            probe.CommandResult("package_import", ["uv", "run"], 0, None, "import_ok", ""),
            probe.CommandResult(
                "shimmed_orca_reset_step",
                ["uv", "run"],
                0,
                None,
                "shim_ok 5 ActionXY 5 False False",
                "",
            ),
        ],
        packaged_weights_present=False,
    )

    markdown = _render_markdown(report)
    assert "Verdict: `source harness partially reproducible`" in markdown
    assert "Full Gymnasium env creation is still blocked" in markdown
    assert "reset and step the upstream ORCA path once" in markdown
