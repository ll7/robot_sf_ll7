"""Tests for the Social-Navigation-PyEnvs SocialForce runtime probe."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools import probe_social_navigation_pyenvs_socialforce_runtime as probe

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_fake_repo(repo_root: Path) -> None:
    _write(
        repo_root / "crowd_nav" / "policy_no_train" / "socialforce.py",
        """
import socialforce
class SocialForce:
    def __init__(self):
        self.initial_speed = 1
        self.v0 = 10
        self.sigma = 0.3
    def predict(self, state):
        sim = socialforce.Simulator(state, delta_t=self.time_step, initial_speed=self.initial_speed, v0=self.v0, sigma=self.sigma)
        sim.step()
""",
    )


def test_run_probe_requires_upstream_policy_file(tmp_path: Path) -> None:
    """The probe should fail fast when the upstream policy file is missing."""
    with pytest.raises(FileNotFoundError, match="socialforce.py"):
        probe.run_probe(tmp_path / "missing")


def test_run_probe_marks_runtime_compatible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A successful backend probe and compat probe should yield a compatible verdict."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    monkeypatch.setattr(
        probe, "_extract_remote_url", lambda _repo_root: "https://example.com/repo.git"
    )

    results = {
        "backend_signature": probe.CommandResult(
            "backend_signature",
            ["uv"],
            0,
            None,
            json.dumps(
                {
                    "backend_version": "0.2.3",
                    "simulator_signature": "(self, *, ped_space=None, delta_t=0.4)",
                }
            ),
            "",
        ),
        "compat_predict_minimal": probe.CommandResult(
            "compat_predict_minimal",
            ["uv"],
            0,
            None,
            json.dumps(
                {
                    "backend_version": "0.2.3",
                    "shim_accepts_initial_speed": True,
                    "action_xy": [0.1, -0.2],
                    "sim_state_shape": [2, 10],
                }
            ),
            "",
        ),
    }

    monkeypatch.setattr(
        probe,
        "_run_command",
        lambda name, command, cwd, timeout_seconds: results[name],
    )

    report = probe.run_probe(repo_root, timeout_seconds=5)

    assert report.verdict == "compatible runtime reproduced"
    assert report.failure_stage is None
    assert report.source_contract["shim_accepts_initial_speed"] is True
    assert report.source_contract["compat_probe_sim_state_shape"] == [2, 10]


def test_run_probe_marks_blocked_when_compat_probe_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A compat-probe failure should keep the runtime verdict blocked."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    monkeypatch.setattr(probe, "_extract_remote_url", lambda _repo_root: None)

    def fake_run(
        name: str, command: list[str], cwd: Path, timeout_seconds: int
    ) -> probe.CommandResult:
        if name == "backend_signature":
            return probe.CommandResult(
                name,
                command,
                0,
                None,
                json.dumps({"backend_version": "0.2.3", "simulator_signature": "sig"}),
                "",
            )
        return probe.CommandResult(
            name,
            command,
            1,
            "current socialforce package API mismatches upstream simulator constructor",
            "",
            "TypeError: Simulator.__init__() got an unexpected keyword argument 'initial_speed'",
        )

    monkeypatch.setattr(probe, "_run_command", fake_run)
    report = probe.run_probe(repo_root, timeout_seconds=5)

    assert report.verdict == "blocked by dependency/runtime mismatch"
    assert report.failure_stage == "compat_predict_minimal"


def test_detect_failure_summary_covers_known_api_mismatch() -> None:
    """Known constructor mismatches should map to a stable failure summary."""
    assert (
        probe._detect_failure_summary(
            "",
            "TypeError: Simulator.__init__() got an unexpected keyword argument 'initial_speed'",
        )
        == "current socialforce package API mismatches upstream simulator constructor"
    )


def test_render_markdown_mentions_compatibility_runtime(tmp_path: Path) -> None:
    """Markdown should explain the shim-based compatible-runtime verdict."""
    report = probe.ProbeReport(
        issue=653,
        repo_root=str(tmp_path / "repo"),
        repo_remote_url="https://github.com/example/Social-Navigation-PyEnvs",
        backend_spec="socialforce==0.2.3",
        verdict="compatible runtime reproduced",
        failure_stage=None,
        failure_summary=None,
        source_contract={
            "upstream_policy": "crowd_nav.policy_no_train.socialforce.SocialForce",
            "external_simulator_constructor_expects": ["initial_speed", "v0", "sigma"],
            "runtime_strategy": "compatibility shim around external socialforce package",
            "backend_simulator_signature": "sig",
            "compat_probe_action_xy": [0.0, 0.0],
            "compat_probe_sim_state_shape": [2, 10],
        },
        commands=[probe.CommandResult("compat_predict_minimal", ["uv"], 0, None, "", "")],
    )

    markdown = probe._render_markdown(report)

    assert "Verdict: `compatible runtime reproduced`" in markdown
    assert "compatibility shim" in markdown
    assert "compat simulator state shape" in markdown
