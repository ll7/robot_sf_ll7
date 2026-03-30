"""Tests for the gym-collision-avoidance model parity probe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.tools import probe_gym_collision_avoidance_model_parity as probe

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


def test_run_probe_requires_side_env_python(tmp_path: Path) -> None:
    """Missing side-env interpreters should fail fast with a concrete error."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    with pytest.raises(FileNotFoundError, match="Side-environment interpreter missing"):
        probe.run_probe(repo_root, tmp_path / "missing-python", timeout_seconds=1)


def test_run_probe_reports_native_model_parity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exact argmax and probability parity should yield the reproduced verdict."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    side_env_python = tmp_path / "side" / ".venv" / "bin" / "python"
    side_env_python.parent.mkdir(parents=True)
    side_env_python.write_text("", encoding="utf-8")

    upstream_payload = {
        "vec_obs": [[0.1, 0.2]],
        "upstream_probs": [0.1, 0.9],
        "upstream_argmax": 1,
        "upstream_raw_action": [1.0, 0.0],
        "upstream_final_action": [1.0, 0.0],
        "upstream_actions": [[0.5, 0.0], [1.0, 0.0]],
        "checkpoint_prefix": "/tmp/checkpoint",
        "obs_shape": [1, 2],
        "states_used": ["dist_to_goal", "pref_speed"],
    }
    local_payload = {
        "local_argmax": 1,
        "local_raw_action": [1.0, 0.0],
        "prob_max_abs_diff": 0.0,
        "actions_max_abs_diff": 0.0,
    }
    results = {
        "upstream_native_observation_and_policy": probe.CommandResult(
            name="upstream_native_observation_and_policy",
            command=["python"],
            returncode=0,
            failure_summary=None,
            stdout_tail=f"banner\n{__import__('json').dumps(upstream_payload)}\n",
            stderr_tail="",
        ),
        "local_sacadrl_model_parity": probe.CommandResult(
            name="local_sacadrl_model_parity",
            command=["uv", "run", "python"],
            returncode=0,
            failure_summary=None,
            stdout_tail=f"{__import__('json').dumps(local_payload)}\n",
            stderr_tail="",
        ),
    }

    monkeypatch.setattr(
        probe,
        "_run_command",
        lambda name, command, cwd, timeout_seconds: results[name],
    )
    report = probe.run_probe(repo_root, side_env_python, timeout_seconds=10)

    assert report.verdict == "native-model parity reproduced"
    assert report.parity_summary["upstream_argmax"] == 1
    assert report.parity_summary["local_argmax"] == 1


def test_run_probe_reports_material_model_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Argmax or probability drift should mark the parity result as a material mismatch."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    side_env_python = tmp_path / "side" / ".venv" / "bin" / "python"
    side_env_python.parent.mkdir(parents=True)
    side_env_python.write_text("", encoding="utf-8")

    upstream_payload = {
        "vec_obs": [[0.1, 0.2]],
        "upstream_probs": [0.1, 0.9],
        "upstream_argmax": 1,
        "upstream_raw_action": [1.0, 0.0],
        "upstream_final_action": [1.0, 0.0],
        "upstream_actions": [[0.5, 0.0], [1.0, 0.0]],
        "checkpoint_prefix": "/tmp/checkpoint",
        "obs_shape": [1, 2],
        "states_used": ["dist_to_goal", "pref_speed"],
    }
    local_payload = {
        "local_argmax": 0,
        "local_raw_action": [0.5, 0.0],
        "prob_max_abs_diff": 0.1,
        "actions_max_abs_diff": 0.0,
    }
    results = {
        "upstream_native_observation_and_policy": probe.CommandResult(
            name="upstream_native_observation_and_policy",
            command=["python"],
            returncode=0,
            failure_summary=None,
            stdout_tail=f"{__import__('json').dumps(upstream_payload)}\n",
            stderr_tail="",
        ),
        "local_sacadrl_model_parity": probe.CommandResult(
            name="local_sacadrl_model_parity",
            command=["uv", "run", "python"],
            returncode=0,
            failure_summary=None,
            stdout_tail=f"{__import__('json').dumps(local_payload)}\n",
            stderr_tail="",
        ),
    }

    monkeypatch.setattr(
        probe,
        "_run_command",
        lambda name, command, cwd, timeout_seconds: results[name],
    )
    report = probe.run_probe(repo_root, side_env_python, timeout_seconds=10)

    assert report.verdict == "material model mismatch"


def test_run_probe_uses_repo_root_for_local_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The local parity stage should run from the repository root, not the caller cwd."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    side_env_python = tmp_path / "side" / ".venv" / "bin" / "python"
    side_env_python.parent.mkdir(parents=True)
    side_env_python.write_text("", encoding="utf-8")

    commands: list[tuple[str, list[str], Path]] = []
    upstream_payload = {
        "vec_obs": [[0.1, 0.2]],
        "upstream_probs": [0.1, 0.9],
        "upstream_argmax": 1,
        "upstream_raw_action": [1.0, 0.0],
        "upstream_final_action": [1.0, 0.0],
        "upstream_actions": [[0.5, 0.0], [1.0, 0.0]],
        "checkpoint_prefix": "/tmp/checkpoint",
        "obs_shape": [1, 2],
        "states_used": ["dist_to_goal", "pref_speed"],
    }
    local_payload = {
        "local_argmax": 1,
        "local_raw_action": [1.0, 0.0],
        "prob_max_abs_diff": 0.0,
        "actions_max_abs_diff": 0.0,
    }

    def fake_run(
        name: str, command: list[str], cwd: Path, timeout_seconds: int
    ) -> probe.CommandResult:
        commands.append((name, command, cwd))
        if name == "upstream_native_observation_and_policy":
            return probe.CommandResult(
                name=name,
                command=command,
                returncode=0,
                failure_summary=None,
                stdout_tail=f"banner\n{__import__('json').dumps(upstream_payload)}\n",
                stderr_tail="",
            )
        return probe.CommandResult(
            name=name,
            command=command,
            returncode=0,
            failure_summary=None,
            stdout_tail=f"{__import__('json').dumps(local_payload)}\n",
            stderr_tail="",
        )

    monkeypatch.setattr(probe, "_run_command", fake_run)
    probe.run_probe(repo_root, side_env_python, timeout_seconds=10)

    local_stage = next(item for item in commands if item[0] == "local_sacadrl_model_parity")
    assert local_stage[2] == repo_root.resolve()
    assert "payload.json" in " ".join(local_stage[1])


def test_parse_json_stdout_reads_last_json_line() -> None:
    """The parser should ignore banner noise and recover the trailing JSON payload."""
    result = probe.CommandResult(
        name="sample",
        command=["python"],
        returncode=0,
        failure_summary=None,
        stdout_tail='banner\n{"foo": 1}\n',
        stderr_tail="",
    )
    assert probe._parse_json_stdout(result) == {"foo": 1}


def test_upstream_payload_script_uses_checkpoint_constant() -> None:
    """The embedded upstream payload should derive its checkpoint prefix from the shared constant."""
    script = probe._upstream_payload_script()
    assert probe.CHECKPOINT_PREFIX_RELATIVE.as_posix() in script
    assert "__CHECKPOINT_PREFIX_RELATIVE__" not in script


def test_render_markdown_mentions_observation_mapping_gap(tmp_path: Path) -> None:
    """Successful markdown should make the remaining mapping gap explicit."""
    repo_root = tmp_path / "repo"
    _write_fake_repo(repo_root)
    report = probe.ProbeReport(
        issue=661,
        repo_root=str(repo_root),
        repo_remote_url="https://github.com/mit-acl/gym-collision-avoidance",
        side_env_python=str(tmp_path / "python"),
        verdict="native-model parity reproduced",
        failure_stage=None,
        failure_summary=None,
        parity_summary={
            "upstream_argmax": 4,
            "local_argmax": 4,
            "prob_max_abs_diff": 0.0,
            "actions_max_abs_diff": 0.0,
            "obs_shape": [1, 132],
            "states_used": ["dist_to_goal"],
            "upstream_final_action": [1.0, 0.0],
            "local_raw_action": [1.0, 0.0],
        },
        source_contract=probe._extract_source_contract(),
        commands=[],
    )
    markdown = probe._render_markdown(report)
    assert "Robot SF observation mapping and benchmark behavior" in markdown


def test_main_returns_zero_for_native_model_parity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI exit code should be zero when the parity probe reproduces the native model."""
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    report = probe.ProbeReport(
        issue=661,
        repo_root=str(tmp_path / "repo"),
        repo_remote_url="https://github.com/mit-acl/gym-collision-avoidance",
        side_env_python=str(tmp_path / "python"),
        verdict="native-model parity reproduced",
        failure_stage=None,
        failure_summary=None,
        parity_summary={
            "upstream_argmax": 4,
            "local_argmax": 4,
            "prob_max_abs_diff": 0.0,
            "actions_max_abs_diff": 0.0,
            "obs_shape": [1, 132],
            "states_used": ["dist_to_goal"],
            "upstream_final_action": [1.0, 0.0],
            "local_raw_action": [1.0, 0.0],
        },
        source_contract=probe._extract_source_contract(),
        commands=[],
    )
    monkeypatch.setattr(probe, "run_probe", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe_gym_collision_avoidance_model_parity.py",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
    )

    assert probe.main() == 0
    assert output_json.exists()
    assert output_md.exists()
