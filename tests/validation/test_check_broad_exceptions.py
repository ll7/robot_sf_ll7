"""Tests for the broad-exception inventory ratchet."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validation" / "check_broad_exceptions.py"


def _git(repo: Path, *args: str) -> None:
    """Run a git command in a temporary repository."""
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _make_repo(tmp_path: Path) -> Path:
    """Create a tiny tracked repository with one ratcheted Python file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")
    target = repo / "scripts" / "demo" / "tool.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "def main() -> None:\n"
        "    try:\n"
        "        raise RuntimeError('demo')\n"
        "    except RuntimeError:\n"
        "        pass\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_broad_exception_ratchet_passes_generated_baseline(tmp_path: Path) -> None:
    """A freshly generated baseline passes without source changes."""
    repo = _make_repo(tmp_path)
    baseline = Path("scripts/validation/broad_exception_baseline.json")
    (repo / baseline.parent).mkdir(parents=True, exist_ok=True)

    write_result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(repo),
            "--baseline",
            str(baseline),
            "--write-baseline",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert write_result.returncode == 0, write_result.stderr

    check_result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo), "--baseline", str(baseline)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check_result.returncode == 0, check_result.stderr
    assert "Broad exception ratchet passed with 0 entries." in check_result.stdout


def test_broad_exception_ratchet_fails_on_added_broad_catch(tmp_path: Path) -> None:
    """Adding an unapproved broad catch fails against the existing baseline."""
    repo = _make_repo(tmp_path)
    baseline = Path("scripts/validation/broad_exception_baseline.json")
    (repo / baseline.parent).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(repo),
            "--baseline",
            str(baseline),
            "--write-baseline",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    (repo / "scripts" / "demo" / "tool.py").write_text(
        "def main() -> None:\n"
        "    try:\n"
        "        raise RuntimeError('demo')\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo), "--baseline", str(baseline)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Broad exception count increased from 0 to 1." in result.stderr
    assert "Unapproved broad exception handlers were added:" in result.stderr
    assert "scripts/demo/tool.py:4: except Exception:" in result.stderr


def test_broad_exception_ratchet_fails_on_unapproved_replacement(tmp_path: Path) -> None:
    """Moving a broad catch to a new fingerprint fails even when count is unchanged."""
    repo = _make_repo(tmp_path)
    baseline = Path("scripts/validation/broad_exception_baseline.json")
    (repo / baseline.parent).mkdir(parents=True, exist_ok=True)
    script = repo / "scripts" / "demo" / "tool.py"
    script.write_text(
        "def main() -> None:\n"
        "    try:\n"
        "        raise RuntimeError('demo')\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(repo),
            "--baseline",
            str(baseline),
            "--write-baseline",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    script.write_text(
        "def helper() -> None:\n"
        "    try:\n"
        "        raise RuntimeError('demo')\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo), "--baseline", str(baseline)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Broad exception count increased" not in result.stderr
    assert "Unapproved broad exception handlers were added:" in result.stderr
    assert "scripts/demo/tool.py:4: except Exception:" in result.stderr


def test_broad_exception_ratchet_rejects_stale_summary_counts(tmp_path: Path) -> None:
    """A stale counts summary must fail even when entry fingerprints are unchanged."""
    repo = _make_repo(tmp_path)
    baseline = Path("scripts/validation/broad_exception_baseline.json")
    (repo / baseline.parent).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(repo),
            "--baseline",
            str(baseline),
            "--write-baseline",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    baseline_path = repo / baseline
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload["counts"]["total"] = 99
    baseline_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo), "--baseline", str(baseline)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Broad exception baseline summary is stale" in result.stderr


def test_ratchet_is_wired_into_ci_lint_phase() -> None:
    """The CI lint phase must invoke the ratchet so it is actually enforced (issue #3478)."""
    ci_driver = (ROOT / "scripts" / "dev" / "ci_driver.sh").read_text(encoding="utf-8")
    assert "check_broad_exceptions.py" in ci_driver, (
        "broad-exception ratchet is not invoked by ci_driver.sh; the guard would be dead"
    )


def test_ratchet_is_wired_into_pr_ready_check() -> None:
    """The local PR-readiness gate must also invoke the ratchet (issue #3478)."""
    pr_ready = (ROOT / "scripts" / "dev" / "pr_ready_check.sh").read_text(encoding="utf-8")
    assert "check_broad_exceptions.py" in pr_ready, (
        "broad-exception ratchet is not invoked by pr_ready_check.sh; local enforcement is missing"
    )


def test_pedfix_diagnostic_boundary_has_reviewed_inventory_entry() -> None:
    """The actual diagnostic handler must pass the ratchet with a written review reason."""
    from scripts.validation.check_broad_exceptions import check_against_baseline, inventory

    path = "scripts/validation/measure_pedfix_planners.py"
    entries = inventory(ROOT, (path,))
    baseline = json.loads((ROOT / "scripts/validation/broad_exception_baseline.json").read_text())
    approved = [entry for entry in baseline["entries"] if entry["path"] == path]
    scoped = {
        "entries": approved,
        "counts": {"total": len(approved), "by_path": {path: len(approved)} if approved else {}},
    }
    assert check_against_baseline(entries, scoped) == []
    assert len(approved) == 1
    assert approved[0]["review_reason"]


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("adapter unavailable"), LookupError("plugin defect"), KeyboardInterrupt()],
)
def test_pedfix_diagnostic_boundary_records_failures_and_propagates_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
) -> None:
    """Real probe bytes preserve failed dev cells without simulating or swallowing interruption."""
    roster_path = (
        tmp_path
        / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
    )
    roster_path.parent.mkdir(parents=True)
    roster_path.write_text("planners:\n- key: goal\n  algo: goal\n  benchmark_profile: test\n")
    calls = []

    def episode(scenario, seed, **kwargs):
        calls.append((scenario["name"], seed))
        if len(calls) == 1:
            raise failure
        return {"success": True}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["measure_pedfix_planners.py", "--mode", "roster"])
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "fixture-sha\n")
    stubs = {
        "robot_sf.benchmark.map_runner.map_runner": {
            "build_map_policy": lambda *args, **kwargs: None
        },
        "robot_sf.benchmark.map_runner.map_runner_env": {
            "build_env_config": lambda *args, **kwargs: None
        },
        "robot_sf.benchmark.map_runner.map_runner_episode": {"run_map_episode": episode},
        "robot_sf.training.scenario_loader": {
            "load_scenarios": lambda *args: [
                {"name": "classic_head_on_corridor_medium"},
                {"name": "francis2023_circular_crossing"},
            ]
        },
    }
    for name, attributes in stubs.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attributes))
    probe = ROOT / "scripts/validation/measure_pedfix_planners.py"
    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            runpy.run_path(str(probe), run_name="__main__")
        assert len(calls) == 1
        assert capsys.readouterr().out == ""
    else:
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_path(str(probe), run_name="__main__")
        assert exit_info.value.code == 1
        captured = capsys.readouterr()
        rows = [json.loads(line) for line in captured.out.splitlines()]
        assert [row["status"] for row in rows] == ["error", "ok", "ok", "ok"]
        assert rows[0]["error"] == repr(failure)
        assert "record" not in rows[0]
        assert type(failure).__name__ in captured.err
        assert [seed for _, seed in calls] == [1001, 1002, 1001, 1002]
