"""Behavior tests for the receipt-owner compatibility wrapper (issue #8447)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GH_PR_MERGE = REPO_ROOT / "scripts" / "dev" / "gh_pr_merge.sh"

FULL_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"


def _fake_python_bin(tmp_path: Path) -> Path:
    """Write a fake Python launcher that records delegated policy/receipt calls."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "plan = json.loads(pathlib.Path(os.environ['FAKE_PYTHON_PLAN']).read_text())\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['-m', 'scripts.dev.single_account_merge_receipt']:\n"
        "    script = 'single_account_merge_receipt.py'\n"
        "    delegated_args = args[2:]\n"
        "else:\n"
        "    script = pathlib.Path(args[0]).name if args else ''\n"
        "    delegated_args = args[1:]\n"
        "log_path = pathlib.Path(os.environ['FAKE_PYTHON_LOG'])\n"
        "with log_path.open('a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps({'script': script, 'args': delegated_args}) + '\\n')\n"
        "if script == 'github_transport_policy.py':\n"
        "    response = plan.get('policy', {})\n"
        "elif script == 'single_account_merge_receipt.py':\n"
        "    mode_index = delegated_args.index('--mode') + 1\n"
        "    response = plan.get(delegated_args[mode_index], {})\n"
        "    output_index = delegated_args.index('--output') + 1 if '--output' in delegated_args else None\n"
        "    if output_index is not None and response.get('write_output', True):\n"
        "        pathlib.Path(delegated_args[output_index]).write_text(\n"
        "            response.get('receipt', '{\\\"status\\\": \\\"ready\\\"}'),\n"
        "            encoding='utf-8',\n"
        "        )\n"
        "else:\n"
        "    print(f'unexpected delegated script: {script}', file=sys.stderr)\n"
        "    sys.exit(99)\n"
        "if response.get('stdout'):\n"
        "    print(response['stdout'])\n"
        "if response.get('stderr'):\n"
        "    print(response['stderr'], file=sys.stderr)\n"
        "sys.exit(response.get('exit', 0))\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return bin_dir


def _failing_gh_bin(tmp_path: Path) -> Path:
    """Write a real executable used to stop the live receipt read safely."""
    bin_dir = tmp_path / "real-bin"
    bin_dir.mkdir()
    fake_gh = bin_dir / "gh"
    fake_gh.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "pathlib.Path(os.environ['FAKE_GH_LOG']).write_text(os.getcwd(), encoding='utf-8')\n"
        "print('fake gh failure', file=sys.stderr)\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    return bin_dir


def _run_wrapper(
    tmp_path: Path,
    plan: dict[str, object] | None = None,
    *,
    include_repo_arg: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the wrapper with policy and receipt-owner subprocesses under test control."""
    plan_path = tmp_path / "fake_python_plan.json"
    log_path = tmp_path / "fake_python_calls.jsonl"
    plan_path.write_text(json.dumps(plan or {}), encoding="utf-8")
    log_path.write_text("", encoding="utf-8")
    env = os.environ.copy()
    env["FAKE_PYTHON_PLAN"] = str(plan_path)
    env["FAKE_PYTHON_LOG"] = str(log_path)
    env["PATH"] = str(_fake_python_bin(tmp_path)) + os.pathsep + env.get("PATH", "")
    args = [str(GH_PR_MERGE), "1234", "--match-head-commit", FULL_SHA]
    if include_repo_arg:
        args.extend(("--repo", "o/r"))
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )


def _calls(tmp_path: Path) -> list[dict[str, object]]:
    """Read the delegated subprocess calls recorded by the fake launcher."""
    return [
        json.loads(line)
        for line in (tmp_path / "fake_python_calls.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


def _successful_plan() -> dict[str, object]:
    return {
        "policy": {"exit": 0},
        "report-only": {"exit": 0, "stdout": "receipt report"},
        "apply": {"exit": 0, "stdout": "receipt applied"},
    }


def test_help_is_safe_without_transport_or_receipt_execution(tmp_path: Path) -> None:
    """Help must not invoke policy, repository discovery, or the receipt owner."""
    plan_path = tmp_path / "fake_python_plan.json"
    log_path = tmp_path / "fake_python_calls.jsonl"
    plan_path.write_text(json.dumps({"policy": {"exit": 99}}), encoding="utf-8")
    log_path.write_text("", encoding="utf-8")
    env = os.environ.copy()
    env["FAKE_PYTHON_PLAN"] = str(plan_path)
    env["FAKE_PYTHON_LOG"] = str(log_path)
    env["PATH"] = str(_fake_python_bin(tmp_path)) + os.pathsep + env.get("PATH", "")

    result = subprocess.run(
        [str(GH_PR_MERGE), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert "receipt owner" in result.stdout
    assert log_path.read_text(encoding="utf-8") == ""


def test_wrapper_delegates_report_and_apply_with_exact_head(tmp_path: Path) -> None:
    """The shell compatibility path delegates both phases with the same binding."""
    result = _run_wrapper(tmp_path, _successful_plan())

    assert result.returncode == 0, result.stderr
    calls = _calls(tmp_path)
    assert [call["script"] for call in calls] == [
        "github_transport_policy.py",
        "single_account_merge_receipt.py",
        "single_account_merge_receipt.py",
    ]
    report_args = calls[1]["args"]
    apply_args = calls[2]["args"]
    assert isinstance(report_args, list)
    assert isinstance(apply_args, list)
    assert report_args[report_args.index("--mode") + 1] == "report-only"
    assert apply_args[apply_args.index("--mode") + 1] == "apply"
    for delegated_args in (report_args, apply_args):
        assert delegated_args[delegated_args.index("--expected-head") + 1] == FULL_SHA
        assert delegated_args[delegated_args.index("--repo") + 1] == "o/r"


def test_wrapper_runs_receipt_owner_as_a_module_with_the_real_interpreter(
    tmp_path: Path,
) -> None:
    """The shell path must import the receipt package before any live API read."""
    env = os.environ.copy()
    env["FAKE_GH_LOG"] = str(tmp_path / "fake-gh-cwd")
    env["PATH"] = str(_failing_gh_bin(tmp_path)) + os.pathsep + env.get("PATH", "")

    result = subprocess.run(
        [str(GH_PR_MERGE), "1234", "--match-head-commit", FULL_SHA, "--repo", "o/r"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )

    assert result.returncode == 1
    assert "ModuleNotFoundError: No module named 'scripts'" not in result.stderr
    assert "canonical receipt report was not ready" in result.stderr
    assert (tmp_path / "fake-gh-cwd").read_text(encoding="utf-8") == str(REPO_ROOT)


def test_report_block_blocks_apply(tmp_path: Path) -> None:
    """A blocked or unavailable report cannot be followed by an apply attempt."""
    result = _run_wrapper(
        tmp_path,
        {
            "policy": {"exit": 0},
            "report-only": {"exit": 1, "stderr": "receipt blocked"},
        },
    )

    assert result.returncode == 1
    assert "report was not ready" in result.stderr
    assert [call["script"] for call in _calls(tmp_path)] == [
        "github_transport_policy.py",
        "single_account_merge_receipt.py",
    ]


def test_apply_failure_is_not_reported_as_success(tmp_path: Path) -> None:
    """The wrapper preserves a nonzero owner result and never degrades it to success."""
    result = _run_wrapper(
        tmp_path,
        {
            "policy": {"exit": 0},
            "report-only": {"exit": 0},
            "apply": {"exit": 7, "stderr": "live closing evidence changed"},
        },
    )

    assert result.returncode == 7
    assert "refused or failed" in result.stderr
    assert len(_calls(tmp_path)) == 3


def test_wrapper_resolves_repository_from_github_origin(tmp_path: Path) -> None:
    """Origin discovery remains read-only and passes the resolved repository to the owner."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", "git@github.com:o/r.git"],
        check=True,
    )

    result = _run_wrapper(tmp_path, _successful_plan(), include_repo_arg=False, cwd=checkout)

    assert result.returncode == 0, result.stderr
    calls = _calls(tmp_path)
    for call in calls[1:]:
        delegated_args = call["args"]
        assert isinstance(delegated_args, list)
        assert delegated_args[delegated_args.index("--repo") + 1] == "o/r"


def test_wrapper_rejects_non_github_origin_before_delegation(tmp_path: Path) -> None:
    """An origin from another host cannot silently redirect a merge request."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", "git@gitlab.com:o/r.git"],
        check=True,
    )

    result = _run_wrapper(tmp_path, _successful_plan(), include_repo_arg=False, cwd=checkout)

    assert result.returncode == 2
    assert "cannot resolve owner/name" in result.stderr
    assert [call["script"] for call in _calls(tmp_path)] == ["github_transport_policy.py"]


@pytest.mark.parametrize(
    "args",
    (
        ["1234", "--match-head-commit"],
        ["1234", "--match-head-commit", "not-a-sha"],
        ["0", "--match-head-commit", FULL_SHA],
    ),
)
def test_wrapper_rejects_malformed_or_unsafe_bindings(tmp_path: Path, args: list[str]) -> None:
    """Malformed PR or head arguments fail before any transport or owner call."""
    env = os.environ.copy()
    env["PATH"] = str(_fake_python_bin(tmp_path)) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [str(GH_PR_MERGE), *args, "--repo", "o/r"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )

    assert result.returncode == 2
