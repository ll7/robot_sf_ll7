"""Tests for the bounded offline core-workflow smoke harness."""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.dev import run_offline_smoke as smoke

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/dev/run_offline_smoke.py"


def _run_smoke(output_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--offline",
            "--isolated",
            "--json",
            "--output-root",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_successful_smoke_writes_path_free_deterministic_receipt(tmp_path: Path) -> None:
    output_root = tmp_path / "smoke"
    result = _run_smoke(output_root)

    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt == json.loads((output_root / smoke.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["schema"] == smoke.SCHEMA
    assert receipt["mode"] == smoke.MODE
    assert receipt["status"] == smoke.STATUS_PASSED
    assert [row["id"] for row in receipt["capabilities"]] == list(smoke.ALL_CAPABILITIES)
    assert all(
        row["status"] == smoke.STATUS_PASSED
        for row in receipt["capabilities"][: len(smoke.CORE_CAPABILITIES)]
    )
    optional = {
        row["id"]: row["reason_code"] for row in receipt["capabilities"] if not row["required"]
    }
    assert list(optional) == smoke.OPTIONAL_CAPABILITIES
    assert list(optional.values()) == [
        smoke.REASONS[key]
        for key in "scheduler gpu carla private_data institutional model_empty network".split()
    ]
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_digest")
    assert digest == hashlib.sha256(smoke._compact_json(unsigned).encode()).hexdigest()
    receipt_text = json.dumps(receipt, sort_keys=True)
    assert str(REPO_ROOT) not in receipt_text
    assert str(tmp_path) not in receipt_text
    assert "/home/" not in receipt_text
    assert all((output_root / relative).is_file() for relative in receipt["outputs"])


def test_negative_optional_capability_fixtures(tmp_path: Path) -> None:
    assert smoke._probe_scheduler("").reason_code == smoke.REASONS["scheduler"]
    assert (
        smoke._probe_gpu({"CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void"}).reason_code
        == smoke.REASONS["gpu"]
    )
    cache = tmp_path / "empty-cache"
    cache.mkdir()
    assert smoke._probe_model_cache(cache).reason_code == smoke.REASONS["model_empty"]
    assert (
        smoke._probe_model_cache(cache, tmp_path / "missing-model.zip").reason_code
        == (smoke.REASONS["model"])
    )
    assert (
        smoke._probe_private_data({"ROBOT_SF_PRIVATE_DATA": "redacted"}).reason_code
        == (smoke.REASONS["private_exposed"])
    )
    assert (
        smoke._probe_institutional_context({"SLURM_JOB_ID": "redacted"}).reason_code
        == (smoke.REASONS["institutional_exposed"])
    )


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux loader contract")
def test_child_environment_uses_only_active_python_library_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tool-cache Python children can start without inheriting arbitrary loader paths."""
    lib_dir = tmp_path / "python" / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / f"libpython{sys.version_info.major}.{sys.version_info.minor}.so.1.0").touch()
    monkeypatch.setattr(sys, "base_prefix", str(lib_dir.parent))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/untrusted/ambient/path")
    output_root, manager = smoke._prepare_root(REPO_ROOT, tmp_path / "smoke")
    try:
        environment, _ = smoke._controlled_environment(output_root, REPO_ROOT)
        assert environment["LD_LIBRARY_PATH"] == str(lib_dir)
    finally:
        if manager is not None:
            manager.cleanup()


def test_negative_path_root_and_network_fixtures(tmp_path: Path) -> None:
    leaky = tmp_path / "leaky.txt"
    leaky.write_text("/home/example/private-ops/secret\n", encoding="utf-8")
    with pytest.raises(smoke.SmokeFailure) as leakage:
        smoke._check_output_tree(tmp_path, {})
    assert leakage.value.reason_code == smoke.REASONS["path_leak"]

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "foreign.txt").write_text("fixture", encoding="utf-8")
    with pytest.raises(smoke.SmokeFailure) as ownership:
        smoke._fresh_root(REPO_ROOT, occupied)
    assert ownership.value.reason_code == smoke.REASONS["root_not_owned"]
    with smoke._network_denied(), pytest.raises(OSError, match="network disabled"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)


def test_cli_help_keeps_heavy_modules_unloaded(tmp_path: Path) -> None:
    """Exercise the installed CLI entry path with the smoke's isolated environment."""
    root, manager = smoke._prepare_root(REPO_ROOT, tmp_path / "help-root")
    try:
        environment, _ = smoke._controlled_environment(root, REPO_ROOT)
        result = smoke._run_child(
            REPO_ROOT,
            environment,
            """
import sys
from robot_sf.cli import main
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
heavy = ('torch', 'stable_baselines3', 'tensorflow', 'pygame',
         'robot_sf.sim', 'robot_sf.gym_env', 'robot_sf.render')
loaded = sorted(name for name in sys.modules
                if any(name == prefix or name.startswith(prefix + '.') for prefix in heavy))
assert not loaded, 'Help imported heavy modules: ' + ', '.join(loaded)
""",
        )
        assert result.returncode == 0, result.stderr
        assert "usage: robot-sf" in result.stdout
    finally:
        if manager is not None:
            manager.cleanup()


def test_child_timeout_receipt_has_elapsed_and_partial_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real hung child remains required and records its timeout evidence."""
    monkeypatch.setattr(smoke, "CHILD_TIMEOUT_SECONDS", 1.0)
    row = smoke._stage(
        "cli_help",
        lambda: smoke._run_child(
            REPO_ROOT,
            {},
            "import sys, time; print('starting help', file=sys.stderr, flush=True); time.sleep(5)",
        ),
        smoke.REASONS["cli_help"],
    ).to_dict()
    assert row["required"] is True
    assert row["status"] == "failed"
    assert row["reason_code"] == "CHILD_TIMEOUT"
    assert row["elapsed_seconds"] >= 1.0
    assert row["stderr"] == "starting help\n"


def test_child_oserror_receipt_preserves_diagnostic_without_paths(
    tmp_path: Path,
) -> None:
    """An actual failed spawn records errno and sanitized error text in JSON."""
    capability = smoke._stage(
        "cli_help",
        lambda: smoke._run_child(REPO_ROOT, {}, "pass", cwd=tmp_path / "missing"),
        smoke.REASONS["cli_help"],
    )
    row = capability.to_dict()
    assert row["reason_code"] == "CHILD_OSERROR"
    assert row["required"] is True
    assert row["status"] == "failed"
    assert row["elapsed_seconds"] >= 0
    assert row["errno"] == 2
    assert "No such file or directory" in row["stderr"]
    assert str(tmp_path) not in json.dumps(row)
    rows = [smoke.Capability(name, True, "passed", "OK") for name in smoke.ALL_CAPABILITIES]
    rows[1] = capability
    receipt = smoke._build_receipt(rows)
    assert receipt["capabilities"][1] == row
    assert receipt["status"] == "failed"
