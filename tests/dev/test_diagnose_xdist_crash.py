"""Tests for the pytest-xdist crash diagnostic (scripts/dev, issue #5633).

Covers signature classification, runtime fingerprinting, safe-concurrency
recommendation, and the rendered fail-closed message. Pure logic is tested at
CPU level; the CLI --help path and JSON output are exercised via subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.dev import diagnose_xdist_crash as dxc

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "diagnose_xdist_crash.py"


# --- signature classification ------------------------------------------------


def test_classify_crash_detects_segfault() -> None:
    """A captured segfault should classify as the segfault class."""
    log = "worker 3 crashed\nSegmentation fault (core dumped)\nxdist reported broken pipe\n"
    assert "segfault" in dxc.classify_crash(log)


def test_classify_crash_detects_abort_and_broken_pipe() -> None:
    """Abort + broken-pipe signatures should both be reported."""
    log = "Aborted\nBrokenPipeError: [Errno 32] Broken pipe\n"
    classes = dxc.classify_crash(log)
    assert "abort" in classes
    assert "broken-pipe" in classes


def test_classify_crash_empty_log_returns_no_classes() -> None:
    """An empty or clean log must not invent an environment crash."""
    assert dxc.classify_crash("") == ()
    assert dxc.classify_crash("100 passed in 2.13s") == ()


def test_classify_crash_preserves_signature_order() -> None:
    """Reported classes should follow the canonical signature ordering."""
    log = "broken pipe\nAborted\nworker 3 crashed\nSegmentation fault\n"
    classes = dxc.classify_crash(log)
    assert classes == ("segfault", "abort", "xdist-worker-crash", "broken-pipe")


def test_first_matching_lines_returns_excerpts() -> None:
    """Excerpts should surface the crash-bearing lines for triage."""
    log = "collecting...\nSegmentation fault (core dumped)\ntidying up\n"
    excerpts = dxc.first_matching_lines(log)
    assert len(excerpts) >= 1
    assert "Segmentation fault" in excerpts[0]


def test_first_matching_lines_capped() -> None:
    """Excerpt extraction should respect the max-line cap."""
    log = "\n".join(f"Segmentation fault {i}" for i in range(10))
    assert len(dxc.first_matching_lines(log, max_lines=3)) == 3


# --- recommendation ----------------------------------------------------------


def test_recommend_serial_for_environment_crash() -> None:
    """Native-extension crashes should recommend a serial rerun."""
    assert dxc.recommend_safe_concurrency(["segfault"]) == "serial"
    assert dxc.recommend_safe_concurrency(["abort", "broken-pipe"]) == "serial"


def test_recommend_reduce_or_serial_for_broken_pipe_only() -> None:
    """A broken-pipe-only signature suggests reducing workers or serial."""
    assert dxc.recommend_safe_concurrency(["broken-pipe"]) == "reduce-or-serial"


def test_recommend_none_for_ordinary_failure() -> None:
    """An ordinary assertion failure carries no concurrency recommendation."""
    assert dxc.recommend_safe_concurrency([]) is None


# --- runtime snapshot --------------------------------------------------------


def test_snapshot_runtime_records_python_and_cpus(monkeypatch) -> None:
    """The snapshot captures Python version, platform, and CPU count."""
    monkeypatch.setattr(dxc.os, "cpu_count", lambda: 8)
    snap = dxc.snapshot_runtime(extension_names=(), cpu_count=None)
    assert snap.logical_cpus == 8
    assert snap.python_version
    assert snap.platform


def test_snapshot_runtime_omits_missing_extensions(monkeypatch) -> None:
    """A dependency-minimal host produces an extension-free snapshot, no raise."""
    monkeypatch.setattr(dxc.importlib.util, "find_spec", lambda _name: None)
    snap = dxc.snapshot_runtime(extension_names=("numpy", "scipy"), cpu_count=4)
    assert snap.extensions == ()


def test_snapshot_runtime_detects_compiled_extension(monkeypatch) -> None:
    """A compiled module should be flagged compiled=True with its version."""

    class _Spec:
        origin = "numpy/_core/_multiarray_umath.cpython-313-x86_64-linux-gnu.so"

    class _Dist:
        version = "2.1.0"

    monkeypatch.setattr(dxc.importlib.util, "find_spec", lambda _name: _Spec())
    monkeypatch.setattr(dxc.importlib.metadata, "version", lambda _name: _Dist.version)
    snap = dxc.snapshot_runtime(extension_names=("numpy",), cpu_count=4)
    assert len(snap.extensions) == 1
    ext = snap.extensions[0]
    assert ext.name == "numpy"
    assert ext.version == "2.1.0"
    assert ext.compiled is True


# --- diagnostic assembly -----------------------------------------------------


def test_build_diagnostic_environment_crash_is_environmental() -> None:
    """An env crash must expose is_environment_crash for the fail-closed gate."""
    diag = dxc.build_diagnostic(
        log_text="Segmentation fault (core dumped)\nworker 1 crashed\n",
        requested_workers="auto",
        dist_mode="load",
        runtime=dxc.RuntimeSnapshot("3.13.0", "Linux", "x86_64", 32),
        serialized_ok=None,
    )
    assert diag.is_environment_crash is True
    assert "segfault" in diag.crash_classes
    assert diag.recommendation == "serial"


def test_build_diagnostic_ordinary_failure_is_not_environmental() -> None:
    """A clean (non-crash) log must not be classified as an environment crash."""
    diag = dxc.build_diagnostic(
        log_text="2 failed, 98 passed in 3.21s",
        runtime=dxc.RuntimeSnapshot("3.13.0", "Linux", "x86_64", 32),
    )
    assert diag.is_environment_crash is False
    assert diag.crash_classes == ()
    assert diag.recommendation is None


def test_serialized_ok_true_text_reflects_env_only_crash() -> None:
    """When serial rerun passed, the message must say the crash was env-only."""
    diag = dxc.build_diagnostic(
        log_text="Segmentation fault (core dumped)\n",
        runtime=dxc.RuntimeSnapshot("3.13.0", "Linux", "x86_64", 32),
        serialized_ok=True,
    )
    rendered = dxc.render_diagnostic(diag)
    assert "serial" in rendered.lower()
    assert "environment-only" in rendered or "env-only" in rendered.lower()


def test_serialized_ok_false_text_reflects_real_failures() -> None:
    """When serial rerun failed, the message must flag real failures to fix."""
    diag = dxc.build_diagnostic(
        log_text="Segmentation fault (core dumped)\n",
        runtime=dxc.RuntimeSnapshot("3.13.0", "Linux", "x86_64", 32),
        serialized_ok=False,
    )
    rendered = dxc.render_diagnostic(diag)
    assert "real" in rendered.lower()
    assert "failure" in rendered.lower()


def test_render_diagnostic_includes_runtime_fingerprint_and_issue() -> None:
    """The rendered diagnostic should cite issue #5633 and the runtime."""
    diag = dxc.build_diagnostic(
        log_text="Aborted\nSIGABRT\n",
        requested_workers="8",
        dist_mode="worksteal",
        runtime=dxc.RuntimeSnapshot(
            "3.12.3",
            "Linux",
            "x86_64",
            16,
            extensions=(dxc.NativeExtensionInfo("numpy", "1.26.0", True),),
        ),
    )
    rendered = dxc.render_diagnostic(diag)
    assert "5633" in rendered
    assert "3.12.3" in rendered
    assert "numpy 1.26.0" in rendered
    assert "fail-closed" in rendered


def test_render_diagnostic_no_crash_still_actionable() -> None:
    """With no crash signature, the message must not claim an env crash."""
    diag = dxc.build_diagnostic(
        log_text="100 passed in 2.00s",
        runtime=dxc.RuntimeSnapshot("3.13.0", "Linux", "x86_64", 32),
    )
    rendered = dxc.render_diagnostic(diag)
    assert "No native-extension/xdist crash signature" in rendered
    assert "ordinary" in rendered.lower()


# --- CLI ---------------------------------------------------------------------


def test_cli_help_exits_zero() -> None:
    """``--help`` must be a cheap exit-0 path printing usage (CI contract)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_cli_classifies_segfault_from_inline_text() -> None:
    """Inline --log-text should classify and render fail-closed guidance."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--log-text",
            "Segmentation fault (core dumped)\nworker 2 crashed\n",
            "--requested-workers",
            "auto",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "segfault" in result.stdout
    assert "5633" in result.stdout


def test_cli_json_output_is_parsable() -> None:
    """``--json`` must emit a parsable diagnostic with crash classes."""
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--json",
            "--log-text",
            "Aborted\nBrokenPipeError\n",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "abort" in payload["crash_classes"]
    assert payload["is_environment_crash"] is True
    assert "runtime" in payload


def test_cli_reads_log_file(tmp_path) -> None:
    """``--log-file`` should read the captured log from disk."""
    log_file = tmp_path / "pytest.log"
    log_file.write_text("SIGSEGV in numpy import\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--log-file", str(log_file)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "segfault" in result.stdout.lower()
