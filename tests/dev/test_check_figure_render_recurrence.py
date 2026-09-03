"""Tests for the isolated figure-render recurrence guard (issue #6770).

Covers the recurrence contract end-to-end with synthetic fixtures (NOT the real nine commands):
success, timeout, missing output, extra output, path escape, input drift, unsafe command,
negative-control handling, empty-eligible fail-closed, structural validation, the strace
write-tracker, output redirection, stable registry-order reporting, workflow trigger coverage, and
portable report provenance.

The real committed registry is also exercised once as a regression smoke (mirrors the CI guard
command), proving the nine recurrence-eligible entries reproduce under isolation.
"""

# evidence-writer-exempt: these tests write to pytest tmp_path fixtures (local scratch), not to
# tracked evidence artifacts.

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import yaml

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "dev" / "check_figure_render_recurrence.py"
)
REGISTRY = (
    Path(__file__).resolve().parents[2] / "docs" / "context" / "figure_render_registry.v1.yaml"
)
WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "figure-render-recurrence.yml"
)

sys.path.insert(0, str(SCRIPT.parent))
import check_figure_render_recurrence as guard  # noqa: E402

# The guard fails closed without strace (Linux-only), so every test that reaches
# ``run_recurrence``'s strace gate can only pass where strace exists. Linux CI
# (.github/workflows/figure-render-recurrence.yml) installs strace and arbitrates;
# on macOS dev machines these tests skip instead of failing pr_ready_check.
requires_strace = pytest.mark.skipif(
    shutil.which("strace") is None,
    reason="strace is Linux-only and required for the guard's write containment",
)

# ---------------------------------------------------------------------------
# Helpers: synthetic registries and writer scripts
# ---------------------------------------------------------------------------


def _make_registry(tmp_path: Path, entries: list[dict]) -> Path:
    """Write a minimal v1 registry with the given entries and return its path."""
    payload = {
        "version": 1,
        "generated_at": "2026-08-09T00:00:00+00:00",
        "generator": "synthetic-test",
        "issue": 6770,
        "claim_boundary": "synthetic test registry",
        "provenance": {"source_commit": None, "issue": 6770},
        "entries": entries,
    }
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _entry(
    eid: str,
    command: list[str],
    expected_outputs: list[str],
    *,
    inputs: list[dict] | None = None,
    timeout: int = 60,
    eligible: bool = True,
    reason: str | None = None,
    working_dir: str = ".",
) -> dict:
    """Build a structurally valid registry entry for a synthetic command."""
    return {
        "id": eid,
        "source_class": "synthetic",
        "source_path": "synthetic/README.md",
        "command_index": 0,
        "command": command,
        "working_dir": working_dir,
        "inputs": inputs or [],
        "expected_outputs": expected_outputs,
        "timeout_seconds": timeout,
        "environment": {},
        "recurrence_eligible": eligible,
        "exclusion_reason": reason,
        "classification_notes": [],
        "provenance_status": "discovered",
        "last_verified_commit": None,
    }


def _writer_script(tmp_path: Path, name: str, body: str) -> Path:
    """Write a synthetic Python command script under ``tmp_path`` and return its path."""
    script = tmp_path / name
    script.write_text(body, encoding="utf-8")
    return script


def _run(tmp_path: Path, registry: Path, *, zero_entry_policy: str | None = None):
    """Run the guard in-process against a registry, writing the report to tmp_path."""
    report_path = tmp_path / "report.json"
    with warnings.catch_warnings():
        # write_evidence_json warns (benign) when --report is outside the evidence tree.
        warnings.simplefilter("ignore", UserWarning)
        report, exit_code = guard.run_recurrence(registry, report_path, zero_entry_policy)
    return report, exit_code, report_path


def _network_namespace_prefix() -> list[str] | None:
    """Return a usable fail-closed network-namespace command prefix, if available."""
    if sys.platform != "linux":
        return None
    unshare = shutil.which("unshare")
    if unshare is None:
        return None
    user_prefix = [unshare, "--user", "--map-root-user", "--net", "--"]
    probe = subprocess.run([*user_prefix, "true"], capture_output=True, text=True, check=False)
    if probe.returncode == 0:
        return user_prefix
    sudo = shutil.which("sudo")
    if sudo is None:
        return None
    sudo_probe = subprocess.run([sudo, "-n", "true"], capture_output=True, text=True, check=False)
    if sudo_probe.returncode == 0 and Path("/usr/bin/unshare").is_file():
        return [sudo, "-n", "/usr/bin/unshare", "--net", "--"]
    return None


def _out_cmd(script: Path, declared_output: str, extra_args: list[str] | None = None) -> list[str]:
    """Build a relative argv that runs a writer script with ``--out <declared>``.

    Real registry commands are relative token vectors (``uv run python scripts/...``); synthetic
    fixtures mirror that by invoking ``python3 <script-name>`` from the entry's working directory so
    no absolute token trips the safety re-check.
    """
    cmd = ["python3", script.name, "--out", declared_output]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _entry_in(
    tmp_path: Path,
    eid: str,
    command: list[str],
    expected_outputs: list[str],
    *,
    inputs: list[dict] | None = None,
    timeout: int = 60,
    eligible: bool = True,
    reason: str | None = None,
) -> dict:
    """Build a structurally valid synthetic entry that runs from ``tmp_path``."""
    return _entry(
        eid,
        command,
        expected_outputs,
        inputs=inputs,
        timeout=timeout,
        eligible=eligible,
        reason=reason,
        working_dir=str(tmp_path),
    )


# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


def test_build_minimal_env_pins_current_checkout_sources(monkeypatch):
    """The sanitized child environment must not select a stale installed checkout."""
    monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted-caller-path")

    env = guard.build_minimal_env()

    assert env["PYTHONPATH"] == os.pathsep.join(
        (str(guard.REPO_ROOT.resolve()), str((guard.REPO_ROOT / "fast-pysf").resolve()))
    )


def test_re_safety_flags_unsafe_commands():
    assert guard.re_safety_check(["sh", "-c", "a && b"]) == "unsafe_command"
    assert guard.re_safety_check(["ENV=1", "uv", "run", "python", "x.py"]) == "unsafe_command"
    assert guard.re_safety_check(["sbatch", "run.sh", "--out", "o.json"]) == "requires_slurm"
    assert (
        guard.re_safety_check(["python", "fetch.py", "https://example.com/x"]) == "requires_network"
    )
    assert (
        guard.re_safety_check(["python", "render.py", "--out", "/tmp/abs/o.json"])
        == "unsafe_command"
    )


def test_re_safety_accepts_clean_command():
    assert (
        guard.re_safety_check(["uv", "run", "python", "scripts/benchmark/x.py", "--out", "o.json"])
        is None
    )


def test_rewrite_outputs_redirects_exact_and_descendant_tokens():
    cmd = ["python", "x.py", "--output-dir", "docs/a", "--leaf", "docs/a/sub.txt"]
    rewritten, err = guard.rewrite_outputs(cmd, ["docs/a"], Path("/tmp/root"))
    assert err is None
    assert rewritten[3] == "/tmp/root/docs/a"
    assert rewritten[5] == "/tmp/root/docs/a/sub.txt"


def test_rewrite_outputs_fails_when_output_not_a_token():
    # The declared output never appears as a command token, so it cannot be redirected.
    cmd = ["python", "x.py", "--out", "somewhere/else.json"]
    _rewritten, err = guard.rewrite_outputs(cmd, ["docs/committed/o.json"], Path("/tmp/root"))
    assert err == "output_not_redirectable:docs/committed/o.json"


def test_verify_input_hashes_detects_drift_and_missing(tmp_path):
    fixture = tmp_path / "input.json"
    fixture.write_text("payload", encoding="utf-8")
    digest = hashlib.sha256(b"payload").hexdigest()
    entry_ok = {"inputs": [{"path": str(fixture), "sha256": digest}]}
    assert guard.verify_input_hashes(entry_ok) is None

    entry_drift = {"inputs": [{"path": str(fixture), "sha256": "deadbeef" * 8}]}
    assert guard.verify_input_hashes(entry_drift).startswith("input_drift:")

    entry_missing = {"inputs": [{"path": str(tmp_path / "nope.json"), "sha256": digest}]}
    assert "missing" in guard.verify_input_hashes(entry_missing)


def test_validate_registry_structure_accepts_valid_and_rejects_invalid(tmp_path):
    valid = [_entry("e1", ["python", "x.py", "--out", "o.json"], ["o.json"])]
    assert guard.validate_registry_structure({"version": 1, "entries": valid}) == []

    bad_version = guard.validate_registry_structure({"version": 2, "entries": valid})
    assert any("version" in e for e in bad_version)

    eligible_with_reason = _entry(
        "e1", ["python", "x.py", "--out", "o.json"], ["o.json"], reason="unsafe_command"
    )
    eligible_with_reason["recurrence_eligible"] = True
    errs = guard.validate_registry_structure({"version": 1, "entries": [eligible_with_reason]})
    assert any("exclusion_reason" in e for e in errs)

    bad_reason = _entry("e1", ["python", "x.py"], [], eligible=False, reason="bogus_reason")
    errs = guard.validate_registry_structure({"version": 1, "entries": [bad_reason]})
    assert any("non-canonical" in e for e in errs)


def test_collect_write_targets_parses_successful_writes_only():
    trace = tmp_or_cwd_trace()
    targets = guard.collect_write_targets(trace, Path("/cwd"))
    resolved = {str(p) for p in targets}
    assert "/cwd/out/x" in resolved
    assert "/cwd/out/sub" in resolved
    assert "/cwd/gone" in resolved
    assert "/cwd/a" in resolved  # rename source (removed)
    assert "/cwd/b" in resolved  # rename destination (created)
    # Reads and failed syscalls must NOT count as writes.
    assert "/etc/foo" not in resolved
    assert "/cwd/exists" not in resolved  # EEXIST mkdir
    assert "/cwd/perm" not in resolved  # EPERM unlink


def tmp_or_cwd_trace() -> Path:
    """Return an in-memory strace trace file exercising the write/no-write cases."""
    import tempfile

    lines = [
        '100 openat(AT_FDCWD, "/cwd/out/x", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 3',
        '100 openat(AT_FDCWD, "/etc/foo", O_RDONLY|O_CLOEXEC) = 4',
        '100 mkdir("/cwd/out/sub", 0755) = 0',
        '100 mkdir("/cwd/exists", 0777) = -1 EEXIST (File exists)',
        '100 unlink("/cwd/gone") = 0',
        '100 unlink("/cwd/perm") = -1 EPERM (Operation not permitted)',
        '100 rename("/cwd/a", "/cwd/b") = 0',
        "100 --- SIGCHLD ---",
        "100 +++ exited with 0 +++",
    ]
    handle = tempfile.NamedTemporaryFile("w", suffix=".strace", delete=False, encoding="utf-8")
    handle.write("\n".join(lines) + "\n")
    handle.close()
    return Path(handle.name)


def test_classify_writes_flags_extra_and_escape_but_not_benign(tmp_path):
    temp_root = (tmp_path / "outputs").resolve()
    temp_root.mkdir()
    declared = [temp_root / "report.json"]
    benign_roots = [(tmp_path / "cache").resolve()]
    (tmp_path / "cache").mkdir()
    patterns = guard.BENIGN_WRITE_PATTERNS
    declared_output = temp_root / "report.json"
    declared_output.write_text("ok")
    extra = temp_root / "sibling.json"
    extra.write_text("extra")
    escape = (tmp_path / "escape.txt").resolve()
    escape.write_text("escaped")
    benign = (tmp_path / "cache" / "uv.lock").resolve()
    benign.write_text("cache")
    bytecode = (tmp_path / "mod.cpython-313.pyc").resolve()
    bytecode.write_text("pyc")
    targets = {declared_output, extra, escape, benign, bytecode}
    extra_out, escapes = guard.classify_writes(targets, temp_root, declared, benign_roots, patterns)
    assert str(extra) in extra_out
    assert str(escape) in escapes
    assert str(benign) not in escapes
    assert str(bytecode) not in escapes
    assert str(declared_output) not in extra_out


# ---------------------------------------------------------------------------
# End-to-end synthetic command tests
# ---------------------------------------------------------------------------


def _writer_body(*, escape: bool = False, extra: bool = False, write_output: bool = True) -> str:
    """Build a synthetic writer script body honoring the requested misbehavior."""
    parts = [
        "import sys, pathlib",
        "args = sys.argv[1:]",
        "out = args[args.index('--out') + 1]",
        "p = pathlib.Path(out)",
    ]
    if write_output:
        parts.append("p.parent.mkdir(parents=True, exist_ok=True)")
        parts.append("p.write_text('ok')")
    if extra:
        parts.append("p.with_name(p.name + '.extra').write_text('extra')")
    if escape:
        # Write outside the temporary output root (a relative canary in the command's working dir).
        parts.append("pathlib.Path('escape_canary.txt').write_text('escaped')")
    parts.append("sys.exit(int(args[args.index('--exit') + 1]) if '--exit' in args else 0)")
    return "\n".join(parts) + "\n"


@requires_strace
def test_success_reproduces(tmp_path):
    script = _writer_script(tmp_path, "ok.py", _writer_body())
    entry = _entry_in(tmp_path, "ok_cmd", _out_cmd(script, "out/report.json"), ["out/report.json"])
    registry = _make_registry(tmp_path, [entry])
    report, exit_code, _ = _run(tmp_path, registry)
    assert exit_code == 0
    assert report["passed_count"] == 1
    assert report["commands"][0]["status"] == "passed"
    assert report["commands"][0]["output_check"] == "ok"
    assert report["commands"][0]["containment_check"] == "ok"


@requires_strace
def test_timeout_fails(tmp_path):
    body = "import sys, time; time.sleep(30); sys.exit(0)\n"
    script = _writer_script(tmp_path, "slow.py", body)
    entry = _entry_in(
        tmp_path, "slow_cmd", _out_cmd(script, "out/report.json"), ["out/report.json"], timeout=3
    )
    registry = _make_registry(tmp_path, [entry])
    report, exit_code, _ = _run(tmp_path, registry)
    assert exit_code == 1
    result = report["commands"][0]
    assert result["status"] == "failed"
    assert result["timed_out"] is True
    assert result["error_reason"] == "timeout"


@requires_strace
def test_missing_output_fails(tmp_path):
    script = _writer_script(tmp_path, "noop.py", _writer_body(write_output=False))
    entry = _entry_in(
        tmp_path, "noop_cmd", _out_cmd(script, "out/report.json"), ["out/report.json"]
    )
    registry = _make_registry(tmp_path, [entry])
    report, exit_code, _ = _run(tmp_path, registry)
    assert exit_code == 1
    result = report["commands"][0]
    assert result["error_reason"] == "missing_output"
    assert result["output_check"].startswith("missing:")


@requires_strace
def test_extra_output_fails(tmp_path):
    script = _writer_script(tmp_path, "extra.py", _writer_body(extra=True))
    entry = _entry_in(
        tmp_path, "extra_cmd", _out_cmd(script, "out/report.json"), ["out/report.json"]
    )
    registry = _make_registry(tmp_path, [entry])
    report, exit_code, _ = _run(tmp_path, registry)
    assert exit_code == 1
    result = report["commands"][0]
    assert result["error_reason"] == "extra_output"
    assert result["output_check"].startswith("extra:")


@requires_strace
def test_path_escape_fails(tmp_path):
    script = _writer_script(tmp_path, "escape.py", _writer_body(escape=True))
    entry = _entry_in(
        tmp_path, "escape_cmd", _out_cmd(script, "out/report.json"), ["out/report.json"]
    )
    registry = _make_registry(tmp_path, [entry])
    report, exit_code, _ = _run(tmp_path, registry)
    assert exit_code == 1
    result = report["commands"][0]
    assert result["error_reason"] == "undeclared_write"
    assert result["containment_check"].startswith("escape:")


@requires_strace
def test_input_drift_fails_without_execution(tmp_path):
    fixture = tmp_path / "input.json"
    fixture.write_text("payload", encoding="utf-8")
    script = _writer_script(tmp_path, "ok.py", _writer_body())
    entry = _entry_in(
        tmp_path,
        "drift_cmd",
        _out_cmd(script, "out/report.json"),
        ["out/report.json"],
        inputs=[{"path": str(fixture), "sha256": "deadbeef" * 8}],
    )
    registry = _make_registry(tmp_path, [entry])
    report, exit_code, _ = _run(tmp_path, registry)
    assert exit_code == 1
    result = report["commands"][0]
    assert result["error_reason"] == "input_drift"
    assert result["output_check"] == "not_run"
    assert result["containment_check"] == "not_run"


@requires_strace
def test_unsafe_command_fails_without_execution(tmp_path):
    unsafe = ["sh", "-c", "echo hi && echo bad", "--out", "out/report.json"]
    entry = _entry_in(tmp_path, "unsafe_cmd", unsafe, ["out/report.json"])
    registry = _make_registry(tmp_path, [entry])
    report, exit_code, _ = _run(tmp_path, registry)
    assert exit_code == 1
    result = report["commands"][0]
    assert result["error_reason"] == "unsafe_command"
    assert result["safety_check"] == "unsafe_command"
    assert result["output_check"] == "not_run"


@requires_strace
def test_undeclared_nonzero_exit_fails(tmp_path):
    script = _writer_script(tmp_path, "fail.py", _writer_body())
    entry = _entry_in(
        tmp_path,
        "fail_cmd",
        _out_cmd(script, "out/report.json", ["--exit", "1"]),
        ["out/report.json"],
    )
    registry = _make_registry(tmp_path, [entry])
    report, exit_code, _ = _run(tmp_path, registry)
    assert exit_code == 1
    result = report["commands"][0]
    assert result["error_reason"] == "nonzero_exit"
    assert result["exit_code"] == 1


@requires_strace
def test_declared_negative_control_passes_on_expected_nonzero(tmp_path, monkeypatch):
    script = _writer_script(tmp_path, "detector.py", _writer_body())
    entry = _entry_in(
        tmp_path,
        "detector_cmd",
        _out_cmd(script, "out/report.json", ["--exit", "1"]),
        ["out/report.json"],
    )
    monkeypatch.setattr(
        guard,
        "NEGATIVE_CONTROLS",
        {"detector_cmd": {"reason": "test negative control", "expected_exit_code": 1}},
    )
    registry = _make_registry(tmp_path, [entry])
    report, exit_code, _ = _run(tmp_path, registry)
    assert exit_code == 0
    result = report["commands"][0]
    assert result["status"] == "passed"
    assert result["negative_control"] is True
    assert result["expected_exit_code"] == 1
    assert result["exit_code"] == 1


@requires_strace
def test_declared_negative_control_fails_on_wrong_exit(tmp_path, monkeypatch):
    # Declared to exit 1, but the command exits 0 (detector failed to detect): must fail.
    script = _writer_script(tmp_path, "ok.py", _writer_body())
    entry = _entry_in(
        tmp_path,
        "detector_cmd",
        _out_cmd(script, "out/report.json"),
        ["out/report.json"],
    )
    monkeypatch.setattr(
        guard,
        "NEGATIVE_CONTROLS",
        {"detector_cmd": {"reason": "test negative control", "expected_exit_code": 1}},
    )
    registry = _make_registry(tmp_path, [entry])
    report, exit_code, _ = _run(tmp_path, registry)
    assert exit_code == 1
    assert report["commands"][0]["error_reason"] == "nonzero_exit"


@requires_strace
def test_excluded_entries_reported_and_never_executed(tmp_path):
    script = _writer_script(tmp_path, "ok.py", _writer_body())
    eligible = _entry_in(tmp_path, "ok_cmd", _out_cmd(script, "out/a.json"), ["out/a.json"])
    excluded = _entry_in(
        tmp_path,
        "excluded_cmd",
        ["sh", "-c", "rm -rf / && true", "--out", "out/b.json"],
        ["out/b.json"],
        eligible=False,
        reason="unsafe_command",
    )
    registry = _make_registry(tmp_path, [eligible, excluded])
    report, exit_code, _ = _run(tmp_path, registry)
    assert exit_code == 0
    assert {c["id"] for c in report["commands"]} == {"ok_cmd"}
    assert report["exclusions"] == [{"id": "excluded_cmd", "exclusion_reason": "unsafe_command"}]
    assert report["executed_count"] == 1


@requires_strace
def test_stable_ordering_matches_registry_order(tmp_path):
    # Commands share command_index=0 (per-source); the report must still preserve registry order.
    entries = []
    for letter in "abc":
        script = _writer_script(tmp_path, f"{letter}.py", _writer_body())
        entries.append(
            _entry_in(
                tmp_path,
                f"{letter}_cmd",
                _out_cmd(script, f"out/{letter}.json"),
                [f"out/{letter}.json"],
            )
        )
    registry = _make_registry(tmp_path, entries)
    report, exit_code, _ = _run(tmp_path, registry)
    assert exit_code == 0
    assert [c["id"] for c in report["commands"]] == ["a_cmd", "b_cmd", "c_cmd"]


@requires_strace
def test_empty_eligible_fails_closed(tmp_path):
    excluded = _entry(
        "excluded_cmd", ["sh", "-c", "true"], ["o.json"], eligible=False, reason="unsafe_command"
    )
    registry = _make_registry(tmp_path, [excluded])
    report, exit_code, _ = _run(tmp_path, registry)
    assert exit_code == 2
    assert report["structural_error"] == "empty_eligible_set"
    assert report["executed_count"] == 0


@requires_strace
def test_empty_eligible_with_policy_exits_zero(tmp_path):
    excluded = _entry(
        "excluded_cmd", ["sh", "-c", "true"], ["o.json"], eligible=False, reason="unsafe_command"
    )
    registry = _make_registry(tmp_path, [excluded])
    report, exit_code, _ = _run(tmp_path, registry, zero_entry_policy="none-eligible-by-design")
    assert exit_code == 0
    assert report["zero_entry_policy"] == "none-eligible-by-design"


def test_structural_error_on_missing_registry(tmp_path):
    report, exit_code, _ = _run(tmp_path, tmp_path / "does_not_exist.yaml")
    assert exit_code == 2
    assert report["structural_error"].startswith("registry_load_error:")


def test_missing_strace_fails_closed_without_execution(tmp_path, monkeypatch):
    """Without strace the guard must exit 2 with strace_missing and execute nothing."""
    script = _writer_script(tmp_path, "ok.py", _writer_body())
    entry = _entry_in(tmp_path, "ok_cmd", _out_cmd(script, "out/report.json"), ["out/report.json"])
    registry = _make_registry(tmp_path, [entry])
    monkeypatch.setattr(guard.shutil, "which", lambda _name: None)
    report, exit_code, _ = _run(tmp_path, registry)
    assert exit_code == 2
    assert report["structural_error"] == "strace_missing"
    assert report["executed_count"] == 0
    assert report["commands"] == []


def test_pull_request_paths_cover_all_eligible_inputs():
    """Every committed input of an eligible entry must trigger the pull-request guard."""
    if not WORKFLOW.is_file() or not REGISTRY.is_file():
        pytest.skip("figure-render workflow or registry is absent")
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    workflow_events = workflow.get("on", workflow.get(True, {}))
    trigger_paths = workflow_events["pull_request"]["paths"]
    eligible_inputs = {
        item["path"]
        for entry in yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["entries"]
        if entry.get("recurrence_eligible") is True
        for item in entry.get("inputs", [])
    }
    missing = sorted(
        path
        for path in eligible_inputs
        if not any(fnmatch.fnmatchcase(path, pattern) for pattern in trigger_paths)
    )
    assert not missing, f"eligible inputs missing from pull_request.paths: {missing}"


def test_network_sandbox_contract_is_declared_and_fail_closed():
    """The CI workflow must own and use the fail-closed OS-level network sandbox."""
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    sandbox = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "run_without_network.sh"
    assert not sandbox.exists()
    assert "Verify the OS-level network-deny boundary" in workflow_text
    assert "run_without_network()" in workflow_text
    assert "unshare --user --map-root-user --net -- true" in workflow_text
    assert "sudo -n /usr/bin/unshare --net --" in workflow_text
    assert "/usr/bin/env -i" in workflow_text
    assert "scripts/dev/run_without_network.sh" not in workflow_text


def test_network_sandbox_denies_socket_access_when_available():
    """A usable Linux sandbox must deny outbound sockets; unavailable sandboxes fail closed in CI."""
    prefix = _network_namespace_prefix()
    if prefix is None:
        pytest.skip("fail-closed Linux network namespace is unavailable")
    proc = subprocess.run(
        [
            *prefix,
            sys.executable,
            "-c",
            (
                "import socket; "
                "s=socket.socket(); s.settimeout(1); "
                "\ntry: s.connect(('1.1.1.1', 443))\n"
                "except OSError: raise SystemExit(0)\n"
                "else: raise SystemExit(1)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_tracked_report_paths_are_repository_relative():
    """Tracked recurrence provenance must not expose an absolute worktree path."""
    report = guard._build_report(
        registry_path=guard.REGISTRY_DEFAULT,
        report_path=guard.REPORT_DEFAULT.resolve(),
        head="test-head",
        registry={"entries": []},
        command_outcomes=[],
        negative_control_warnings=[],
        zero_entry_policy=None,
        structural_error=None,
    )
    assert report["registry_path"] == "docs/context/figure_render_registry.v1.yaml"
    assert (
        report["report_path"]
        == "docs/context/evidence/issue_6770_figure_render_recurrence_report.json"
    )


# ---------------------------------------------------------------------------
# Real-registry regression smoke (mirrors the CI guard command)
# ---------------------------------------------------------------------------


@requires_strace
def test_real_registry_recurrence_passes(tmp_path):
    """The nine recurrence-eligible entries reproduce under isolation at the pinned commit."""
    if not REGISTRY.is_file():
        pytest.skip("committed figure-render registry is absent")
    report, exit_code, _ = _run(tmp_path, REGISTRY)
    assert exit_code == 0, report
    assert report["eligible_count"] == 9
    assert report["executed_count"] == 9
    assert report["passed_count"] == 9
    assert report["failed_count"] == 0
    statuses = {c["status"] for c in report["commands"]}
    assert statuses == {"passed"}
    # The violation-detector entry is the single declared negative control and must reproduce.
    negative = [c for c in report["commands"] if c["negative_control"]]
    assert len(negative) == 1
    assert negative[0]["exit_code"] == 1
    assert negative[0]["expected_exit_code"] == 1


@requires_strace
def test_cli_runs_against_committed_registry(tmp_path):
    """The CLI entry point exits 0 against the committed registry (smoke for the CI job)."""
    if not REGISTRY.is_file():
        pytest.skip("committed figure-render registry is absent")
    report_path = tmp_path / "cli_report.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--registry", str(REGISTRY), "--report", str(report_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema"] == guard.REPORT_SCHEMA
    assert payload["passed_count"] == 9
