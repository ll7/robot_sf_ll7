"""Entry-point reference contract for installed CLI scripts (issue #8722)."""

from __future__ import annotations

import importlib.util
import os
import re
import socket
import textwrap
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
META_PATH = REPO_ROOT / "docs" / "cli_reference_meta.yaml"
OUTPUT_PATH = REPO_ROOT / "docs" / "cli_reference.md"
INDEX_RST = REPO_ROOT / "docs" / "index.rst"


def _load_generator():
    """Load the canonical CLI reference generator without package imports."""
    import sys

    module_path = REPO_ROOT / "scripts" / "dev" / "generate_cli_reference.py"
    spec = importlib.util.spec_from_file_location("generate_cli_reference", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _project_scripts() -> dict[str, str]:
    with PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    raw = data["project"]["scripts"]
    return {name: raw[name] for name in sorted(raw)}


def _overview_commands(text: str) -> list[str]:
    start = text.index("| Command |")
    section = text[start : text.index("## Commands", start)]
    return re.findall(r"\| `(.*?)` \|", section)


def _section_commands(text: str) -> list[str]:
    return re.findall(r"^### `(.*?)`$", text, flags=re.MULTILINE)


def test_entry_point_overview_lists_every_script_exactly_once() -> None:
    """Every declared script appears exactly once in the overview table."""
    scripts = _project_scripts()
    text = OUTPUT_PATH.read_text(encoding="utf-8")
    overview = _overview_commands(text)
    assert sorted(overview) == sorted(scripts)
    assert len(overview) == len(set(overview)) == len(scripts)


def test_entry_point_sections_cover_every_script_exactly_once() -> None:
    """Every declared script has exactly one per-command section."""
    scripts = _project_scripts()
    text = OUTPUT_PATH.read_text(encoding="utf-8")
    sections = _section_commands(text)
    assert sorted(sections) == sorted(scripts)
    assert len(sections) == len(set(sections))


def test_entry_point_documented_scripts_exist_in_pyproject() -> None:
    """No documented entry point may drift outside [project.scripts]."""
    scripts = _project_scripts()
    text = OUTPUT_PATH.read_text(encoding="utf-8")
    for name in _section_commands(text):
        assert name in scripts, f"documented nonexistent script: {name!r}"
    for name in _overview_commands(text):
        assert name in scripts, f"overview documents nonexistent script: {name!r}"


def test_entry_point_records_profile_availability_and_guide() -> None:
    """Each entry records a valid profile, boundary, and existing guide."""
    gen = _load_generator()
    scripts = gen.load_project_scripts(PYPROJECT)
    raw = gen.load_metadata(META_PATH)
    entries, meta_errors = gen.validate_metadata_entries(scripts, raw, REPO_ROOT)
    assert meta_errors == []
    text = OUTPUT_PATH.read_text(encoding="utf-8")
    for name in sorted(scripts):
        entry = entries[name]
        assert entry["profile"] in gen.ALLOWED_PROFILES
        assert entry["purpose"].strip()
        assert entry["availability"].strip()
        assert (REPO_ROOT / entry["guide"]).is_file()
        # Generated markdown carries the same contract fields.
        assert f"- Profile: `{entry['profile']}`" in text
        assert entry["availability"].split(";")[0][:40] in text


def test_entry_point_robot_sf_subcommands_link_to_task_guides() -> None:
    """Nested robot-sf subcommands are listed once with a task guide link."""
    text = OUTPUT_PATH.read_text(encoding="utf-8")
    assert "Nested `robot-sf` subcommands" in text
    # Full argparse flag dumps are out of scope; the table stays a summary.
    assert "--artifact-root" not in text
    assert "--skip-env-smoke" not in text
    meta = yaml.safe_load(META_PATH.read_text(encoding="utf-8"))
    expected_subs = sorted(meta["entries"]["robot-sf"]["subcommands"])
    assert len(expected_subs) >= 11
    for sub in expected_subs:
        guide = meta["entries"]["robot-sf"]["subcommands"][sub]
        target = guide[len("docs/") :] if guide.startswith("docs/") else guide
        assert f"| `{sub}` |" in text
        assert f"[{target}]({target})" in text
        assert (REPO_ROOT / guide).is_file()


def test_entry_point_reference_has_no_handwritten_counts() -> None:
    """The reference must not contain hand-written command-count claims."""
    text = OUTPUT_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "15 cli subcommands" not in lowered
    assert "30 subcommands" not in lowered
    assert "fully operational" not in lowered
    assert "implementation complete" not in lowered
    assert "generated file" in lowered
    assert "uv run python scripts/dev/generate_cli_reference.py" in text
    assert "Landlock" in text
    assert "seccomp" in text
    assert "unsupported hosts fail closed" in lowered


def test_entry_point_sphinx_index_links_cli_reference() -> None:
    """The Sphinx navigation layer exposes the generated reference."""
    text = INDEX_RST.read_text(encoding="utf-8")
    assert "CLI Reference <cli_reference>" in text


def test_entry_point_help_smoke_and_byte_stable_render() -> None:
    """Live --help succeeds for every entry and the render matches the file."""
    gen = _load_generator()
    rendered, errors, scripts, entries, probes = gen.generate(
        REPO_ROOT, PYPROJECT, META_PATH, gen.HELP_TIMEOUT_S
    )
    assert errors == []
    for name in sorted(scripts):
        probe = probes[name]
        assert probe.import_ok, f"{name}: {probe.import_error}"
        assert probe.help_ok, f"{name}: {probe.help_error}"
        assert probe.synopsis.strip(), f"{name}: empty synopsis"
    # Byte-stable regeneration: identical bytes on a pure re-render.
    rerendered = gen.render_markdown(scripts, entries, probes)
    assert rerendered == rendered
    committed = OUTPUT_PATH.read_text(encoding="utf-8")
    assert committed == rendered
    assert committed.endswith("\n") and not committed.endswith("\n\n")


def test_entry_point_help_ignores_ambient_terminal_and_carla_environment(
    monkeypatch,
) -> None:
    """Fixed probe settings keep generated help stable across ambient host variables."""
    gen = _load_generator()
    probe_results = []
    for columns, lines, carla_host in (
        ("40", "3", "ambient-first.invalid:2000"),
        ("120", "200", "ambient-second.invalid:3000"),
    ):
        monkeypatch.setenv("COLUMNS", columns)
        monkeypatch.setenv("LINES", lines)
        monkeypatch.setenv("CARLA_HOST", carla_host)
        probe_results.append(
            gen.probe_help(
                "robot-sf",
                "robot_sf.cli:main",
                REPO_ROOT,
                timeout_s=gen.HELP_TIMEOUT_S,
            )
        )
    first, second = probe_results
    assert first.help_ok, first.help_error
    assert second.help_ok, second.help_error
    assert first.help_text == second.help_text
    assert first.synopsis == second.synopsis
    assert first.subcommands == second.subcommands
    assert first.subcommand_help == second.subcommand_help


def test_entry_point_probe_denies_real_side_effect_escape_attempts(
    monkeypatch, tmp_path: Path
) -> None:
    """The OS policy blocks writes, network, processes, threads, and credentials."""
    gen = _load_generator()
    source_root = tmp_path / "source"
    source_root.mkdir()
    module_name = f"cli_probe_adversarial_{os.getpid()}"
    outside_marker = tmp_path / "outside-task-root.marker"
    credential_path = tmp_path / "credential.txt"
    credential_path.write_text("must-not-cross-boundary", encoding="utf-8")
    credential_fd = os.open(credential_path, os.O_RDONLY)
    inherited_fd = os.dup(credential_fd)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    module_path = source_root / f"{module_name}.py"
    module_path.write_text(
        textwrap.dedent(
            f"""
            import argparse
            import os
            import socket
            import subprocess
            import sys
            import threading
            from pathlib import Path

            OUTSIDE = Path({str(outside_marker)!r})
            PORT = {port}
            INHERITED_FD = {inherited_fd}

            def attempt(action):
                try:
                    action()
                except BaseException:
                    return "blocked"
                return "ran"

            def write_outside():
                OUTSIDE.write_text("escape", encoding="utf-8")

            IMPORT_STATUS = attempt(write_outside)
            PARSER_STATUS = "not-run"

            def _build_parser():
                global PARSER_STATUS
                PARSER_STATUS = attempt(write_outside)
                parser = argparse.ArgumentParser(description="adversarial probe")
                parser.add_subparsers().add_parser("parser-" + PARSER_STATUS)
                return parser

            def run_network():
                connection = socket.create_connection(("127.0.0.1", PORT), timeout=1)
                connection.close()

            def run_process():
                completed = subprocess.run(
                    [sys.executable, "-c", "print('process-ran')"],
                    capture_output=True,
                    check=False,
                    text=True,
                )
                if "process-ran" not in completed.stdout:
                    raise RuntimeError("child process produced no proof of execution")

            def run_thread():
                thread_state = []

                def worker():
                    thread_state.append("ran")
                    try:
                        write_outside()
                    except BaseException:
                        pass

                thread = threading.Thread(target=worker)
                thread.start()
                thread.join()
                if thread_state != ["ran"]:
                    raise RuntimeError("thread did not run")

            def inspect_inherited_fd():
                os.fstat(INHERITED_FD)

            def main(argv):
                if argv != ["--help"]:
                    return 1
                task_marker = Path.cwd() / "task-root.marker"
                try:
                    task_marker.write_text("allowed", encoding="utf-8")
                    task_status = "allowed"
                except BaseException:
                    task_status = "blocked"
                print("usage: adversarial-probe [--help]")
                print("import-write:" + IMPORT_STATUS)
                print("main-write:" + attempt(write_outside))
                print("task-write:" + task_status)
                print("network:" + attempt(run_network))
                print("process:" + attempt(run_process))
                print("thread:" + attempt(run_thread))
                print("fd-credential:" + ("leaked" if attempt(inspect_inherited_fd) == "ran" else "hidden"))
                print("credential:" + ("leaked" if os.environ.get("CLI_PROBE_SECRET") else "hidden"))
                return 0
            """
        ).lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLI_PROBE_SECRET", "must-not-cross-boundary")

    try:
        result = gen.probe_help(
            "adversarial-probe",
            f"{module_name}:main",
            source_root,
            timeout_s=2,
        )
        assert result.help_ok, result.help_error
        assert "import-write:blocked" in result.help_text
        assert "main-write:blocked" in result.help_text
        assert "task-write:allowed" in result.help_text
        assert "network:blocked" in result.help_text
        assert "process:blocked" in result.help_text
        assert "thread:blocked" in result.help_text
        assert "fd-credential:hidden" in result.help_text
        assert "credential:hidden" in result.help_text
        assert result.subcommands == ["parser-blocked"]
        assert not outside_marker.exists()
        listener.settimeout(0.2)
        try:
            connection, _address = listener.accept()
        except TimeoutError:
            pass
        else:
            connection.close()
            raise AssertionError("probe reached the real loopback listener")
    finally:
        listener.close()
        os.close(inherited_fd)
        os.close(credential_fd)


def test_entry_point_negative_unimportable_callable() -> None:
    """Fail-closed when a declared callable cannot import (core profile)."""
    gen = _load_generator()
    scripts = {"bad-cmd": "no_such_module_xyz:main"}
    entries = {
        "bad-cmd": {
            "purpose": "x",
            "profile": "core",
            "availability": "y",
            "guide": "docs/benchmark.md",
            "subcommands": {},
        }
    }
    probes = {
        "bad-cmd": gen.ProbeResult(
            script="bad-cmd",
            spec="no_such_module_xyz:main",
            import_ok=False,
            import_error="ModuleNotFoundError: x",
        )
    }
    errors = gen.check_entry_points(scripts, entries, probes)
    assert any("cannot import" in e for e in errors)


def test_entry_point_negative_failing_help_is_fail_closed() -> None:
    """Fail-closed when --help fails for a required (non-carla) command."""
    gen = _load_generator()
    scripts = {"req-cmd": "some.mod:main"}
    entries = {
        "req-cmd": {
            "purpose": "x",
            "profile": "core",
            "availability": "y",
            "guide": "docs/benchmark.md",
            "subcommands": {},
        }
    }
    probes = {
        "req-cmd": gen.ProbeResult(
            script="req-cmd",
            spec="some.mod:main",
            import_ok=True,
            help_ok=False,
            help_error="--help exit 1: boom",
        )
    }
    errors = gen.check_entry_points(scripts, entries, probes)
    assert any("--help failed" in e for e in errors)


def test_entry_point_negative_optional_help_recorded_unavailable() -> None:
    """Optional carla help failures are recorded as unavailable, not errors."""
    gen = _load_generator()
    scripts = {"opt-cmd": "some.mod:main"}
    entries = {
        "opt-cmd": {
            "purpose": "x",
            "profile": "carla",
            "availability": "y",
            "guide": "docs/benchmark.md",
            "subcommands": {},
        }
    }
    probes = {
        "opt-cmd": gen.ProbeResult(
            script="opt-cmd",
            spec="some.mod:main",
            import_ok=True,
            help_ok=False,
            help_error="--help exit 1: missing optional dep",
        )
    }
    assert gen.check_entry_points(scripts, entries, probes) == []
    assert probes["opt-cmd"].status.startswith("unavailable:")


def test_entry_point_negative_missing_doc_owner() -> None:
    """Fail-closed when a declared script lacks a metadata owner."""
    gen = _load_generator()
    scripts = {"a-cmd": "m:a", "b-cmd": "m:b"}
    entries = {
        "a-cmd": {
            "purpose": "x",
            "profile": "core",
            "availability": "y",
            "guide": "docs/benchmark.md",
            "subcommands": {},
        }
    }
    probes = {
        "a-cmd": gen.ProbeResult(
            script="a-cmd", spec="m:a", import_ok=True, help_ok=True, synopsis="s"
        ),
        "b-cmd": gen.ProbeResult(
            script="b-cmd", spec="m:b", import_ok=True, help_ok=True, synopsis="s"
        ),
    }
    errors = gen.check_entry_points(scripts, entries, probes)
    assert any("missing documentation owner" in e and "b-cmd" in e for e in errors)


def test_entry_point_negative_stale_documented_entry() -> None:
    """Fail-closed when metadata documents a script absent from pyproject."""
    gen = _load_generator()
    scripts = {"real-cmd": "m:a"}
    entries = {
        "real-cmd": {
            "purpose": "x",
            "profile": "core",
            "availability": "y",
            "guide": "docs/benchmark.md",
            "subcommands": {},
        },
        "ghost-cmd": {
            "purpose": "x",
            "profile": "core",
            "availability": "y",
            "guide": "docs/benchmark.md",
            "subcommands": {},
        },
    }
    probes = {
        "real-cmd": gen.ProbeResult(
            script="real-cmd", spec="m:a", import_ok=True, help_ok=True, synopsis="s"
        ),
        "ghost-cmd": gen.ProbeResult(
            script="ghost-cmd", spec="m:b", import_ok=True, help_ok=True, synopsis="s"
        ),
    }
    errors = gen.check_entry_points(scripts, entries, probes)
    assert any("stale documented entry" in e and "ghost-cmd" in e for e in errors)


def test_entry_point_negative_missing_subcommand_owner() -> None:
    """Fail-closed when a live robot-sf subcommand lacks a guide owner."""
    gen = _load_generator()
    scripts = {"robot-sf": "robot_sf.cli:main"}
    entries = {
        "robot-sf": {
            "purpose": "x",
            "profile": "core",
            "availability": "y",
            "guide": "docs/adoption_path.md",
            "subcommands": {"doctor": "docs/adoption_path.md"},
        }
    }
    probes = {
        "robot-sf": gen.ProbeResult(
            script="robot-sf",
            spec="robot_sf.cli:main",
            import_ok=True,
            help_ok=True,
            synopsis="s",
            subcommands=["doctor", "ghost-sub"],
        )
    }
    errors = gen.check_entry_points(scripts, entries, probes)
    assert any("ghost-sub" in e for e in errors)


def test_entry_point_metadata_guides_exist() -> None:
    """Every metadata guide path resolves to a committed file."""
    meta = yaml.safe_load(META_PATH.read_text(encoding="utf-8"))
    assert meta["version"] == 1
    for name, entry in meta["entries"].items():
        assert (REPO_ROOT / entry["guide"]).is_file(), name
        for sub, guide in (entry.get("subcommands") or {}).items():
            path = guide["guide"] if isinstance(guide, dict) else guide
            assert (REPO_ROOT / path).is_file(), f"{name} {sub}"
