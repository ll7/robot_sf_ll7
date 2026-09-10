"""Entry-point reference contract for installed CLI scripts (issue #8722)."""

from __future__ import annotations

import importlib.util
import re
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
