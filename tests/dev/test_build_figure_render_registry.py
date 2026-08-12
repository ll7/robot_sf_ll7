"""Tests for the canonical figure-render registry builder (issue #6769).

Covers each declared source class, deterministic classification rules, fail-closed duplicate
identity, stable ordering, and ``--check`` drift detection.
"""

# evidence-writer-exempt: these tests write to pytest tmp_path fixtures (local scratch), not to
# tracked evidence artifacts.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "build_figure_render_registry.py"
REGISTRY = (
    Path(__file__).resolve().parents[2] / "docs" / "context" / "figure_render_registry.v1.yaml"
)
AUDIT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "context"
    / "evidence"
    / "issue_6769_figure_render_registry_audit.json"
)

sys.path.insert(0, str(SCRIPT.parent))
import build_figure_render_registry as reg  # noqa: E402

# ---------------------------------------------------------------------------
# Pure-function unit tests (stable, no repo-state coupling).
# ---------------------------------------------------------------------------


def test_group_commands_splits_on_blank_lines_and_continuations():
    text = "uv run python a.py \\\n  --x 1\n\nuv run python b.py"
    cmds = reg._group_commands(text)
    assert cmds == ["uv run python a.py --x 1", "uv run python b.py"]


def test_group_commands_drops_shell_comments():
    text = "uv run python a.py\n# a comment\nuv run python b.py"
    assert reg._group_commands(text) == ["uv run python a.py", "uv run python b.py"]


def test_extract_named_fields_exact_key_only():
    # Exact field names are captured; near-misses are not.
    data = {
        "render_command_shape": "not-a-command",  # near miss
        "build_command": "uv run python build.py",  # exact
        "nested": {"render_command": "uv run python render.py"},  # exact, nested
        "rebuild_command": "uv run python reb.py",  # near miss
    }
    out = reg._extract_named_fields(data)
    assert "uv run python build.py" in out
    assert "uv run python render.py" in out
    assert "not-a-command" not in out
    assert "uv run python reb.py" not in out


def test_extract_labeled_fenced_requires_reproduc_render_label(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "## Other\n\n```\nuv run python ignored.py\n```\n\n"
        "## Reproduction\n\n```\nuv run python real.py --out docs/x.json\n```\n"
    )
    cmds = reg._extract_labeled_fenced(readme)
    assert cmds == ["uv run python real.py --out docs/x.json"]


# ---------------------------------------------------------------------------
# Classification rules via synthetic sources.
# ---------------------------------------------------------------------------


def _classify(raw: str) -> reg.Entry:
    source = reg.CommandSource(
        source_class=reg.SOURCE_CLASS_README,
        source_path="synthetic/README.md",
        raw_commands=[raw],
    )
    return reg.classify(source, 0, raw, head=None)


def test_classify_unsafe_shell_metachar():
    e = _classify("uv run python a.py | tee log.txt")
    assert not e.recurrence_eligible
    assert e.exclusion_reason == "unsafe_command"


def test_classify_env_assignment_prefix_is_unsafe():
    e = _classify("DISPLAY= MPLBACKEND=Agg uv run python render.py --out docs/o.json")
    assert not e.recurrence_eligible
    assert e.exclusion_reason == "unsafe_command"
    assert e.environment  # env prefix captured


def test_classify_slurm_submission():
    e = _classify("sbatch --job render scripts/run.sh --out docs/o.json")
    assert not e.recurrence_eligible
    assert e.exclusion_reason == "requires_slurm"


def test_classify_network_access():
    e = _classify("uv run python fetch.py https://example.com/x --out docs/o.json")
    assert not e.recurrence_eligible
    assert e.exclusion_reason == "requires_network"


def test_classify_placeholder_path():
    e = _classify("uv run python render.py --output <fresh-artifact-dir>")
    assert not e.recurrence_eligible
    assert e.exclusion_reason == "missing_committed_fixture"


def test_classify_absolute_output_path():
    e = _classify("uv run python render.py --out /tmp/robot_sf_x/o.json")
    assert not e.recurrence_eligible
    assert e.exclusion_reason == "unsafe_command"


def test_classify_missing_committed_input():
    e = _classify(
        "uv run python scripts/benchmark/x.py "
        "docs/context/evidence/does_not_exist_12345/input.json "
        "--out docs/context/evidence/issue_6769_test/o.json"
    )
    assert not e.recurrence_eligible
    assert e.exclusion_reason == "missing_committed_fixture"


def test_classify_no_explicit_output_is_non_deterministic():
    # A real committed driver invoked with no output flag has no verifiable recurrence contract.
    e = _classify("uv run python scripts/dev/build_figure_render_registry.py")
    assert not e.recurrence_eligible
    assert e.exclusion_reason == "non_deterministic_contract"


def test_classify_eligible_committed_input_and_output():
    # A real committed script writing to a committed evidence path is recurrence-eligible.
    e = _classify(
        "uv run python scripts/benchmark/build_horizon_timestep_ablation_report.py "
        "--issue 2837 --output-dir docs/context/evidence/issue_2837_horizon_timestep_ablation_2026-06-15"
    )
    assert e.recurrence_eligible
    assert e.exclusion_reason is None
    assert e.expected_outputs
    assert any(i["path"].startswith("scripts/") for i in e.inputs)


# ---------------------------------------------------------------------------
# Discovery against the real committed evidence tree.
# ---------------------------------------------------------------------------


def test_discover_build_command_txt_finds_known_file():
    sources = reg.discover_build_command_txt()
    paths = [s.source_path for s in sources]
    assert any("issue_5447_ch7_case_capsules/build_command.v1.txt" in p for p in paths)
    # The known build_command file yields exactly two commands (materialize + build_ch7).
    five447 = next(s for s in sources if "issue_5447" in s.source_path)
    assert len(five447.raw_commands) == 2


def test_discover_manifest_fields_respects_exact_field_boundary():
    # No committed manifest under docs/context/evidence uses an exact render_command/build_command
    # field (near-misses like render_command_shape / rebuild_command must NOT count).
    sources = reg.discover_manifest_fields()
    assert sources == []


def test_discover_readme_reproduction_only_catalog_linked():
    sources = reg.discover_readme_reproduction()
    assert sources, "expected at least one catalog-linked labeled README reproduction block"
    for s in sources:
        assert s.source_path.startswith("docs/context/evidence/")
        assert s.source_path.endswith("README.md")
        assert s.raw_commands


# ---------------------------------------------------------------------------
# Registry integrity, ordering, and duplicate-identity fail-closed.
# ---------------------------------------------------------------------------


def test_build_registry_entries_are_stably_sorted():
    registry, _ = reg.build_registry()
    entries = registry["entries"]
    keys = [(e["source_class"], e["source_path"], e["command_index"]) for e in entries]
    assert keys == sorted(keys)
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "entry ids must be unique"


def test_build_registry_schema_and_audit_consistency():
    registry, audit = reg.build_registry()
    assert registry["version"] == reg.REGISTRY_VERSION
    assert registry["issue"] == reg.ISSUE_NUMBER
    entries = registry["entries"]
    assert audit["total_entries"] == len(entries)
    assert audit["eligible_count"] == sum(1 for e in entries if e["recurrence_eligible"])
    assert audit["eligible_count"] + audit["ineligible_count"] == audit["total_entries"]
    allowed = {
        "external_input",
        "requires_slurm",
        "requires_network",
        "missing_committed_fixture",
        "non_deterministic_contract",
        "unsafe_command",
        "historical_only",
        "superseded",
    }
    for e in entries:
        if e["exclusion_reason"] is not None:
            assert e["exclusion_reason"] in allowed
            assert not e["recurrence_eligible"]
        else:
            assert e["recurrence_eligible"]
            assert e["expected_outputs"], "eligible entries must declare explicit expected outputs"


def test_duplicate_command_identity_fails_closed(monkeypatch):
    src = reg.CommandSource(
        source_class=reg.SOURCE_CLASS_README,
        source_path="docs/context/evidence/issue_dup/README.md",
        raw_commands=["uv run python scripts/benchmark/build_report.py --issue 1"],
    )
    monkeypatch.setattr(reg, "discover_all", lambda: [src, src])
    with pytest.raises(SystemExit):
        reg.build_registry()


def test_check_ignores_volatile_commit_fields(tmp_path, monkeypatch):
    """--check must not report drift when only provenance commit SHAs change between commits."""
    import importlib

    original_registry = reg.REGISTRY_PATH
    original_audit = reg.AUDIT_PATH
    tmp_registry = tmp_path / "figure_render_registry.v1.yaml"
    tmp_audit = tmp_path / "audit.json"
    reg.REGISTRY_PATH = tmp_registry
    reg.AUDIT_PATH = tmp_audit
    try:
        registry, audit = reg.build_registry()
        reg.write_outputs(registry, audit)
        assert reg.check_drift() == 0
        # Change only volatile provenance commit SHAs; substantive content is unchanged.
        data = yaml.safe_load(tmp_registry.read_text())
        data["provenance"]["source_commit"] = "deadbeef" * 5
        for entry in data["entries"]:
            entry["last_verified_commit"] = "cafebabe" * 5
        tmp_registry.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        assert reg.check_drift() == 0
    finally:
        reg.REGISTRY_PATH = original_registry
        reg.AUDIT_PATH = original_audit
        importlib.reload(reg)


def test_check_detects_drift(tmp_path, monkeypatch):
    # Point the module paths at temp copies and verify --check flags a mutated registry.
    import importlib

    original_registry = reg.REGISTRY_PATH
    original_audit = reg.AUDIT_PATH
    tmp_registry = tmp_path / "figure_render_registry.v1.yaml"
    tmp_audit = tmp_path / "audit.json"
    reg.REGISTRY_PATH = tmp_registry
    reg.AUDIT_PATH = tmp_audit
    try:
        registry, audit = reg.build_registry()
        reg.write_outputs(registry, audit)
        assert reg.check_drift() == 0
        # Mutate the committed registry: flip one eligibility flag.
        data = yaml.safe_load(tmp_registry.read_text())
        data["entries"][0]["recurrence_eligible"] = not data["entries"][0]["recurrence_eligible"]
        tmp_registry.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        assert reg.check_drift() == 1
    finally:
        reg.REGISTRY_PATH = original_registry
        reg.AUDIT_PATH = original_audit
        importlib.reload(reg)


def test_check_ignores_hash_drift_for_ineligible_entry(tmp_path):
    """Excluded commands do not block drift checks when an unused input file changes."""
    import importlib

    original_registry = reg.REGISTRY_PATH
    original_audit = reg.AUDIT_PATH
    reg.REGISTRY_PATH = tmp_path / "figure_render_registry.v1.yaml"
    reg.AUDIT_PATH = tmp_path / "audit.json"
    try:
        registry, audit = reg.build_registry()
        reg.write_outputs(registry, audit)
        target = next(
            entry
            for entry in registry["entries"]
            if not entry["recurrence_eligible"] and entry["inputs"]
        )
        data = yaml.safe_load(reg.REGISTRY_PATH.read_text())
        mutated = next(entry for entry in data["entries"] if entry["id"] == target["id"])
        mutated["inputs"][0]["sha256"] = "deadbeef" * 8
        reg.REGISTRY_PATH.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        assert reg.check_drift() == 0
    finally:
        reg.REGISTRY_PATH = original_registry
        reg.AUDIT_PATH = original_audit
        importlib.reload(reg)


def test_check_keeps_hash_drift_for_eligible_entry_fail_closed(tmp_path):
    """Executable recurrence inputs retain exact SHA drift enforcement."""
    import importlib

    original_registry = reg.REGISTRY_PATH
    original_audit = reg.AUDIT_PATH
    reg.REGISTRY_PATH = tmp_path / "figure_render_registry.v1.yaml"
    reg.AUDIT_PATH = tmp_path / "audit.json"
    try:
        registry, audit = reg.build_registry()
        reg.write_outputs(registry, audit)
        target = next(
            entry
            for entry in registry["entries"]
            if entry["recurrence_eligible"] and entry["inputs"]
        )
        data = yaml.safe_load(reg.REGISTRY_PATH.read_text())
        mutated = next(entry for entry in data["entries"] if entry["id"] == target["id"])
        mutated["inputs"][0]["sha256"] = "deadbeef" * 8
        reg.REGISTRY_PATH.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        assert reg.check_drift() == 1
    finally:
        reg.REGISTRY_PATH = original_registry
        reg.AUDIT_PATH = original_audit
        importlib.reload(reg)


# ---------------------------------------------------------------------------
# CLI surface: --write idempotency and --check against committed files.
# ---------------------------------------------------------------------------


def test_cli_check_matches_committed_files():
    # The committed registry/audit must be in sync with current discovery rules.
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_committed_outputs_exist():
    assert REGISTRY.is_file()
    assert AUDIT.is_file()
    data = yaml.safe_load(REGISTRY.read_text())
    audit = json.loads(AUDIT.read_text())
    assert data["version"] == 1
    assert audit["schema"].startswith("issue_6769_figure_render_registry_audit")
