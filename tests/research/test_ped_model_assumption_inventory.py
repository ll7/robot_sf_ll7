"""Tests for the pedestrian-model assumption inventory / preflight (issue #3481).

These exercise the read-only inventory against synthetic checkouts (tmp_path) and the real
repository tree. They assert the probe semantics (present / missing-files / missing-symbols /
import-failure), the prerequisite present/absent/external classification, the fail-closed
verdict logic, and the CLI exit codes. No simulation behavior is executed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.research import ped_model_assumption_inventory as inv
from robot_sf.research.ped_model_assumption_inventory import (
    CURRENT_ASSUMPTIONS,
    ENTRY_POINT_SURFACES,
    EXPERIMENT_PREREQUISITES,
    EntryPointSpec,
    PrerequisiteSpec,
    PrerequisiteStatus,
    SurfaceStatus,
    build_inventory_report,
    probe_entry_point,
    probe_prerequisite,
    render_markdown,
    repo_root,
)
from scripts.research.check_ped_model_assumption_inventory import main as cli_main

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Static inventory shape.
# ---------------------------------------------------------------------------


def test_inventories_are_non_empty_and_uniquely_keyed():
    """Every inventory list is populated and free of duplicate keys."""
    assert CURRENT_ASSUMPTIONS
    assert ENTRY_POINT_SURFACES
    assert EXPERIMENT_PREREQUISITES
    for items in (CURRENT_ASSUMPTIONS, ENTRY_POINT_SURFACES, EXPERIMENT_PREREQUISITES):
        keys = [i.key for i in items]
        assert len(keys) == len(set(keys)), f"duplicate keys in {keys}"


def test_specs_are_json_serializable():
    """Every spec's ``to_dict`` round-trips through JSON without error."""
    for items in (CURRENT_ASSUMPTIONS, ENTRY_POINT_SURFACES, EXPERIMENT_PREREQUISITES):
        for item in items:
            json.dumps(item.to_dict())


def test_assumptions_target_the_three_issue_axes():
    """The assumption inventory names the FoV, HSFM-heading, and TTC gaps the issue targets."""
    keys = {a.key for a in CURRENT_ASSUMPTIONS}
    assert {
        "no_fov_attenuation",
        "heading_coupled_to_velocity",
        "euclidean_distance_repulsion",
    } <= keys


# ---------------------------------------------------------------------------
# Entry-point surface probing.
# ---------------------------------------------------------------------------


def _write(root: Path, rel: str, body: str = "") -> Path:
    """Create ``root/rel`` (with parents) and return the path."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_probe_missing_files_when_path_absent(tmp_path):
    """A declared file that does not exist yields MISSING_FILES and blocks (required)."""
    spec = EntryPointSpec(
        key="x",
        title="x",
        module="os.path",  # importable, but file probe runs first
        file_path="does/not/exist.py",
        required_symbols=("join",),
    )
    result = probe_entry_point(spec, root=tmp_path)
    assert result.status is SurfaceStatus.MISSING_FILES
    assert not result.present
    assert result.is_blocker


def test_probe_missing_module_when_import_fails(tmp_path):
    """A present file backing an unimportable module yields MISSING_MODULE."""
    _write(tmp_path, "pkg/mod.py")
    spec = EntryPointSpec(
        key="x",
        title="x",
        module="robot_sf_no_such_module_3481",
        file_path="pkg/mod.py",
    )
    result = probe_entry_point(spec, root=tmp_path)
    assert result.status is SurfaceStatus.MISSING_MODULE
    assert result.is_blocker


def test_probe_missing_symbols_when_symbol_absent(tmp_path):
    """An importable module missing a required symbol yields MISSING_SYMBOLS."""
    _write(tmp_path, "os_path_stub.py")  # back a real importable module's file probe
    spec = EntryPointSpec(
        key="x",
        title="x",
        module="os.path",
        file_path="os_path_stub.py",
        required_symbols=("join", "definitely_not_a_real_symbol_3481"),
    )
    result = probe_entry_point(spec, root=tmp_path)
    assert result.status is SurfaceStatus.MISSING_SYMBOLS
    assert result.missing_symbols == ("definitely_not_a_real_symbol_3481",)
    assert result.is_blocker


def test_probe_present_when_file_and_symbols_exist(tmp_path):
    """A present file plus all required symbols yields PRESENT and does not block."""
    _write(tmp_path, "os_path_stub.py")
    spec = EntryPointSpec(
        key="x",
        title="x",
        module="os.path",
        file_path="os_path_stub.py",
        required_symbols=("join",),
    )
    result = probe_entry_point(spec, root=tmp_path)
    assert result.status is SurfaceStatus.PRESENT
    assert result.present
    assert not result.is_blocker


def test_optional_surface_never_blocks(tmp_path):
    """An optional surface that is missing is not a verdict blocker."""
    spec = EntryPointSpec(
        key="x",
        title="x",
        module="os.path",
        file_path="absent.py",
        required=False,
    )
    result = probe_entry_point(spec, root=tmp_path)
    assert not result.present
    assert not result.is_blocker


# ---------------------------------------------------------------------------
# Prerequisite probing.
# ---------------------------------------------------------------------------


def test_prerequisite_absent_when_no_path_matches(tmp_path):
    """A local prerequisite with no matching path is ABSENT (a planned blocker)."""
    spec = PrerequisiteSpec(
        key="p",
        title="p",
        description="d",
        blocks=("b",),
        probe_paths=("nope/here.py",),
    )
    result = probe_prerequisite(spec, root=tmp_path)
    assert result.status is PrerequisiteStatus.ABSENT
    assert not result.satisfied


def test_prerequisite_present_when_path_exists(tmp_path):
    """A prerequisite flips to PRESENT once any probe path lands."""
    _write(tmp_path, "configs/pedestrian/hsfm_ttc.yaml")
    spec = PrerequisiteSpec(
        key="p",
        title="p",
        description="d",
        blocks=("b",),
        probe_paths=("configs/pedestrian/hsfm_ttc.yaml",),
    )
    result = probe_prerequisite(spec, root=tmp_path)
    assert result.status is PrerequisiteStatus.PRESENT
    assert result.satisfied
    assert result.matched_paths == ("configs/pedestrian/hsfm_ttc.yaml",)


def test_prerequisite_glob_path_matches(tmp_path):
    """Glob probe paths match created files."""
    _write(tmp_path, "tests/fixtures/ped_npc/narrow_passage_sliding.yaml")
    spec = PrerequisiteSpec(
        key="p",
        title="p",
        description="d",
        blocks=("b",),
        probe_paths=("tests/fixtures/ped_npc/*.yaml",),
    )
    result = probe_prerequisite(spec, root=tmp_path)
    assert result.satisfied


def test_external_prerequisite_is_external_without_paths(tmp_path):
    """An external prerequisite with no probe path is a standing EXTERNAL blocker."""
    spec = PrerequisiteSpec(
        key="p",
        title="p",
        description="d",
        blocks=("b",),
        external=True,
    )
    result = probe_prerequisite(spec, root=tmp_path)
    assert result.status is PrerequisiteStatus.EXTERNAL
    assert not result.satisfied


# ---------------------------------------------------------------------------
# Aggregate verdict.
# ---------------------------------------------------------------------------


def test_verdict_fails_closed_when_required_surfaces_missing(tmp_path):
    """Against an empty synthetic checkout every required surface is missing, so the verdict fails."""
    # On an empty tmp tree the declared entry-point files are absent (file probe runs before
    # import), so each required surface blocks and the verdict fails closed.
    report = build_inventory_report(root=tmp_path)
    assert not report.ok
    assert report.surface_blockers


def test_real_checkout_inventory_passes_and_lists_pending_blockers():
    """On the live repository every required entry-point imports; HSFM/TTC work is still pending.

    This keeps the declared required surfaces honest against the current tree (if a named
    force-core or ped-NPC surface is renamed, this fails) while documenting that the force-law
    family still lacks stronger proof and external calibration.
    """
    report = build_inventory_report(root=repo_root())
    assert report.ok, [b.to_dict() for b in report.surface_blockers]
    pending_keys = {p.spec.key for p in report.pending_prerequisites}
    assert "hsfm_heading_state" in pending_keys
    assert "fov_attenuation" not in pending_keys
    assert "ttc_predictive_term" not in pending_keys
    assert "narrow_passage_fixture" in pending_keys
    assert "bottleneck_fixture" in pending_keys
    assert "versioned_parameters" not in pending_keys
    assert "design_note" not in pending_keys
    # The calibration-data prerequisite is an external standing blocker.
    external = {p.spec.key for p in report.prerequisites if p.status is PrerequisiteStatus.EXTERNAL}
    assert "calibration_data" in external


def test_report_to_dict_summary_counts_are_consistent():
    """The report summary counts agree with the probed lists."""
    report = build_inventory_report(root=repo_root())
    summary = report.to_dict()["summary"]
    assert summary["required_surfaces_total"] == len(report.required_surfaces)
    assert summary["prerequisites_total"] == len(EXPERIMENT_PREREQUISITES)
    assert summary["pending_prerequisites"] == len(report.pending_prerequisites)


def test_render_markdown_mentions_verdict_and_sections():
    """The Markdown render includes the verdict banner and the three sections."""
    report = build_inventory_report(root=repo_root())
    text = render_markdown(report)
    assert f"#{inv.ISSUE}" in text
    assert "Current force-model assumptions" in text
    assert "Entry-point surfaces" in text
    assert "Experiment prerequisites" in text


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def test_cli_default_exits_zero_on_real_checkout(capsys):
    """The default CLI run passes (exit 0) on the live checkout and prints Markdown."""
    code = cli_main([])
    out = capsys.readouterr().out
    assert code == 0
    assert "Pedestrian-model assumption inventory" in out


def test_cli_json_mode_emits_report(capsys):
    """``--json`` emits a parseable report with the expected top-level keys."""
    code = cli_main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["issue"] == 3481
    assert set(payload) >= {"ok", "assumptions", "surfaces", "prerequisites", "summary"}


def test_cli_list_mode_emits_static_inventory(capsys):
    """``--list`` emits the static inventory without a verdict and exits 0."""
    code = cli_main(["--list"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert len(payload["assumptions"]) == len(CURRENT_ASSUMPTIONS)
    assert len(payload["entry_point_surfaces"]) == len(ENTRY_POINT_SURFACES)
    assert len(payload["experiment_prerequisites"]) == len(EXPERIMENT_PREREQUISITES)


@pytest.mark.parametrize("argv", [[], ["--json"], ["--list"]])
def test_cli_modes_return_int(argv, capsys):
    """All CLI modes return an int exit code (no exceptions)."""
    code = cli_main(argv)
    capsys.readouterr()
    assert isinstance(code, int)
