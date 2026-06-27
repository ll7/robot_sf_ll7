"""Tests for the heavy forecast-model inventory / preflight (issue #2845).

These exercise the read-only inventory against synthetic checkouts (tmp_path) and the real
repository tree. They assert the probe semantics (present / missing-files / missing-symbols /
import-failure), the prerequisite present/absent/external classification, the fail-closed import
verdict, the minimum-offline-experiment ready/blocked roll-up, and the CLI exit codes. No model
is trained, no inference is run, and no simulation behavior is executed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.research import forecast_heavy_model_inventory as inv
from robot_sf.research.forecast_heavy_model_inventory import (
    ENTRY_POINT_SURFACES,
    EXPERIMENT_PREREQUISITES,
    MODEL_FAMILIES,
    CostTier,
    EntryPointSpec,
    MinimumExperimentStatus,
    PrerequisiteSpec,
    PrerequisiteStatus,
    SurfaceStatus,
    build_inventory_report,
    probe_entry_point,
    probe_prerequisite,
    render_markdown,
    repo_root,
)
from scripts.research.check_forecast_heavy_model_inventory import main as cli_main

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Static inventory shape.
# ---------------------------------------------------------------------------


def test_inventories_are_non_empty_and_uniquely_keyed():
    """Every inventory list is populated and free of duplicate keys."""
    assert MODEL_FAMILIES
    assert ENTRY_POINT_SURFACES
    assert EXPERIMENT_PREREQUISITES
    for items in (MODEL_FAMILIES, ENTRY_POINT_SURFACES, EXPERIMENT_PREREQUISITES):
        keys = [i.key for i in items]
        assert len(keys) == len(set(keys)), f"duplicate keys in {keys}"


def test_specs_are_json_serializable():
    """Every spec's ``to_dict`` round-trips through JSON without error."""
    for items in (MODEL_FAMILIES, ENTRY_POINT_SURFACES, EXPERIMENT_PREREQUISITES):
        for item in items:
            json.dumps(item.to_dict())


def test_model_families_cover_the_four_issue_archetypes():
    """The family inventory names the transformer, AgentFormer, CVAE, and diffusion archetypes."""
    keys = {m.key for m in MODEL_FAMILIES}
    assert {"transformer", "agentformer", "cvae", "diffusion"} <= keys


def test_model_family_tier_fields_use_cost_tier_enum():
    """Each family exposes the four required qualitative tradeoff axes as CostTier values."""
    for m in MODEL_FAMILIES:
        assert isinstance(m.compute_cost, CostTier)
        assert isinstance(m.inference_latency, CostTier)
        assert isinstance(m.uncertainty_quality, CostTier)
        assert isinstance(m.integration_burden, CostTier)
        # offline_use_cases is what the per-use-case recommendation keys off; never empty.
        assert m.offline_use_cases


def test_diffusion_has_worst_inference_latency():
    """Sanity check on the planning estimates: diffusion is the costliest for online latency."""
    by_key = {m.key: m for m in MODEL_FAMILIES}
    order = list(CostTier)
    diffusion_rank = order.index(by_key["diffusion"].inference_latency)
    for key in ("transformer", "cvae"):
        assert diffusion_rank >= order.index(by_key[key].inference_latency)


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
        module="robot_sf_no_such_module_2845",
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
        required_symbols=("join", "definitely_not_a_real_symbol_2845"),
    )
    result = probe_entry_point(spec, root=tmp_path)
    assert result.status is SurfaceStatus.MISSING_SYMBOLS
    assert result.missing_symbols == ("definitely_not_a_real_symbol_2845",)
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
    _write(tmp_path, "configs/forecast/heavy_model_cpu_budget.yaml")
    spec = PrerequisiteSpec(
        key="p",
        title="p",
        description="d",
        blocks=("b",),
        probe_paths=("configs/forecast/heavy_model_cpu_budget.yaml",),
    )
    result = probe_prerequisite(spec, root=tmp_path)
    assert result.status is PrerequisiteStatus.PRESENT
    assert result.satisfied
    assert result.matched_paths == ("configs/forecast/heavy_model_cpu_budget.yaml",)


def test_prerequisite_glob_path_matches(tmp_path):
    """Glob probe paths match created files."""
    _write(tmp_path, "configs/forecast/heavy_model_holdout_v1.yaml")
    spec = PrerequisiteSpec(
        key="p",
        title="p",
        description="d",
        blocks=("b",),
        probe_paths=("configs/forecast/heavy_model_holdout*.yaml",),
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
# Aggregate verdict + minimum-experiment roll-up.
# ---------------------------------------------------------------------------


def test_import_verdict_fails_closed_when_required_surfaces_missing(tmp_path):
    """Against an empty synthetic checkout every required surface is missing, so the verdict fails."""
    # On an empty tmp tree the declared entry-point files are absent (file probe runs before
    # import), so each required surface blocks and the import verdict fails closed.
    report = build_inventory_report(root=tmp_path)
    assert not report.ok
    assert report.surface_blockers
    # When surfaces are broken the minimum experiment is necessarily blocked too.
    assert report.minimum_experiment_status is MinimumExperimentStatus.BLOCKED
    assert report.exit_code() == 1


def test_real_checkout_imports_but_minimum_experiment_is_blocked():
    """On the live repo every required surface imports, yet the minimum experiment is blocked.

    This keeps the declared required surfaces honest against the current tree (if a named
    forecast surface is renamed, this fails) while documenting that the offline heavy-model
    experiment cannot run yet: its local prerequisites (dataset, adapter, runtime budget) and
    external prerequisites (dependency decision, trained checkpoint) are not satisfied.
    """
    report = build_inventory_report(root=repo_root())
    assert report.ok, [b.to_dict() for b in report.surface_blockers]
    assert report.exit_code() == 0
    assert report.minimum_experiment_status is MinimumExperimentStatus.BLOCKED
    pending_keys = {p.spec.key for p in report.pending_prerequisites}
    assert {"staged_holdout_dataset", "heavy_model_adapter", "cpu_runtime_budget"} <= pending_keys
    external = {p.spec.key for p in report.prerequisites if p.status is PrerequisiteStatus.EXTERNAL}
    assert {"dependency_decision", "trained_checkpoint"} <= external


def test_minimum_experiment_ready_only_when_local_prereqs_satisfied(tmp_path):
    """The roll-up reports READY when surfaces import and all non-external prereqs are present.

    Surfaces import from the live interpreter regardless of ``root`` (file-presence is the only
    root-relative surface check), so creating the local prerequisite probe files under
    ``tmp_path`` plus the declared surface files is enough to flip the roll-up to READY. External
    prerequisites stay unsatisfied but must not block local readiness.
    """
    # Create the declared entry-point files so the file-presence probe passes.
    for surface in ENTRY_POINT_SURFACES:
        _write(tmp_path, surface.file_path)
    # Satisfy every non-external prerequisite by creating a file for its first probe path,
    # expanding any glob wildcard into a concrete filename so the probe matches.
    for prereq in EXPERIMENT_PREREQUISITES:
        if prereq.external or not prereq.probe_paths:
            continue
        probe = prereq.probe_paths[0]
        concrete = probe.replace("*", "match").replace("?", "x").replace("[", "").replace("]", "")
        _write(tmp_path, concrete)
    report = build_inventory_report(root=tmp_path)
    assert report.ok
    assert not report.local_pending_prerequisites
    assert report.minimum_experiment_status is MinimumExperimentStatus.READY
    # External standing blockers are still reported as pending even when local readiness holds.
    assert report.pending_prerequisites


def test_report_to_dict_summary_counts_are_consistent():
    """The report summary counts agree with the probed lists."""
    report = build_inventory_report(root=repo_root())
    payload = report.to_dict()
    summary = payload["summary"]
    assert summary["model_families_total"] == len(MODEL_FAMILIES)
    assert summary["required_surfaces_total"] == len(report.required_surfaces)
    assert summary["prerequisites_total"] == len(EXPERIMENT_PREREQUISITES)
    assert summary["pending_prerequisites"] == len(report.pending_prerequisites)
    assert summary["local_pending_prerequisites"] == len(report.local_pending_prerequisites)
    assert payload["minimum_experiment_status"] == report.minimum_experiment_status.value


def test_render_markdown_mentions_verdict_and_sections():
    """The Markdown render includes the verdict banner and the three sections."""
    report = build_inventory_report(root=repo_root())
    text = render_markdown(report)
    assert f"#{inv.ISSUE}" in text
    assert "Candidate model families" in text
    assert "Offline-evaluation entry-point surfaces" in text
    assert "Minimum offline-experiment prerequisites" in text
    assert report.minimum_experiment_status.value.upper() in text


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def test_cli_default_exits_zero_on_real_checkout(capsys):
    """The default CLI run passes the import verdict (exit 0) and prints Markdown."""
    code = cli_main([])
    out = capsys.readouterr().out
    assert code == 0
    assert "Heavy forecast-model inventory" in out


def test_cli_json_mode_emits_report(capsys):
    """``--json`` emits a parseable report with the expected top-level keys."""
    code = cli_main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["issue"] == 2845
    assert set(payload) >= {
        "ok",
        "minimum_experiment_status",
        "model_families",
        "surfaces",
        "prerequisites",
        "summary",
    }


def test_cli_list_mode_emits_static_inventory(capsys):
    """``--list`` emits the static inventory without a verdict and exits 0."""
    code = cli_main(["--list"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert len(payload["model_families"]) == len(MODEL_FAMILIES)
    assert len(payload["entry_point_surfaces"]) == len(ENTRY_POINT_SURFACES)
    assert len(payload["experiment_prerequisites"]) == len(EXPERIMENT_PREREQUISITES)


@pytest.mark.parametrize("argv", [[], ["--json"], ["--list"]])
def test_cli_modes_return_int(argv, capsys):
    """All CLI modes return an int exit code (no exceptions)."""
    code = cli_main(argv)
    capsys.readouterr()
    assert isinstance(code, int)
