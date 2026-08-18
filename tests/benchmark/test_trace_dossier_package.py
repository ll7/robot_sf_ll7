"""Tests for diagnostic trace dossier package composition."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from robot_sf.benchmark import trace_dossier_package as package_module
from robot_sf.benchmark.trace_dossier_package import (
    TraceDossierPackageError,
    build_trace_dossier_package,
    validate_trace_dossier_package_manifest,
)
from scripts.tools.campaign_result_store import write_result_store
from scripts.tools.export_trace_dossier import TraceDossierExportError

_ROOT = Path(__file__).resolve().parents[2]
_RELEASE = _ROOT / "configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml"
_REAL_RELEASE = _ROOT / "configs/benchmarks/releases/issue_7086_trace_dossier_diagnostic_v0_1.yaml"
_TRACE_FIXTURE = (
    _ROOT / "tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json"
)
_REAL_TRACE_SERIES = (
    _ROOT
    / "docs/context/evidence/issue_4848_group_crossing_exemplars_2026-07/goal/"
    / "classic_group_crossing_medium_seed22_median/trace_series.json"
)
_CLI = _ROOT / "scripts/tools/build_trace_dossier_package.py"


def _write_source(tmp_path: Path, *, name: str = "fixture_episode_001.json") -> Path:
    payload = json.loads(_TRACE_FIXTURE.read_text(encoding="utf-8"))
    payload["source"].update(
        {
            "scenario_id": "francis2023_blind_corner",
            "seed": 111,
            "planner_id": "goal",
            "episode_id": "fixture_episode_001",
        }
    )
    source = tmp_path / name
    source.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return source


def _write_store(tmp_path: Path, source: Path) -> Path:
    store = tmp_path / "campaign-store"
    write_result_store(
        store,
        [
            {
                "run_id": "run-fixture",
                "episode_id": "fixture_episode_001",
                "planner": "goal",
                "scenario_id": "francis2023_blind_corner",
                "scenario_family": "classic",
                "seed": 111,
                "row_status": "native",
                "artifact_uri": str(source),
                "artifact_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
        study_id="trace-dossier-fixture",
        command="fixture",
        source_commit="fixture-commit",
    )
    return store


def _write_real_store(tmp_path: Path) -> Path:
    """Bind the retained #4848 trace-series exemplar to its campaign identity."""
    store = tmp_path / "real-campaign-store"
    write_result_store(
        store,
        [
            {
                "run_id": "run-13334",
                "episode_id": "classic_group_crossing_medium--22--605d6793ad25c1f5",
                "planner": "goal",
                "scenario_id": "classic_group_crossing_medium",
                "scenario_family": "group_crossing",
                "seed": 22,
                "row_status": "native",
                "artifact_uri": str(_REAL_TRACE_SERIES),
                "artifact_sha256": hashlib.sha256(_REAL_TRACE_SERIES.read_bytes()).hexdigest(),
            }
        ],
        study_id="issue4206_trace_capable_h600_rerun_20260704",
        command="retained #4848 exemplar fixture",
        source_commit="0b0214ced856eac77fa9a4c15b02921eabab1661",
    )
    return store


def _candidate(source: Path, **overrides: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "campaign_id": "trace-dossier-fixture",
        "cell_id": "cell-fixture",
        "scenario_id": "francis2023_blind_corner",
        "scenario_family": "classic",
        "planner_id": "goal",
        "release_arm_id": "smoke",
        "seed": 111,
        "seed_id": "111",
        "episode_id": "fixture_episode_001",
        "verdict": "nominal",
        "label_strength": 1.0,
        "primary_order": 0.5,
        "trace_artifact_uri": str(source),
        "trace_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    candidate.update(overrides)
    return candidate


def test_package_composes_cell_binding_export_render_and_checksums(tmp_path: Path) -> None:
    """A selected existing trace becomes one mechanically bound diagnostic package."""
    source = _write_source(tmp_path)
    store = _write_store(tmp_path, source)

    result = build_trace_dossier_package(
        candidates=[_candidate(source)],
        release_manifest_path=_RELEASE,
        campaign_store_dir=store,
        output_dir=tmp_path / "package",
    )

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "trace_dossier_package.v1"
    assert payload["evidence_boundary"] == "diagnostic_only"
    assert payload["campaign_cell"]["cell_id"] == "cell-fixture"
    assert payload["selection"]["selected_seed_id"] == "111"
    assert payload["cell_binding"]["selected_trace"]["episode_id"] == "fixture_episode_001"
    assert payload["cell_binding"]["selected_verdict_count"] == 1
    assert payload["render"]["evidence_boundary"] == "diagnostic_only"
    assert (tmp_path / "package" / "SHA256SUMS").is_file()
    renderer_manifest = json.loads(
        (tmp_path / "package/render/renderer_manifest.json").read_text(encoding="utf-8")
    )
    assert renderer_manifest["source_trace"]["path"] == "export/trace.json"
    assert renderer_manifest["outputs"]["png"]["path"] == "render/dossier.png"
    assert renderer_manifest["outputs"]["svg"]["path"] == "render/dossier.svg"
    assert renderer_manifest["outputs"]["pdf"]["path"] == "render/dossier.pdf"
    assert renderer_manifest["outputs"]["caption"]["path"] == "render/caption.md"


def test_package_is_byte_deterministic_for_same_inputs(tmp_path: Path) -> None:
    """Repeated composition keeps all package-relative metadata and bytes stable."""
    source = _write_source(tmp_path)
    store = _write_store(tmp_path, source)
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_trace_dossier_package(
        candidates=[_candidate(source)],
        release_manifest_path=_RELEASE,
        campaign_store_dir=store,
        output_dir=first,
    )
    build_trace_dossier_package(
        candidates=[_candidate(source)],
        release_manifest_path=_RELEASE,
        campaign_store_dir=store,
        output_dir=second,
    )

    first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    assert first_files == second_files
    for relative in first_files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes(), relative


def test_package_composes_retained_real_trace_to_publication_candidate(tmp_path: Path) -> None:
    """One retained #4848 trace reaches the complete diagnostic package path."""
    store = _write_real_store(tmp_path)
    source_sha256 = hashlib.sha256(_REAL_TRACE_SERIES.read_bytes()).hexdigest()
    candidate = _candidate(
        _REAL_TRACE_SERIES,
        campaign_id="issue4206_trace_capable_h600_rerun_20260704",
        cell_id="classic_group_crossing_medium__goal",
        scenario_id="classic_group_crossing_medium",
        scenario_family="group_crossing",
        release_arm_id="goal",
        seed=22,
        seed_id="22",
        episode_id="classic_group_crossing_medium--22--605d6793ad25c1f5",
        verdict="success",
        primary_order=0.5,
        trace_artifact_uri=str(_REAL_TRACE_SERIES),
        trace_sha256=source_sha256,
    )

    result = build_trace_dossier_package(
        candidates=[candidate],
        release_manifest_path=_REAL_RELEASE,
        campaign_store_dir=store,
        output_dir=tmp_path / "real-package",
    )

    output = tmp_path / "real-package"
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["evidence_boundary"] == "diagnostic_only"
    assert payload["campaign_cell"]["scenario_id"] == "classic_group_crossing_medium"
    assert payload["selection"]["selected_seed_id"] == "22"
    assert payload["render"]["svg"] == "render/dossier.svg"
    assert payload["render"]["pdf"] == "render/dossier.pdf"
    assert payload["render"]["caption"] == "render/caption.md"
    assert (output / "render/dossier.svg").stat().st_size > 0
    assert (output / "render/dossier.pdf").stat().st_size > 0
    caption = (output / "render/caption.md").read_text(encoding="utf-8")
    assert "classic_group_crossing_medium" in caption
    assert "not benchmark" in caption
    renderer_manifest = json.loads(
        (output / "render/renderer_manifest.json").read_text(encoding="utf-8")
    )
    assert set(renderer_manifest["outputs"]) == {"png", "svg", "pdf", "caption"}


def test_package_rejects_export_identity_mismatch(tmp_path: Path) -> None:
    """A cell row cannot relabel the episode selected by the campaign result store."""
    source = _write_source(tmp_path)
    store = _write_store(tmp_path, source)
    with pytest.raises(TraceDossierPackageError, match="authoritative campaign row"):
        build_trace_dossier_package(
            candidates=[_candidate(source, episode_id="wrong-episode")],
            release_manifest_path=_RELEASE,
            campaign_store_dir=store,
            output_dir=tmp_path / "package",
        )


def test_package_rejects_selected_source_checksum_mismatch(tmp_path: Path) -> None:
    """A candidate checksum cannot overwrite source provenance from the result store."""
    source = _write_source(tmp_path)
    store = _write_store(tmp_path, source)
    with pytest.raises(TraceDossierPackageError, match="source artifact SHA-256"):
        build_trace_dossier_package(
            candidates=[_candidate(source, trace_sha256="0" * 64)],
            release_manifest_path=_RELEASE,
            campaign_store_dir=store,
            output_dir=tmp_path / "package",
        )


def test_package_rejects_candidate_campaign_id_not_in_store(tmp_path: Path) -> None:
    """A candidate cannot relabel the authoritative campaign result store."""
    source = _write_source(tmp_path)
    store = _write_store(tmp_path, source)
    with pytest.raises(TraceDossierPackageError, match="campaign result store study_id"):
        build_trace_dossier_package(
            candidates=[_candidate(source, campaign_id="campaign-fixture")],
            release_manifest_path=_RELEASE,
            campaign_store_dir=store,
            output_dir=tmp_path / "package",
        )


def test_package_rejects_source_inside_output_before_writing(tmp_path: Path) -> None:
    """A source named like a package file cannot be overwritten during composition."""
    output = tmp_path / "package"
    output.mkdir()
    source = _write_source(output, name="package_manifest.json")
    store = _write_store(tmp_path, source)
    original_source = source.read_bytes()

    with pytest.raises(TraceDossierPackageError, match="source artifact.*inside package output"):
        build_trace_dossier_package(
            candidates=[_candidate(source)],
            release_manifest_path=_RELEASE,
            campaign_store_dir=store,
            output_dir=output,
        )

    assert source.read_bytes() == original_source
    assert sorted(path.name for path in output.iterdir()) == ["package_manifest.json"]


def test_package_schema_validates_nested_selection_and_binding(tmp_path: Path) -> None:
    """The package schema must validate the nested versioned metadata blocks."""
    source = _write_source(tmp_path)
    store = _write_store(tmp_path, source)
    result = build_trace_dossier_package(
        candidates=[_candidate(source)],
        release_manifest_path=_RELEASE,
        campaign_store_dir=store,
        output_dir=tmp_path / "package",
    )

    invalid_selection = deepcopy(result.manifest)
    invalid_selection["selection"].pop("selected_seed_id")
    with pytest.raises(TraceDossierPackageError, match="selection"):
        validate_trace_dossier_package_manifest(invalid_selection)

    invalid_binding = deepcopy(result.manifest)
    invalid_binding["cell_binding"]["selected_trace"].pop("trace_sha256")
    with pytest.raises(TraceDossierPackageError, match="cell_binding"):
        validate_trace_dossier_package_manifest(invalid_binding)


def test_package_rejects_malformed_renderer_manifest_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed renderer metadata becomes a package error instead of a raw lookup error."""
    source = _write_source(tmp_path)
    store = _write_store(tmp_path, source)

    def write_malformed_manifest(*args: object, **kwargs: object) -> None:
        output_png = kwargs["output_png"]
        manifest_path = kwargs["manifest_path"]
        assert isinstance(output_png, Path)
        assert isinstance(manifest_path, Path)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        output_png.write_bytes(b"not-a-real-png")
        manifest_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(package_module, "render_trace_dossier", write_malformed_manifest)
    with pytest.raises(TraceDossierPackageError, match="renderer manifest cannot be normalized"):
        build_trace_dossier_package(
            candidates=[_candidate(source)],
            release_manifest_path=_RELEASE,
            campaign_store_dir=store,
            output_dir=tmp_path / "package",
        )


def test_package_propagates_missing_source_as_blocked(tmp_path: Path) -> None:
    """Missing source artifacts never degrade into a fixture or diagnostic substitute."""
    source = _write_source(tmp_path)
    store = _write_store(tmp_path, source)
    candidate = _candidate(source)
    source.unlink()
    with pytest.raises(TraceDossierExportError):
        build_trace_dossier_package(
            candidates=[candidate],
            release_manifest_path=_RELEASE,
            campaign_store_dir=store,
            output_dir=tmp_path / "package",
        )


def test_package_cli_smoke_is_diagnostic_only(tmp_path: Path) -> None:
    """The public CLI composes a package without admitting benchmark evidence."""
    source = _write_source(tmp_path)
    store = _write_store(tmp_path, source)
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps([_candidate(source)]), encoding="utf-8")
    output = tmp_path / "package"
    completed = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--candidates",
            str(candidates),
            "--release-manifest",
            str(_RELEASE),
            "--campaign-store",
            str(store),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "wrote trace dossier package" in completed.stdout
    assert (output / "package_manifest.json").is_file()
