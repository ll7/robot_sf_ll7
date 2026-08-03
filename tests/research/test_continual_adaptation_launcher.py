"""Tests for the fail-closed research-lane continual-adaptation launcher (#6659).

The launcher gates on the merged protocol contract and then writes the bounded
adaptation plus the nominal/shift/forgetting evaluation surfaces as DIAGNOSTIC
outputs only. These tests cover the shipped example manifest path, the
diagnostic-only invariants (no promotion decision, no evidence bundle, no
benchmark/paper claim), the fail-closed gate, and the CLI entry point. No real
training, evaluation, checkpoint write, or network access is involved.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

import robot_sf.research.continual_adaptation_launcher as launcher_module
from robot_sf.research.continual_adaptation_launcher import (
    CONTINUAL_ADAPTATION_LAUNCHER_MODE,
    EVALUATION_SURFACES,
    build_adaptation_diagnostic,
    build_evaluation_diagnostics,
    get_continual_adaptation_output_root,
    render_markdown,
    run_continual_adaptation_diagnostics,
)
from robot_sf.research.continual_adaptation_protocol import (
    CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY,
    ContinualAdaptationProtocolError,
    load_continual_adaptation_run,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MANIFEST_PATH = (
    REPO_ROOT / "configs" / "training" / "continual_adaptation_run_issue_6582.yaml"
)
_TOOL = REPO_ROOT / "scripts" / "research" / "run_continual_adaptation_diagnostics.py"
_spec = importlib.util.spec_from_file_location("run_continual_adaptation_diagnostics", _TOOL)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _load_example_manifest() -> dict:
    """Load the shipped #6582 example manifest used as the launcher contract."""
    return load_continual_adaptation_run(EXAMPLE_MANIFEST_PATH)


def test_example_manifest_launch_writes_diagnostic_outputs(tmp_path: Path) -> None:
    """The shipped example manifest is protocol-valid and writes all diagnostics."""
    manifest = _load_example_manifest()
    output_dir = tmp_path / "diag"
    report = run_continual_adaptation_diagnostics(
        manifest, source=EXAMPLE_MANIFEST_PATH, output_dir=output_dir
    )

    assert report.protocol_status == "valid"
    assert report.launcher_mode == CONTINUAL_ADAPTATION_LAUNCHER_MODE
    assert report.evidence_boundary == CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY
    assert report.run_id == manifest["run_id"]
    assert report.derived_adapted_policy_identifier != report.baseline_policy_identifier

    for name in ("adaptation", *EVALUATION_SURFACES, "report"):
        out_file = output_dir / f"{name}.json"
        assert out_file.is_file()
        payload = json.loads(out_file.read_text(encoding="utf-8"))
        # Every output stamps the evidence boundary.
        assert payload["evidence_boundary"] == CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY

    assert report.output_files == [
        str(output_dir / f"{name}.json") for name in ("adaptation", *EVALUATION_SURFACES, "report")
    ]


def test_launcher_emits_no_promotion_or_evidence(tmp_path: Path) -> None:
    """The launcher never emits a promotion decision, evidence bundle, or claim."""
    report = run_continual_adaptation_diagnostics(
        _load_example_manifest(), source=EXAMPLE_MANIFEST_PATH, output_dir=tmp_path
    )
    assert report.emits_promotion_decision is False
    assert report.evidence_bundle_generated is False
    assert report.makes_benchmark_or_paper_claim is False

    report_payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report_payload["emits_promotion_decision"] is False
    assert report_payload["evidence_bundle_generated"] is False
    assert report_payload["makes_benchmark_or_paper_claim"] is False
    # No evidence bundle artifact is ever written.
    assert not (tmp_path / "evidence_bundle.json").exists()


def test_launcher_ignores_manifest_promotion_decision(tmp_path: Path) -> None:
    """Even a manifest asking to 'promote' yields no launcher promotion decision."""
    manifest = _load_example_manifest()
    manifest["promotion_decision"] = {"decision": "promote", "rationale": "ask for promotion"}
    # 'promote' without a complete results block fails closed at the protocol gate.
    with pytest.raises(ContinualAdaptationProtocolError):
        run_continual_adaptation_diagnostics(manifest, output_dir=tmp_path)


def test_adaptation_diagnostic_echoes_manifest() -> None:
    """The adaptation diagnostic echoes the declared bounded adaptation."""
    manifest = _load_example_manifest()
    diagnostic = build_adaptation_diagnostic(manifest)
    assert diagnostic["status"] == "diagnostic_only_not_executed"
    assert diagnostic["allowed_parameters"] == manifest["adaptation"]["allowed_parameters"]
    assert diagnostic["experience_budget"]["bounded"] is True
    assert diagnostic["experience_budget"]["steps"] == 200000
    assert diagnostic["experience_budget"]["units"] == "gradient_steps"
    assert diagnostic["adaptation_scenarios"] == manifest["scenarios"]["adaptation"]
    assert diagnostic["training_executed"] is False
    assert diagnostic["checkpoint_written"] is False
    assert diagnostic["safety_wrapper_mutated"] is False


def test_evaluation_diagnostics_cover_all_surfaces() -> None:
    """Nominal/shift/forgetting diagnostics echo thresholds; shift echoes shifts."""
    manifest = _load_example_manifest()
    evaluations = build_evaluation_diagnostics(manifest)
    assert set(evaluations) == set(EVALUATION_SURFACES)
    for surface in EVALUATION_SURFACES:
        record = evaluations[surface]
        assert record["status"] == "diagnostic_only_not_executed"
        assert record["evaluation_scenarios"] == manifest["scenarios"]["evaluation"]
        assert record["threshold"]["metric"] == manifest["thresholds"][surface]["metric"]
        assert record["metric_computed"] is False
        assert record["evidence"] is False
    # Only the shift surface carries the declared synthetic shifts.
    assert evaluations["shift"]["shifts"][0]["id"] == manifest["shifts"][0]["id"]
    assert "shifts" not in evaluations["nominal"]
    assert "shifts" not in evaluations["forgetting"]


def test_invalid_manifest_fails_closed_and_writes_nothing(tmp_path: Path) -> None:
    """A protocol-invalid manifest raises and writes no diagnostic output."""
    manifest = _load_example_manifest()
    manifest["safety_wrapper"]["mutation_permitted"] = True
    output_dir = tmp_path / "diag"
    with pytest.raises(ContinualAdaptationProtocolError, match="safety wrapper"):
        run_continual_adaptation_diagnostics(manifest, output_dir=output_dir)
    assert not output_dir.exists()


def test_overlapping_scenarios_fail_closed(tmp_path: Path) -> None:
    """Overlapping adaptation/evaluation scenario IDs fail closed."""
    manifest = _load_example_manifest()
    manifest["scenarios"]["evaluation"] = manifest["scenarios"]["adaptation"]
    with pytest.raises(ContinualAdaptationProtocolError, match="disjoint"):
        run_continual_adaptation_diagnostics(manifest, output_dir=tmp_path)


def test_schema_violation_raises(tmp_path: Path) -> None:
    """A schema-violating manifest fails closed at the protocol gate."""
    manifest = _load_example_manifest()
    del manifest["thresholds"]
    with pytest.raises(ContinualAdaptationProtocolError):
        run_continual_adaptation_diagnostics(manifest, output_dir=tmp_path)


def test_repo_local_output_override_stays_under_artifact_root() -> None:
    """Repository-local output overrides cannot target tracked paths."""
    output_dir = REPO_ROOT / "docs" / "continual_adaptation_diagnostics_test"
    with pytest.raises(ContinualAdaptationProtocolError, match="artifact root"):
        run_continual_adaptation_diagnostics(_load_example_manifest(), output_dir=output_dir)
    assert not output_dir.exists()


def test_repo_local_output_override_inside_artifact_root_is_allowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A repository-local override under the configured artifact root is allowed."""
    repository_root = tmp_path / "repo"
    artifact_root = repository_root / "output"
    output_dir = artifact_root / "continual_adaptation_diagnostics" / "run"
    monkeypatch.setattr(launcher_module, "_REPOSITORY_ROOT", repository_root)
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(artifact_root))

    report = run_continual_adaptation_diagnostics(_load_example_manifest(), output_dir=output_dir)

    assert output_dir.is_dir()
    assert report.output_files[0] == str(output_dir / "adaptation.json")


def test_default_output_root_honors_artifact_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default diagnostic root lives under the (overridable) artifact root."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", "/tmp/ca_artifact_root")
    assert get_continual_adaptation_output_root() == Path(
        "/tmp/ca_artifact_root/continual_adaptation_diagnostics"
    )


def test_default_output_rejects_run_id_that_escapes_artifact_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A manifest run ID cannot redirect the implicit output outside the artifact root."""
    repository_root = tmp_path / "repo"
    artifact_root = repository_root / "output"
    repository_root.mkdir()
    monkeypatch.setattr(launcher_module, "_REPOSITORY_ROOT", repository_root)
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(artifact_root))
    manifest = _load_example_manifest()
    manifest["run_id"] = "../../../escaped"

    with pytest.raises(ContinualAdaptationProtocolError, match="escapes the configured artifact"):
        run_continual_adaptation_diagnostics(manifest)

    assert not (tmp_path / "escaped").exists()


def test_render_markdown_states_diagnostic_boundary(tmp_path: Path) -> None:
    """The rendered summary states the diagnostic-only boundary."""
    report = run_continual_adaptation_diagnostics(_load_example_manifest(), output_dir=tmp_path)
    text = render_markdown(report)
    assert CONTINUAL_ADAPTATION_EVIDENCE_BOUNDARY in text
    assert "Diagnostic only" in text
    assert report.run_id in text


def test_script_main_valid_returns_0(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The CLI returns 0 and emits JSON for the shipped example manifest."""
    output_dir = tmp_path / "out"
    exit_code = mod.main(
        ["--manifest", str(EXAMPLE_MANIFEST_PATH), "--output-dir", str(output_dir), "--json"]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["protocol_status"] == "valid"
    assert payload["emits_promotion_decision"] is False
    assert (output_dir / "report.json").is_file()


def test_script_main_invalid_returns_1(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The CLI fails closed (exit 1) on a protocol-invalid manifest."""
    manifest = _load_example_manifest()
    manifest["safety_wrapper"]["mutation_permitted"] = True
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "out"
    exit_code = mod.main(["--manifest", str(bad_path), "--output-dir", str(output_dir)])
    assert exit_code == 1
    assert "failed closed" in capsys.readouterr().err
    assert not output_dir.exists()


def test_script_main_missing_manifest_returns_1(capsys: pytest.CaptureFixture) -> None:
    """The CLI fails closed (exit 1) when the manifest path does not exist."""
    exit_code = mod.main(["--manifest", "does/not/exist.yaml"])
    assert exit_code == 1
    assert "failed closed" in capsys.readouterr().err
