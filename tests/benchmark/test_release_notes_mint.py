"""Real production mint refusals, without running any environment."""

from functools import partial
from pathlib import Path

import pytest

from robot_sf.benchmark import release_protocol as protocol
from robot_sf.evidence.writers import write_text
from tests.benchmark.test_sealed_source_pins import git
from tests.benchmark.test_sealed_source_pins import sealed_repository as _sealed_repository

sealed_repository = _sealed_repository


write_text = partial(write_text, issue_ref="robot_sf#10110")


@pytest.mark.parametrize(
    "kind,expected", [("main", True), ("slice", True), ("runtime-smoke", False), ("0.0.7", False)]
)
def test_shared_release_detector_handles_real_native_and_archived_manifests(
    sealed_repository, kind, expected
):
    """Release identifiers, rather than borrowed scenario names, select notes admission."""
    import hashlib

    from robot_sf.benchmark.release_notes import is_release_0_0_8
    from tests.benchmark.test_sealed_source_pins import materialize

    repo = sealed_repository
    if kind in {"main", "slice"}:
        manifest = materialize(repo, kind)
    else:
        filename = (
            "paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_5.yaml"
            if kind == "runtime-smoke"
            else "benchmark_data_release_s30_h600.yaml"
        )
        path = repo / "configs/benchmarks/releases" / filename
        if kind == "0.0.7":
            # Exact manifest bytes retained at the published 0.0.7 Git tag.
            assert hashlib.sha256(path.read_bytes()).hexdigest() == (
                "6b2fd25d185e01c28a9883ccec6b3716ee996a052b44dd89cc89c16f6c80a182"
            )
        manifest = protocol.load_release_manifest(path)
    archived = protocol.build_resolved_release_manifest(manifest, repository_root=repo)
    assert "matrix_path" in archived["scenario"]
    assert "scenario_matrix" not in archived
    assert "scenario_matrix_path" not in archived
    assert is_release_0_0_8(manifest) is expected
    assert is_release_0_0_8(archived) is expected


def test_documented_runtime_smoke_preflight_is_not_notes_gated(monkeypatch, capsys, tmp_path):
    """The real v0_5 smoke can reach preflight without a production notes receipt."""
    import json

    from scripts.tools import run_benchmark_release as runner

    root = Path(__file__).resolve().parents[2]
    manifest_path = (
        "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_5.yaml"
    )
    # Admission and resolution use real files; only the ORCA availability check
    # and downstream preflight workers are replaced. No reset or step is needed.
    monkeypatch.setattr(runner, "check_orca_rvo2_preflight", lambda *_a: None)
    monkeypatch.setattr(
        runner, "_run_spawn_matrix_preflight", lambda **_kw: ({}, {"status": "valid"})
    )
    reached = []

    def prepare(cfg, **kwargs):
        reached.append(cfg)
        return {
            "campaign_id": kwargs["campaign_id"],
            **{
                key: tmp_path / key
                for key in (
                    "campaign_root",
                    "validate_config_path",
                    "preview_scenarios_path",
                    "matrix_summary_json_path",
                    "matrix_summary_csv_path",
                )
            },
        }

    def refuse_execution(*_a, **_kw):
        pytest.fail("runtime-smoke preflight attempted campaign execution")

    monkeypatch.setattr(runner, "prepare_campaign_preflight", prepare)
    monkeypatch.setattr(runner, "run_campaign", refuse_execution)
    monkeypatch.chdir(root)
    # The manifest and mode match docs/RELEASE.md's documented preflight command.
    exit_code = runner.main(["--manifest", manifest_path, "--mode", "preflight"])
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("status") != "release_notes_refused", payload
    assert exit_code == 0, payload
    assert len(reached) == 1
    assert payload["manifest_validation"]["status"] == "valid"
    assert "release_notes_gate" not in payload["resolved_manifest"]


def test_production_mint_refuses_missing_disclosure(sealed_repository):
    repo = sealed_repository
    notes = repo / "docs/release/0.0.8/release_notes.md"
    # On base, supply the new document as a tracked input without the checker.
    if not notes.exists():
        notes.parent.mkdir(parents=True)
    write_text(notes, "# Release 0.0.8 notes\n\nNo speed disclosure.\n")
    git(repo, "add", "docs/release/0.0.8/release_notes.md")
    git(repo, "commit", "-qm", "remove required speed disclosure")
    source = git(repo, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="D-039: missing required statement"):
        protocol.write_resolved_release_identity(
            template_path=Path(
                "configs/benchmarks/releases/benchmark_data_release_s30_h600.template.yaml"
            ),
            output_path=Path("output/mint/release_identity.resolved.json"),
            source_commit=source,
            release_tag=f"paper-matrix-v2-h600-s30-{source}",
            concept_doi="10.5281/zenodo.99000001",
            version_doi="10.5281/zenodo.99000002",
            repository_root=repo,
        )
    assert not (repo / "output/mint/release_identity.resolved.json").exists()


@pytest.mark.parametrize("kind", ["main", "slice"])
def test_runner_refuses_stale_mint_receipt_before_preflight(
    sealed_repository, monkeypatch, capsys, kind
):
    import json

    from robot_sf.evidence.writers import write_json
    from scripts.tools import run_benchmark_release as runner
    from tests.benchmark.test_sealed_source_pins import materialize

    repo = sealed_repository
    manifest = materialize(repo, kind)
    capsys.readouterr()
    receipt_path = manifest.path.parent / "release_notes_gate.v1.json"
    receipt = json.loads(receipt_path.read_bytes()) if receipt_path.exists() else {}
    receipt["notes_sha256"] = "0" * 64
    write_json(receipt_path, receipt)

    def abort(*args, **kwargs):
        pytest.fail("preflight or execution reached after stale disclosure receipt")

    monkeypatch.setattr(runner, "check_orca_rvo2_preflight", lambda *_a: None)
    monkeypatch.setattr(runner, "_run_spawn_matrix_preflight", abort)
    monkeypatch.setattr(runner, "prepare_campaign_preflight", abort)
    monkeypatch.setattr(runner, "run_campaign", abort)
    assert runner.main(["--manifest", str(manifest.path), "--mode", "preflight"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "release_notes_refused"
    assert "digest mismatch against mint receipt" in payload["status_reason"]


@pytest.mark.parametrize("kind", ["main", "slice"])
def test_production_mint_verifier_refuses_stale_disclosure_receipt(sealed_repository, capsys, kind):
    import json

    from robot_sf.evidence.writers import write_json
    from scripts.tools import resolve_benchmark_release_identity as resolver
    from tests.benchmark.test_sealed_source_pins import materialize

    repo = sealed_repository
    manifest = materialize(repo, kind)
    capsys.readouterr()
    path = manifest.path.parent / "release_notes_gate.v1.json"
    receipt = json.loads(path.read_bytes()) if path.exists() else {}
    receipt["notes_sha256"] = "0" * 64
    write_json(path, receipt)
    # This is the public verifier argv used by private production mint's
    # verify_materialized_manifest, not a separate test-only admission seam.
    assert (
        resolver.main(["verify", "--identity", str(manifest.path), "--repository-root", str(repo)])
        == 2
    )
    rejected = json.loads(capsys.readouterr().out)
    assert rejected["status"] == "rejected"
    assert "digest mismatch against mint receipt" in rejected["reason"]


@pytest.mark.parametrize("kind", ["main", "slice"])
def test_public_export_refuses_edited_notes(sealed_repository, monkeypatch, kind):
    """Exercise the exporter call site with real minted main and slice metadata."""
    import json

    from robot_sf.benchmark import artifact_publication as publication
    from robot_sf.evidence.writers import write_json
    from tests.benchmark.test_artifact_publication import _make_run
    from tests.benchmark.test_sealed_source_pins import materialize

    repo = sealed_repository
    manifest = materialize(repo, kind)
    resolved = protocol.build_resolved_release_manifest(manifest)
    receipt_path = manifest.path.parent / "release_notes_gate.v1.json"
    resolved["release_notes_gate"] = (
        json.loads(receipt_path.read_bytes()) if receipt_path.exists() else {}
    )
    run = repo / "run"
    _make_run(run, with_video=False)
    (run / "release").mkdir()
    write_json(run / "release/release_manifest.resolved.json", resolved)
    write_json(
        run / "release/release_result.json", {"status": "ok", "source_commit": manifest.source_sha}
    )
    monkeypatch.setattr(publication, "get_repository_root", lambda: repo)
    notes = repo / "docs/release/0.0.8/release_notes.md"
    original = notes.read_text()
    write_text(notes, original + "\nEdited after mint.\n")
    with pytest.raises(ValueError, match="digest mismatch against mint receipt"):
        publication.export_publication_bundle(run, repo / "bundles", bundle_name="stale")
    assert not (repo / "bundles/stale").exists()
    # Restore exactly the minted bytes, including their existing review marker.
    from shutil import copyfile

    from tests.benchmark.test_sealed_source_pins import ROOT

    copyfile(ROOT / "docs/release/0.0.8/release_notes.md", notes)
    result = publication.export_publication_bundle(run, repo / "bundles", bundle_name="valid")
    assert (
        result.bundle_dir / "payload/release_metadata/release_notes.md"
    ).read_bytes() == notes.read_bytes()


@pytest.mark.parametrize("kind", ["main", "slice"])
def test_public_zenodo_publish_refuses_edited_notes(sealed_repository, monkeypatch, capsys, kind):
    """The public CLI must refuse before constructing an authenticated session."""
    from robot_sf import release_cli
    from tests.benchmark.test_release_cli_edge_cases import _args
    from tests.benchmark.test_sealed_source_pins import materialize

    repo = sealed_repository
    manifest = materialize(repo, kind)
    args = _args("publish", repo)
    args.manifest = manifest.path
    args.metadata = None
    monkeypatch.setattr(release_cli, "get_repository_root", lambda: repo)
    publisher = release_cli.zenodo_publisher
    sessions = []
    published = []
    monkeypatch.setattr(publisher, "build_session", lambda _path: sessions.append(True) or object())
    monkeypatch.setattr(publisher, "load_state", lambda _path: {})
    monkeypatch.setattr(publisher, "load_dataset_metadata", lambda *_a, **_kw: {})
    monkeypatch.setattr(publisher, "publish", lambda *_a, **_kw: published.append(True) or {})
    monkeypatch.setattr(publisher, "write_state", lambda *_a: None)
    assert release_cli.handle(args) == 0
    assert sessions == published == [True]
    sessions.clear()
    published.clear()
    capsys.readouterr()
    notes = repo / "docs/release/0.0.8/release_notes.md"
    # An index flag can hide edited notes from the generic clean-source guard.
    # Disclosure admission must inspect actual bytes regardless of git status.
    git(repo, "update-index", "--assume-unchanged", "docs/release/0.0.8/release_notes.md")
    write_text(notes, notes.read_text() + "\nEdited after mint.\n")
    assert git(repo, "status", "--porcelain") == ""
    assert release_cli.handle(args) == 2
    assert sessions == published == []
    result = __import__("json").loads(capsys.readouterr().out)
    assert result["status"] == "blocked"
    assert "digest mismatch against mint receipt" in result["reason"]


@pytest.mark.parametrize("kind", ["main", "slice"])
def test_cold_preflight_refuses_edited_notes(sealed_repository, monkeypatch, kind):
    """A resigned bundle still has to match the main or slice mint receipt."""
    import json
    import shutil

    from robot_sf.benchmark import artifact_publication as publication
    from robot_sf.evidence.writers import write_json
    from tests.benchmark.test_release_notes_gate import _resign_notes
    from tests.benchmark.test_sealed_source_pins import materialize
    from tests.validation.test_publication_preflight import _build_bundle

    repo = sealed_repository
    manifest = materialize(repo, kind)
    resolved = protocol.build_resolved_release_manifest(manifest)
    receipt_path = manifest.path.parent / "release_notes_gate.v1.json"
    receipt = json.loads(receipt_path.read_bytes()) if receipt_path.exists() else {}
    bundle = _build_bundle(repo / "cold-test", publication_commit=manifest.source_sha)
    payload = bundle / "payload"
    target = payload / "release_metadata"
    target.mkdir()
    notes = target / "release_notes.md"
    shutil.copy2(repo / "docs/release/0.0.8/release_notes.md", notes)
    write_json(target / "release_notes_gate.v1.json", receipt)
    resolved["release_notes_gate"] = json.loads(
        (target / "release_notes_gate.v1.json").read_bytes()
    )
    write_json(payload / "release/release_manifest.resolved.json", resolved)
    monkeypatch.setattr(publication, "get_repository_root", lambda: repo)
    # No publication metadata contract is fabricated: this witnesses the cold
    # notes route independently of metric reconciliation and role discovery.
    _resign_notes(bundle, notes)
    assert publication.verify_publication_bundle_preflight(bundle)["status"] == "pass"
    write_text(notes, notes.read_text() + "\nEdited exported notes.\n")
    _resign_notes(bundle, notes)
    with pytest.raises(
        publication.PublicationPreflightError, match="digest mismatch against mint receipt"
    ):
        publication.verify_publication_bundle_preflight(bundle)
