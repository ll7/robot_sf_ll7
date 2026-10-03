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


def test_runner_refuses_stale_mint_receipt_before_preflight(sealed_repository, monkeypatch, capsys):
    import json

    from robot_sf.evidence.writers import write_json
    from scripts.tools import run_benchmark_release as runner
    from tests.benchmark.test_sealed_source_pins import materialize

    repo = sealed_repository
    manifest = materialize(repo, "main")
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


def test_production_mint_verifier_refuses_stale_disclosure_receipt(sealed_repository, capsys):
    import json

    from robot_sf.evidence.writers import write_json
    from scripts.tools import resolve_benchmark_release_identity as resolver
    from tests.benchmark.test_sealed_source_pins import materialize

    repo = sealed_repository
    manifest = materialize(repo, "main")
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
