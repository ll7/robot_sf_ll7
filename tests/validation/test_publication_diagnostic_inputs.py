"""Publication bundles must retain the exact file inputs named by v2 diagnostics."""

import hashlib
import json

import pytest

from robot_sf.benchmark.artifact_publication import (
    PublicationPreflightError,
    export_publication_bundle,
    verify_publication_bundle_preflight,
)


def _run(tmp_path, monkeypatch):
    """Make a small public export with repository-relative diagnostic inputs."""
    repo = tmp_path / "source"
    repo.mkdir()
    monkeypatch.setattr("robot_sf.benchmark.artifact_publication.get_repository_root", lambda: repo)
    run = repo / "output/campaign"
    (run / "reports").mkdir(parents=True)
    (run / "run_meta.json").write_text(json.dumps({"repo": {"commit": "abc123"}}))
    (run / "runs/goal").mkdir(parents=True)
    (run / "runs/goal/episodes.jsonl").write_text(
        json.dumps({"event_ledger": {"software_commit": "abc123"}}) + "\n"
    )
    provenance = {}
    for name, rel in [
        ("weights", "configs/weights.json"),
        ("anchors", "output/calibration/anchors.json"),
        ("family", "configs/family.yaml"),
        ("extra", "configs/input.json"),
    ]:
        source = repo / rel
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("fixture-" + name)
        provenance[f"snqi_v2_{name}_path"] = rel
        provenance[f"snqi_v2_{name}_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (run / "reports/snqi_v2_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": "snqi-v2-diagnostics.v1",
                "provenance": provenance,
            }
        )
    )
    return repo, run, provenance


def _export(run, tmp_path):
    return export_publication_bundle(
        run, tmp_path / "bundles", release_tag="test-v2", doi="10.5281/zenodo.1234567"
    )


def test_export_includes_every_path_digest_diagnostic_input(tmp_path, monkeypatch):
    """Anchor paths outside the campaign and future additional inputs survive a cold export."""
    repo, run, provenance = _run(tmp_path, monkeypatch)
    bundle = _export(run, tmp_path)
    manifest = json.loads(bundle.manifest_path.read_text())
    entries = {e["path"]: e["sha256"] for e in manifest["files"]}
    for key, rel in provenance.items():
        if not key.endswith("_path"):
            continue
        bundled = "diagnostic_inputs/" + rel
        assert (bundle.bundle_dir / "payload" / bundled).read_bytes() == (repo / rel).read_bytes()
        assert entries[bundled] == provenance[key[:-5] + "_sha256"]
        assert f"{entries[bundled]}  payload/{bundled}" in bundle.checksums_path.read_text()


@pytest.mark.parametrize("mutation", ["missing", "wrong_digest", "unsafe", "no_digest", "symlink"])
def test_export_refuses_unrecoverable_diagnostic_input(tmp_path, monkeypatch, mutation):
    """Exporter cannot produce a future bundle with missing or untrusted input bytes."""
    repo, run, provenance = _run(tmp_path, monkeypatch)
    source = repo / provenance["snqi_v2_anchors_path"]
    if mutation == "missing":
        source.unlink()
    elif mutation == "wrong_digest":
        source.write_text("changed")
    elif mutation == "unsafe":
        provenance["snqi_v2_anchors_path"] = "../outside.json"
    elif mutation == "no_digest":
        del provenance["snqi_v2_anchors_sha256"]
    else:
        source.unlink()
        source.symlink_to(repo / "configs/weights.json")
    (run / "reports/snqi_v2_diagnostics.json").write_text(
        json.dumps(
            {
                "schema_version": "snqi-v2-diagnostics.v1",
                "provenance": provenance,
            }
        )
    )
    with pytest.raises(ValueError, match="diagnostic input"):
        _export(run, tmp_path)


def test_preflight_refuses_missing_input_even_after_resigning_inventory(tmp_path, monkeypatch):
    """Deleting and re-signing an input cannot bypass diagnostic referential integrity."""
    _, run, _ = _run(tmp_path, monkeypatch)
    bundle = _export(run, tmp_path)
    rel = "diagnostic_inputs/output/calibration/anchors.json"
    path = bundle.bundle_dir / "payload" / rel
    path.unlink(missing_ok=True)
    manifest = json.loads(bundle.manifest_path.read_text())
    manifest["files"] = [e for e in manifest["files"] if e["path"] != rel]
    bundle.manifest_path.write_text(json.dumps(manifest))
    bundle.checksums_path.write_text(
        "".join(f"{e['sha256']}  payload/{e['path']}\n" for e in manifest["files"])
    )
    with pytest.raises(PublicationPreflightError, match="diagnostic input.*missing"):
        verify_publication_bundle_preflight(bundle.bundle_dir, require_release_reconciliation=False)


def test_cold_preflight_accepts_without_original_input_locations(tmp_path, monkeypatch):
    """Detached input custody remains verifiable after original source files disappear."""
    repo, run, provenance = _run(tmp_path, monkeypatch)
    bundle = _export(run, tmp_path)
    for key, rel in provenance.items():
        if key.endswith("_path"):
            (repo / rel).unlink()
    report = verify_publication_bundle_preflight(
        bundle.bundle_dir, require_release_reconciliation=False
    )
    assert report["status"] == "pass"


@pytest.mark.parametrize("mutation", ["changed", "unsigned"])
def test_cold_preflight_rejects_re_signed_input_substitution(tmp_path, monkeypatch, mutation):
    """A self-consistent checksum inventory cannot override diagnostic input provenance."""
    _, run, _ = _run(tmp_path, monkeypatch)
    bundle = _export(run, tmp_path)
    rel = "diagnostic_inputs/output/calibration/anchors.json"
    path = bundle.bundle_dir / "payload" / rel
    manifest = json.loads(bundle.manifest_path.read_text())
    if mutation == "changed":
        path.write_text("replacement anchors")
        for entry in manifest["files"]:
            if entry["path"] == rel:
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        manifest["files"] = [e for e in manifest["files"] if e["path"] != rel]
    bundle.manifest_path.write_text(json.dumps(manifest))
    bundle.checksums_path.write_text(
        "".join(f"{e['sha256']}  payload/{e['path']}\n" for e in manifest["files"])
    )
    with pytest.raises(PublicationPreflightError, match="diagnostic input.*(digest|signed)"):
        verify_publication_bundle_preflight(bundle.bundle_dir, require_release_reconciliation=False)
