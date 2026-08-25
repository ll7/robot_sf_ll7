"""Focused tests for the preserved-row release recovery helper."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark import release_acceptance
from scripts.tools import revalidate_benchmark_release as recovery


def _sha256(path: Path) -> str:
    """Return a fixture file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str) -> None:
    """Write a UTF-8 fixture file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _make_verified_retrieval(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Create a small retrieval with the production 109-file contract scaled to two files."""
    root = tmp_path / "retrieval"
    _write(root / "payload.txt", "episode payload\n")
    _write(
        root / "release" / "release_result.json",
        '{"release_status":"full_release_acceptance_failed"}\n',
    )
    entries = {
        "payload.txt": _sha256(root / "payload.txt"),
        "release/release_result.json": _sha256(root / "release" / "release_result.json"),
    }
    _write(root / "SHA256SUMS", "".join(f"{digest}  {path}\n" for path, digest in entries.items()))
    receipt = {
        "status": "verified",
        "file_count": len(entries),
        "manifest_sha256": _sha256(root / "SHA256SUMS"),
        "files": [{"path": path, "sha256": digest} for path, digest in entries.items()],
        "verified_at": "2026-08-25T10:15:34Z",
    }
    _write(root / "artifact-verification-receipt.json", json.dumps(receipt) + "\n")
    return root, entries


def _make_preserved_receipt(root: Path, *, mutate: dict[str, object] | None = None) -> Path:
    """Gzip a receipt payload for the preserved-receipt seam."""
    payload = json.loads((root / "artifact-verification-receipt.json").read_text())
    payload["verified_at"] = "2026-08-25T10:00:00Z"
    if mutate:
        payload.update(mutate)
    path = root.parent / "preserved-receipt.json.gz"
    path.write_bytes(gzip.compress(json.dumps(payload, sort_keys=True).encode(), mtime=0))
    return path


def _verify_with_preserved(root: Path, preserved: Path) -> dict[str, object]:
    """Verify a fixture using the two-receipt refresh contract."""
    return recovery.verify_producer_artifacts(
        root,
        expected_sums_sha256=_sha256(root / "SHA256SUMS"),
        expected_receipt_sha256=_sha256(root / "artifact-verification-receipt.json"),
        expected_preserved_receipt_sha256=recovery._sha256_bytes(
            gzip.decompress(preserved.read_bytes())
        ),
        expected_rejected_result_sha256=_sha256(root / "release/release_result.json"),
        expected_file_count=2,
        preserved_receipt_source=preserved,
    )


def test_verify_producer_artifacts_checks_manifest_and_receipt(tmp_path: Path) -> None:
    """A complete retrieval passes only when all three digest bindings agree."""
    root, _ = _make_verified_retrieval(tmp_path)
    report = recovery.verify_producer_artifacts(
        root,
        expected_sums_sha256=_sha256(root / "SHA256SUMS"),
        expected_receipt_sha256=_sha256(root / "artifact-verification-receipt.json"),
        expected_rejected_result_sha256=_sha256(root / "release/release_result.json"),
        expected_file_count=2,
    )
    assert report["listed_file_count"] == 2
    assert report["total_file_count"] == 4
    assert set(report["files"]) == {"payload.txt", "release/release_result.json"}


def test_verify_producer_artifacts_accepts_preserved_receipt_refresh(tmp_path: Path) -> None:
    """The original receipt is accepted when only verified_at was refreshed."""
    root, _ = _make_verified_retrieval(tmp_path)
    preserved = _make_preserved_receipt(root)
    report = _verify_with_preserved(root, preserved)
    assert report["preserved_artifact_verification_receipt_sha256"] == recovery._sha256_bytes(
        gzip.decompress(preserved.read_bytes())
    )
    assert report["artifact_receipt_refresh"]["difference_paths"] == ["verified_at"]
    assert report["_preserved_receipt_bytes"] == gzip.decompress(preserved.read_bytes())


def test_verify_producer_artifacts_rejects_non_timestamp_receipt_drift(tmp_path: Path) -> None:
    """A semantic receipt change outside verified_at is not a refresh."""
    root, _ = _make_verified_retrieval(tmp_path)
    preserved = _make_preserved_receipt(root, mutate={"artifact_contract": "tampered"})
    with pytest.raises(recovery.DerivedReleaseError, match="outside verified_at"):
        _verify_with_preserved(root, preserved)


@pytest.mark.parametrize(
    "kind",
    ["bad-gzip", "trailing-data", "second-member", "tampered-payload"],
)
def test_verify_producer_artifacts_rejects_invalid_preserved_gzip(
    tmp_path: Path, kind: str
) -> None:
    """Invalid, concatenated, trailing, and tampered gzip inputs fail closed."""
    root, _ = _make_verified_retrieval(tmp_path)
    good = _make_preserved_receipt(root)
    good_bytes = good.read_bytes()
    payload = gzip.decompress(good_bytes)
    preserved = root.parent / f"{kind}.json.gz"
    if kind == "bad-gzip":
        preserved.write_bytes(b"not gzip")
    elif kind == "trailing-data":
        preserved.write_bytes(good_bytes + b"trailing")
    elif kind == "second-member":
        preserved.write_bytes(good_bytes + good_bytes)
    else:
        tampered = json.loads(payload)
        tampered["artifact_contract"] = "tampered"
        preserved.write_bytes(gzip.compress(json.dumps(tampered).encode(), mtime=0))
    with pytest.raises(recovery.DerivedReleaseError):
        recovery.verify_producer_artifacts(
            root,
            expected_sums_sha256=_sha256(root / "SHA256SUMS"),
            expected_receipt_sha256=_sha256(root / "artifact-verification-receipt.json"),
            expected_preserved_receipt_sha256=recovery._sha256_bytes(payload),
            expected_rejected_result_sha256=_sha256(root / "release/release_result.json"),
            expected_file_count=2,
            preserved_receipt_source=preserved,
        )


def test_verify_producer_artifacts_rejects_tamper_and_unlisted_file(tmp_path: Path) -> None:
    """Mutation and inventory drift fail before a derived copy can start."""
    root, _ = _make_verified_retrieval(tmp_path)
    expected = {
        "expected_sums_sha256": _sha256(root / "SHA256SUMS"),
        "expected_receipt_sha256": _sha256(root / "artifact-verification-receipt.json"),
        "expected_rejected_result_sha256": _sha256(root / "release/release_result.json"),
        "expected_file_count": 2,
    }
    _write(root / "payload.txt", "tampered\n")
    with pytest.raises(recovery.DerivedReleaseError, match="checksum mismatch"):
        recovery.verify_producer_artifacts(root, **expected)

    root, _ = _make_verified_retrieval(tmp_path / "extra")
    _write(root / "unlisted.txt", "must block\n")
    with pytest.raises(recovery.DerivedReleaseError, match="file inventory mismatch"):
        recovery.verify_producer_artifacts(
            root,
            expected_sums_sha256=_sha256(root / "SHA256SUMS"),
            expected_receipt_sha256=_sha256(root / "artifact-verification-receipt.json"),
            expected_rejected_result_sha256=_sha256(root / "release/release_result.json"),
            expected_file_count=2,
        )


def test_verify_producer_artifacts_rejects_symlink(tmp_path: Path) -> None:
    """A symlink is never followed into a supposedly immutable producer tree."""
    root, _ = _make_verified_retrieval(tmp_path)
    (root / "symlink.txt").symlink_to(root / "payload.txt")
    with pytest.raises(recovery.DerivedReleaseError, match="symlink"):
        recovery.verify_producer_artifacts(
            root,
            expected_sums_sha256=_sha256(root / "SHA256SUMS"),
            expected_receipt_sha256=_sha256(root / "artifact-verification-receipt.json"),
            expected_rejected_result_sha256=_sha256(root / "release/release_result.json"),
            expected_file_count=2,
        )


def test_sanitise_tree_paths_removes_private_absolute_paths(tmp_path: Path) -> None:
    """Public projections retain portable paths and redact unrelated machine paths."""
    source_root = tmp_path / "source"
    producer_root = tmp_path / "producer"
    copied = tmp_path / "copied"
    copied.mkdir()
    path = copied / "metadata.json"
    _write(
        path,
        json.dumps(
            {
                "source": str(source_root / "configs/scenarios/matrix.yaml"),
                "campaign": str(producer_root / "runs/arm/episodes.jsonl"),
                "private": "/tmp/secret/job.log",
            }
        ),
    )
    recovery._sanitise_tree_paths(copied, source_root=source_root, producer_root=producer_root)
    value = path.read_text(encoding="utf-8")
    assert "configs/scenarios/matrix.yaml" in value
    assert "runs/arm/episodes.jsonl" in value
    assert "/tmp/" not in value
    recovery._assert_no_private_absolute_paths(copied)


def test_copy_projection_carries_preserved_and_refreshed_receipts(tmp_path: Path) -> None:
    """The derived provenance keeps both receipt payloads without mutating input."""
    root, _ = _make_verified_retrieval(tmp_path)
    preserved = _make_preserved_receipt(root)
    evidence = _verify_with_preserved(root, preserved)
    copied = tmp_path / "copied"
    copy_file_map = dict(evidence["file_map"])
    copy_file_map["artifact-verification-receipt.json"] = {
        "bytes": len(evidence["_current_receipt_bytes"]),
        "sha256": evidence["artifact_verification_receipt_sha256"],
    }
    recovery._copy_producer_projection(
        root,
        staging_root=copied,
        source_root=tmp_path / "source",
        expected_file_map=copy_file_map,
        current_receipt_bytes=evidence["_current_receipt_bytes"],
        preserved_receipt_bytes=evidence["_preserved_receipt_bytes"],
    )
    preserved_copy = json.loads(
        (copied / "provenance/producer_artifact_verification_receipt.json").read_text()
    )
    current_copy = json.loads(
        (copied / "provenance/current_producer_artifact_verification_receipt.json").read_text()
    )
    assert preserved_copy["verified_at"] == "2026-08-25T10:00:00Z"
    assert current_copy["verified_at"] == "2026-08-25T10:15:34Z"
    assert (root / "artifact-verification-receipt.json").exists()


def test_manifest_assets_must_rebind_to_frozen_source_root(tmp_path: Path) -> None:
    """Absolute helper-checkout assets cannot enter a frozen-source release."""
    source = tmp_path / "source"
    source.mkdir()
    manifest_path = source / "manifest.yaml"
    config_path = tmp_path / "helper" / "config.yaml"
    _write(manifest_path, "manifest\n")
    _write(config_path, "helper config\n")
    manifest = SimpleNamespace(
        path=manifest_path,
        canonical_campaign_config_path=config_path,
    )
    with pytest.raises(recovery.DerivedReleaseError, match="frozen source repository"):
        recovery._assert_manifest_paths_from_source(manifest, source)


def test_artifact_root_symlink_is_rejected(tmp_path: Path) -> None:
    """A symlink at the supplied root cannot become an immutable input."""
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(recovery.DerivedReleaseError, match="symlink"):
        recovery._assert_safe_directory(linked, label="producer root")


def test_source_binding_redirects_all_relative_asset_resolvers(tmp_path: Path) -> None:
    """Protocol, config, acceptance, and publication resolvers share frozen roots."""
    source = tmp_path / "source"
    validator = tmp_path / "validator"
    source.mkdir()
    validator.mkdir()
    from robot_sf.benchmark import artifact_publication, release_acceptance

    with recovery._source_repository_binding(source, validator_root=validator):
        assert recovery.release_protocol_module.get_repository_root() == source
        assert recovery.camera_config_module.get_repository_root() == source
        assert artifact_publication.get_repository_root() == source
        assert release_acceptance.get_repository_root() == validator


def test_cross_root_file_map_rejects_size_or_digest_drift() -> None:
    """Accepted and published campaign maps must match path, size, and SHA."""
    accepted = {"file_map": {"row.json": {"bytes": 10, "sha256": "a" * 64}}}
    retrieved = {"file_map": {"row.json": {"bytes": 11, "sha256": "a" * 64}}}
    with pytest.raises(recovery.DerivedReleaseError, match="file map mismatch"):
        recovery._assert_equal_file_maps(accepted, retrieved)


def test_staged_copy_rejects_source_revert_or_copy_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every copied raw file is checked immediately against the admitted map."""
    root, _ = _make_verified_retrieval(tmp_path)
    all_paths = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]
    expected = recovery._build_file_map(root, all_paths)
    original_copy = shutil.copy2

    def copy_and_tamper(source: Path, destination: Path, *args: object, **kwargs: object):
        if Path(source).name == "payload.txt":
            Path(source).write_text("reverted source bytes\n", encoding="utf-8")
        result = original_copy(source, destination, *args, **kwargs)
        return result

    monkeypatch.setattr(recovery.shutil, "copy2", copy_and_tamper)
    with pytest.raises(recovery.DerivedReleaseError, match="staged copy does not match"):
        recovery._copy_tree_without_symlinks(
            root,
            tmp_path / "copied",
            expected_file_map=expected,
        )


def test_validator_provenance_requires_exact_reviewed_commit(tmp_path: Path) -> None:
    """A validator checkout at another commit is never accepted."""
    with pytest.raises(recovery.DerivedReleaseError, match="expected reviewed commit"):
        recovery._validator_provenance(
            Path(release_acceptance.__file__).parents[2],
            expected_commit="0" * 40,
        )


def test_build_derived_release_cleans_partial_stage_on_bundle_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed export leaves neither a final target nor a hidden partial stage."""
    producer, _ = _make_verified_retrieval(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "manifest.yaml"
    _write(manifest, "manifest\n")
    config_path = source / "config.yaml"
    _write(config_path, "config\n")
    _write(
        producer / "release/release_manifest.resolved.json",
        json.dumps(
            {
                "release_tag": "paper-matrix-v2-h600-s30-2026-08-test",
                "provenance": {
                    "version_doi": "10.5281/zenodo.1",
                    "repository_url": "https://github.com/ll7/robot_sf_ll7",
                },
            }
        ),
    )
    _write(producer / "reports/campaign_summary.json", json.dumps({"campaign": {}}))
    _write(producer / "reports/campaign_report.md", "# report\n")
    fixture_file_map = {
        path.relative_to(producer).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in producer.rglob("*")
        if path.is_file() and path.name != "artifact-verification-receipt.json"
    }
    # The fixture's manifest/result hashes are intentionally injected through
    # the validation seam; the production command uses the admitted constants.
    monkeypatch.setattr(
        recovery,
        "verify_producer_artifacts",
        lambda _root, **_kwargs: {
            "status": "verified",
            "listed_file_count": 2,
            "total_file_count": 4,
            "sha256sums_sha256": "a" * 64,
            "artifact_verification_receipt_sha256": _sha256(
                producer / "artifact-verification-receipt.json"
            ),
            "rejected_release_result_sha256": "c" * 64,
            "files": {},
            "file_map": fixture_file_map,
            "_current_receipt_bytes": (_root / "artifact-verification-receipt.json").read_bytes(),
            "_preserved_receipt_bytes": None,
        },
    )
    monkeypatch.setattr(
        recovery,
        "EXPECTED_REJECTED_RESULT_SHA256",
        _sha256(producer / "release/release_result.json"),
    )
    monkeypatch.setattr(
        recovery,
        "load_release_manifest",
        lambda _path: SimpleNamespace(canonical_campaign_config_path=config_path),
    )
    monkeypatch.setattr(recovery, "load_campaign_config", lambda _path: SimpleNamespace())
    monkeypatch.setattr(
        recovery,
        "validate_full_benchmark_release_acceptance",
        lambda *_args, **_kwargs: {
            "status": "valid",
            "source_commits": [recovery.FROZEN_SOURCE_SHA],
        },
    )
    monkeypatch.setattr(
        recovery,
        "_validator_provenance",
        lambda _root, **_kwargs: {
            "commit": "d" * 40,
            "file": "robot_sf/benchmark/release_acceptance.py",
            "file_sha256": _sha256(Path(release_acceptance.__file__)),
        },
    )
    monkeypatch.setattr(
        recovery,
        "_verify_campaign_file_map",
        lambda *_args, **_kwargs: {
            "status": "verified",
            "file_map": fixture_file_map,
        },
    )
    monkeypatch.setattr(
        recovery,
        "validate_release_manifest",
        lambda *_args, **_kwargs: {"status": "valid", "problems": []},
    )
    monkeypatch.setattr(recovery, "_assert_frozen_source_repository", lambda *_args: None)
    monkeypatch.setattr(recovery, "write_campaign_report", lambda *_args, **_kwargs: None)

    def fail_export(*_args, **_kwargs):
        raise recovery.PublicationPreflightError("test export failure")

    monkeypatch.setattr(recovery, "export_publication_bundle", fail_export)
    output_root = tmp_path / "output"
    with pytest.raises(recovery.PublicationPreflightError, match="test export failure"):
        recovery.build_derived_release(
            producer_root=producer,
            acceptance_root=producer,
            source_repository_root=source,
            validator_repository_root=source,
            expected_validator_commit="d" * 40,
            manifest_path=manifest,
            output_root=output_root,
            derived_name="derived",
        )
    assert not (output_root / "derived").exists()
    assert not list(output_root.glob(".derived.staging-*"))


def test_build_derived_release_successfully_promotes_complete_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The build path promotes one complete campaign/publication snapshot atomically."""
    producer, _ = _make_verified_retrieval(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "manifest.yaml"
    config_path = source / "config.yaml"
    _write(manifest, "manifest\n")
    _write(config_path, "config\n")
    _write(
        producer / "release/release_manifest.resolved.json",
        json.dumps(
            {
                "release_tag": "paper-matrix-v2-h600-s30-2026-08-test",
                "provenance": {
                    "version_doi": "10.5281/zenodo.1",
                    "repository_url": "https://github.com/ll7/robot_sf_ll7",
                },
            }
        ),
    )
    _write(producer / "reports/campaign_summary.json", json.dumps({"campaign": {}}))
    _write(producer / "reports/campaign_report.md", "# report\n")
    fixture_file_map = {
        path.relative_to(producer).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in producer.rglob("*")
        if path.is_file() and path.name != "artifact-verification-receipt.json"
    }
    current_receipt = (producer / "artifact-verification-receipt.json").read_bytes()
    verifier_result = {
        "status": "verified",
        "listed_file_count": 2,
        "total_file_count": 4,
        "sha256sums_sha256": _sha256(producer / "SHA256SUMS"),
        "artifact_verification_receipt_sha256": _sha256(
            producer / "artifact-verification-receipt.json"
        ),
        "rejected_release_result_sha256": _sha256(producer / "release/release_result.json"),
        "files": {},
        "file_map": fixture_file_map,
        "artifact_receipt_refresh": None,
        "preserved_artifact_verification_receipt_sha256": None,
        "_current_receipt_bytes": current_receipt,
        "_preserved_receipt_bytes": None,
    }
    monkeypatch.setattr(recovery, "verify_producer_artifacts", lambda *_a, **_k: verifier_result)
    monkeypatch.setattr(
        recovery,
        "_verify_campaign_file_map",
        lambda *_a, **_k: {"status": "verified", "file_map": fixture_file_map},
    )
    monkeypatch.setattr(
        recovery,
        "load_release_manifest",
        lambda _path: SimpleNamespace(canonical_campaign_config_path=config_path),
    )
    monkeypatch.setattr(recovery, "load_campaign_config", lambda _path: SimpleNamespace())
    monkeypatch.setattr(
        recovery,
        "validate_release_manifest",
        lambda *_a, **_k: {"status": "valid", "problems": []},
    )
    monkeypatch.setattr(
        recovery,
        "validate_full_benchmark_release_acceptance",
        lambda *_a, **_k: {
            "status": "valid",
            "source_commits": [recovery.FROZEN_SOURCE_SHA],
            "episode_count": 2,
        },
    )
    monkeypatch.setattr(recovery, "_assert_frozen_source_repository", lambda *_a: None)
    monkeypatch.setattr(
        recovery,
        "_validator_provenance",
        lambda *_a, **_k: {
            "commit": "d" * 40,
            "file": "robot_sf/benchmark/release_acceptance.py",
            "file_sha256": _sha256(Path(release_acceptance.__file__)),
        },
    )
    monkeypatch.setattr(recovery, "write_campaign_report", lambda *_a, **_k: None)
    monkeypatch.setattr(recovery, "verify_publication_bundle_preflight", lambda *_a, **_k: {})

    def fake_export(run_dir: Path, out_dir: Path, *, bundle_name: str, **_kwargs: object):
        bundle_dir = out_dir / bundle_name
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True)
        _write(bundle_dir / "publication_manifest.json", "{}\n")
        _write(bundle_dir / "checksums.sha256", "0" * 64 + "  payload.txt\n")
        archive = out_dir / f"{bundle_name}.tar.gz"
        archive.write_bytes(b"archive bytes")
        return SimpleNamespace(
            bundle_dir=bundle_dir,
            archive_path=archive,
            file_count=1,
            total_bytes=1,
        )

    monkeypatch.setattr(recovery, "export_publication_bundle", fake_export)
    output_root = tmp_path / "output"
    result = recovery.build_derived_release(
        producer_root=producer,
        acceptance_root=producer,
        source_repository_root=source,
        validator_repository_root=source,
        expected_validator_commit="d" * 40,
        manifest_path=manifest,
        output_root=output_root,
        derived_name="derived",
    )
    final_campaign = output_root / "derived"
    assert result["status"] == "published_to_staging"
    assert final_campaign.is_dir()
    assert (final_campaign / "derived_publication").is_dir()
    final_inventory = (final_campaign / "SHA256SUMS").read_text()
    assert "derived_publication/" in final_inventory
    assert "derived_publication/derived_publication_bundle.tar.gz" in final_inventory
    assert "derived_publication/publication_custody.json" in final_inventory
    assert not list(output_root.glob(".derived.staging-*"))
