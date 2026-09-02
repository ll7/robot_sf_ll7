"""Focused tests for the preserved-row release recovery helper."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark import release_acceptance
from robot_sf.benchmark.metrics import snqi as curvature_aware_snqi
from robot_sf.benchmark.release_erratum import ErratumContract
from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    load_baseline_mapping,
    load_weight_mapping,
)
from scripts.tools import revalidate_benchmark_release as recovery


def _sha256(path: Path) -> str:
    """Return a fixture file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str) -> None:
    """Write a UTF-8 fixture file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_erratum_identity_rewrites_publication_but_preserves_execution_input(
    tmp_path: Path,
) -> None:
    """Successor coordinates are self-consistent without relabeling the executed source."""
    source_sha = "5" * 40
    old_tag = f"paper-matrix-v2-h600-s30-2026-09-{source_sha}"
    new_tag = f"{old_tag}-erratum.1"
    campaign = tmp_path / "campaign"
    metadata = tmp_path / "metadata.json"
    _write(metadata, '{"metadata":{"title":"erratum"}}\n')
    old_release = {
        "release_id": old_tag,
        "release_tag": old_tag,
        "provenance": {
            "doi": "10.5281/zenodo.22227035",
            "version_doi": "10.5281/zenodo.22227035",
            "concept_doi": "10.5281/zenodo.22227034",
            "source_sha": source_sha,
            "publication_channel": "direct_zenodo_benchmark_dataset",
        },
    }
    _write(campaign / "release/release_manifest.resolved.json", json.dumps(old_release))
    _write(
        campaign / "release/release_result.json",
        json.dumps(
            {
                "publication_preflight_status": "pass",
                "publication_preflight_violations": ["stale"],
                "release_status": "ok",
            }
        ),
    )
    for relative in ("campaign_manifest.json", "manifest.json", "run_meta.json"):
        _write(relative_path := campaign / relative, json.dumps({"benchmark_release": old_release}))
        assert relative_path.is_file()
    launch = campaign / "launch_packet.json"
    _write(launch, json.dumps({"release_tag": old_tag, "source_sha": source_sha}))
    launch_before = launch.read_bytes()
    contract = ErratumContract(
        correction_id="september-2026-derived-metadata-erratum.1",
        predecessor_version_doi="10.5281/zenodo.22227035",
        predecessor_archive_sha256="e" * 64,
        predecessor_archive_size_bytes=54219004,
        predecessor_github_release_tag=old_tag,
        source_sha=source_sha,
        planner_arms=14,
        scenario_count=48,
        seed_count=30,
        episode_rows=20160,
        builder_sha="a" * 40,
        concept_doi="10.5281/zenodo.22227034",
        successor_version_doi="10.5281/zenodo.22229999",
        successor_github_release_tag=new_tag,
        metadata_path=metadata,
        metadata_sha256=_sha256(metadata),
    )

    resolved = recovery._apply_erratum_publication_identity(campaign, contract=contract)

    assert resolved["release_tag"] == new_tag
    assert resolved["provenance"]["version_doi"] == "10.5281/zenodo.22229999"
    assert resolved["provenance"]["scientific_source_sha"] == source_sha
    result = json.loads((campaign / "release/release_result.json").read_text(encoding="utf-8"))
    assert result["publication_preflight_status"] == "pass"
    assert result["publication_preflight_violations"] == []
    assert result["ranking_claims_admitted"] is False
    assert result["derivation"]["builder_sha"] == "a" * 40
    assert launch.read_bytes() == launch_before
    assert (campaign / recovery.ERRATUM_METADATA_RELATIVE).read_bytes() == metadata.read_bytes()


def test_load_recovery_contract_supports_new_checksum_pinned_campaign(tmp_path: Path) -> None:
    contract_path = tmp_path / "recovery.json"
    payload = {
        "schema_version": "benchmark-derived-release-recovery.v1",
        "source_sha": "5" * 40,
        "producer_sums_sha256": "1" * 64,
        "producer_receipt_sha256": "2" * 64,
        "rejected_result_sha256": "3" * 64,
        "producer_file_count": 110,
        "source_campaign_relative": "output/benchmarks/camera_ready/campaign-v1",
        "episode_rows": 20_160,
        "arms": 14,
        "goal_timeout_boundary_rows": [],
    }
    _write(contract_path, json.dumps(payload) + "\n")

    contract = recovery.load_recovery_contract(contract_path)

    assert contract.source_sha == "5" * 40
    assert contract.producer_file_count == 110
    assert contract.goal_timeout_boundary_rows == frozenset()


def test_load_recovery_contract_rejects_unsafe_campaign_path(tmp_path: Path) -> None:
    contract_path = tmp_path / "recovery.json"
    payload = {
        "schema_version": "benchmark-derived-release-recovery.v1",
        "source_sha": "5" * 40,
        "producer_sums_sha256": "1" * 64,
        "producer_receipt_sha256": "2" * 64,
        "rejected_result_sha256": "3" * 64,
        "producer_file_count": 110,
        "source_campaign_relative": "../campaign-v1",
        "episode_rows": 20_160,
        "arms": 14,
    }
    _write(contract_path, json.dumps(payload) + "\n")

    with pytest.raises(recovery.DerivedReleaseError, match="source_campaign_relative"):
        recovery.load_recovery_contract(contract_path)


def test_preserved_receipt_requires_campaign_specific_refreshed_digest() -> None:
    """A generalized recovery cannot inherit the historical job-14890 refresh digest."""
    contract = recovery.RecoveryContract(
        source_sha="5" * 40,
        producer_sums_sha256="1" * 64,
        producer_receipt_sha256="2" * 64,
        rejected_result_sha256="3" * 64,
        producer_file_count=1,
        source_campaign_relative=Path("output/campaign"),
        episode_rows=1,
        arms=1,
        goal_timeout_boundary_rows=frozenset(),
    )

    with pytest.raises(recovery.DerivedReleaseError, match="refreshed_producer"):
        recovery._expected_current_producer_receipt_sha256(
            contract,
            preserved_receipt_source=Path("preserved.json.gz"),
        )


def test_current_receipt_digest_selection_is_explicit() -> None:
    """Receipt selection uses the base digest normally and the pinned refresh with preservation."""
    contract = recovery.RecoveryContract(
        source_sha="5" * 40,
        producer_sums_sha256="1" * 64,
        producer_receipt_sha256="2" * 64,
        refreshed_producer_receipt_sha256="4" * 64,
        rejected_result_sha256="3" * 64,
        producer_file_count=1,
        source_campaign_relative=Path("output/campaign"),
        episode_rows=1,
        arms=1,
        goal_timeout_boundary_rows=frozenset(),
    )

    assert (
        recovery._expected_current_producer_receipt_sha256(
            contract,
            preserved_receipt_source=None,
        )
        == "2" * 64
    )
    assert (
        recovery._expected_current_producer_receipt_sha256(
            contract,
            preserved_receipt_source=Path("preserved.json.gz"),
        )
        == "4" * 64
    )


def _snqi_metrics(*, curvature_mean: float) -> dict[str, float]:
    """Return a minimal metric payload on the pinned curvature-aware basis."""
    root = Path(__file__).resolve().parents[2]
    weights = load_weight_mapping(root / "configs/benchmarks/snqi_weights_camera_ready_v3.json")
    baseline = load_baseline_mapping(root / "configs/benchmarks/snqi_baseline_camera_ready_v3.json")
    metrics = {
        "success": 1.0,
        "time_to_goal_norm": 1.0,
        "collisions": 0.0,
        "near_misses": 0.0,
        "comfort_exposure": 0.0,
        "force_exceed_events": 0.0,
        "jerk_mean": 0.0,
        "curvature_mean": curvature_mean,
    }
    metrics["snqi"] = curvature_aware_snqi(metrics, weights, baseline_stats=baseline)
    assert math.isfinite(metrics["snqi"])
    return metrics


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
                "root_private": "/root/.cache/robot-sf/worktree/output.json",
                "mac_private": "/Users/example/worktrees/release/output.json",
            }
        ),
    )
    recovery._sanitise_tree_paths(copied, source_root=source_root, producer_root=producer_root)
    value = path.read_text(encoding="utf-8")
    assert "configs/scenarios/matrix.yaml" in value
    assert "runs/arm/episodes.jsonl" in value
    assert "/tmp/" not in value
    assert "/root/" not in value
    assert "/Users/" not in value
    recovery._assert_no_private_absolute_paths(copied)


def test_path_sanitiser_does_not_split_a_source_path_at_stream_boundary(tmp_path: Path) -> None:
    """A long text line cannot corrupt a source path that crosses the old chunk boundary."""
    source_root = tmp_path / "source"
    producer_root = tmp_path / "producer"
    copied = tmp_path / "copied"
    copied.mkdir()
    source_path = source_root / "output/benchmarks/campaign/runs/arm/episodes.jsonl"
    split_offset = 1024 * 1024 - 8192 - 10
    path = copied / "large.json"
    path.write_bytes(b"x" * split_offset + str(source_path).encode() + b"x" * 9000 + b"\n")

    recovery._sanitise_tree_paths(copied, source_root=source_root, producer_root=producer_root)

    value = path.read_text(encoding="utf-8")
    assert str(source_root) not in value
    assert "output/benchmarks/campaign/runs/arm/episodes.jsonl" in value
    assert "<external-path>" not in value


def test_path_sanitiser_rejects_unknown_suffix_and_preserves_known_binary(
    tmp_path: Path,
) -> None:
    """Unknown files fail closed while declared binary payloads are untouched."""
    source_root = tmp_path / "source"
    producer_root = tmp_path / "producer"
    copied = tmp_path / "copied"
    copied.mkdir()
    _write(copied / "opaque.weird", "/root/private\n")
    with pytest.raises(recovery.DerivedReleaseError, match="unsupported publication file type"):
        recovery._sanitise_tree_paths(copied, source_root=source_root, producer_root=producer_root)
    copied.joinpath("opaque.weird").unlink()
    payload = b"/root/private\x00\x01"
    (copied / "payload.parquet").write_bytes(payload)
    recovery._sanitise_tree_paths(copied, source_root=source_root, producer_root=producer_root)
    assert (copied / "payload.parquet").read_bytes() == payload


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "nested/name", ""])
def test_generated_names_are_single_safe_components(name: str) -> None:
    """Derived and publication descriptors cannot escape their output root."""
    with pytest.raises(recovery.DerivedReleaseError):
        recovery._validate_safe_component(name, label="derived_name")


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


def test_publication_inputs_must_match_loaded_manifest_paths_and_hashes(tmp_path: Path) -> None:
    """CITATION, Zenodo, and SNQI inputs cannot silently switch source assets."""
    source = tmp_path / "source"
    source.mkdir()
    paths = {
        "citation": source / "CITATION.cff",
        "metadata": source / "metadata.json",
        "weights": source / "weights.json",
        "baseline": source / "baseline.json",
    }
    for path in paths.values():
        _write(path, path.name + "\n")
    manifest = SimpleNamespace(
        citation_path=paths["citation"],
        metadata_path=paths["metadata"],
        metadata_sha256=_sha256(paths["metadata"]),
        snqi_weights_path=paths["weights"],
        snqi_weights_sha256=_sha256(paths["weights"]),
        snqi_baseline_path=paths["baseline"],
        snqi_baseline_sha256=_sha256(paths["baseline"]),
    )
    resolved = {
        "provenance": {
            "citation_path": "CITATION.cff",
            "metadata_path": "metadata.json",
        },
        "metrics": {
            "snqi_weights_path": "weights.json",
            "snqi_baseline_path": "baseline.json",
        },
    }
    result = recovery._assert_publication_inputs_from_manifest(manifest, resolved, source)
    assert result["citation"]["sha256"] == _sha256(paths["citation"])
    resolved["metrics"]["snqi_weights_path"] = "baseline.json"
    with pytest.raises(recovery.DerivedReleaseError, match="canonical loaded manifest"):
        recovery._assert_publication_inputs_from_manifest(manifest, resolved, source)


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
        assert recovery.camera_run_state_module.get_repository_root() == source


def test_private_path_hygiene_rejects_paths_in_text_and_binary(tmp_path: Path) -> None:
    """Public projections reject private path markers in every admitted format."""
    root = tmp_path / "projection"
    _write(root / "notes.txt", "/var/worktrees/private/job.json\n")
    with pytest.raises(recovery.DerivedReleaseError, match="private absolute path"):
        recovery._assert_no_private_absolute_paths(root)

    binary = root / "payload.pdf"
    binary.write_bytes(b"%PDF\x00/var/worktrees/private/job.json\x00")
    (root / "notes.txt").write_text("portable\n", encoding="utf-8")
    with pytest.raises(recovery.DerivedReleaseError, match="private absolute path"):
        recovery._assert_no_private_absolute_paths(root)


def test_optional_accepted_receipt_is_schema_checked_and_bound(tmp_path: Path) -> None:
    """An accepted receipt must describe the accepted map and retrieved receipt."""
    root, _ = _make_verified_retrieval(tmp_path)
    evidence = recovery._verify_campaign_file_map(
        root,
        expected_sums_sha256=_sha256(root / "SHA256SUMS"),
        expected_result_sha256=_sha256(root / "release/release_result.json"),
        expected_file_count=2,
    )
    retrieved = recovery.verify_producer_artifacts(
        root,
        expected_sums_sha256=_sha256(root / "SHA256SUMS"),
        expected_receipt_sha256=_sha256(root / "artifact-verification-receipt.json"),
        expected_rejected_result_sha256=_sha256(root / "release/release_result.json"),
        expected_file_count=2,
    )
    recovery._assert_accepted_receipt_relation(evidence, retrieved)
    _write(root / "artifact-verification-receipt.json", "tampered\n")
    with pytest.raises(recovery.DerivedReleaseError, match="invalid JSON input"):
        recovery._verify_campaign_file_map(
            root,
            expected_sums_sha256=_sha256(root / "SHA256SUMS"),
            expected_result_sha256=_sha256(root / "release/release_result.json"),
            expected_file_count=2,
        )


def test_acceptance_campaign_is_checksums_bound_subset_without_local_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The canonical execution tree need not duplicate collection-only custody files."""
    producer, _ = _make_verified_retrieval(tmp_path)
    producer_evidence = recovery.verify_producer_artifacts(
        producer,
        expected_sums_sha256=_sha256(producer / "SHA256SUMS"),
        expected_receipt_sha256=_sha256(producer / "artifact-verification-receipt.json"),
        expected_rejected_result_sha256=_sha256(producer / "release/release_result.json"),
        expected_file_count=2,
    )
    acceptance = tmp_path / "acceptance"
    _write(acceptance / "payload.txt", (producer / "payload.txt").read_text())
    _write(
        acceptance / "release/release_result.json",
        (producer / "release/release_result.json").read_text(),
    )
    monkeypatch.setattr(
        recovery,
        "EXPECTED_REJECTED_RESULT_SHA256",
        _sha256(producer / "release/release_result.json"),
    )

    evidence = recovery._verify_acceptance_campaign_subset(
        acceptance,
        producer_evidence=producer_evidence,
    )

    assert evidence["status"] == "verified"
    assert evidence["file_count"] == 2
    assert evidence["producer_extra_file_count"] > 0
    assert not (acceptance / "SHA256SUMS").exists()

    _write(acceptance / "payload.txt", "tampered\n")
    with pytest.raises(recovery.DerivedReleaseError, match="not bound"):
        recovery._verify_acceptance_campaign_subset(
            acceptance,
            producer_evidence=producer_evidence,
        )


def test_acceptance_campaign_rejects_unlisted_extra_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every validator-visible acceptance byte must exist in the producer inventory."""
    producer, _ = _make_verified_retrieval(tmp_path)
    producer_evidence = recovery.verify_producer_artifacts(
        producer,
        expected_sums_sha256=_sha256(producer / "SHA256SUMS"),
        expected_receipt_sha256=_sha256(producer / "artifact-verification-receipt.json"),
        expected_rejected_result_sha256=_sha256(producer / "release/release_result.json"),
        expected_file_count=2,
    )
    acceptance = tmp_path / "acceptance"
    _write(
        acceptance / "release/release_result.json",
        (producer / "release/release_result.json").read_text(),
    )
    _write(acceptance / "unlisted.json", "{}\n")
    monkeypatch.setattr(
        recovery,
        "EXPECTED_REJECTED_RESULT_SHA256",
        _sha256(acceptance / "release/release_result.json"),
    )
    with pytest.raises(recovery.DerivedReleaseError, match="not bound"):
        recovery._verify_acceptance_campaign_subset(
            acceptance,
            producer_evidence=producer_evidence,
        )


def test_validator_checkout_must_be_distinct_from_source_and_helper(tmp_path: Path) -> None:
    """A reviewed validator cannot be the source or helper checkout itself."""
    source = tmp_path / "source"
    validator = tmp_path / "validator"
    source.mkdir()
    validator.mkdir()
    recovery._assert_distinct_validator_checkout(validator, source)
    with pytest.raises(recovery.DerivedReleaseError, match="distinct"):
        recovery._assert_distinct_validator_checkout(source, source)
    helper = Path(recovery.__file__).resolve().parents[2]
    with pytest.raises(recovery.DerivedReleaseError, match="distinct"):
        recovery._assert_distinct_validator_checkout(helper, source)


def test_seed_set_path_uses_manifest_relative_loader_rule(tmp_path: Path) -> None:
    """Relative seed assets resolve beside the manifest, not at repository root."""
    source = tmp_path / "source"
    source.mkdir()
    manifest_path = source / "configs" / "release.yaml"
    seed_path = manifest_path.parent / "assets" / "seeds.json"
    _write(manifest_path, "manifest\n")
    _write(seed_path, "{}\n")
    manifest = SimpleNamespace(
        path=manifest_path,
        seed_policy={"seed_sets_path": "assets/seeds.json"},
    )
    recovery._assert_manifest_paths_from_source(manifest, source)
    _write(source / "assets" / "seeds.json", "wrong location\n")
    # The manifest-relative candidate remains authoritative even when a same-
    # named repository-root candidate exists.
    recovery._assert_manifest_paths_from_source(manifest, source)


def test_cross_root_file_map_rejects_size_or_digest_drift() -> None:
    """Accepted and published campaign maps must match path, size, and SHA."""
    accepted = {"file_map": {"row.json": {"bytes": 10, "sha256": "a" * 64}}}
    retrieved = {"file_map": {"row.json": {"bytes": 11, "sha256": "a" * 64}}}
    with pytest.raises(recovery.DerivedReleaseError, match="file map mismatch"):
        recovery._assert_equal_file_maps(accepted, retrieved)


def test_root_checksum_is_temporarily_excluded_from_nested_bundle(tmp_path: Path) -> None:
    """The final root inventory cannot be copied into a checksum-dependent archive."""
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    sums = campaign / "SHA256SUMS"
    sums.write_text("a" * 64 + "  payload.txt\n", encoding="utf-8")
    with recovery._exclude_root_checksum_from_bundle(campaign):
        assert not sums.exists()
        assert (campaign / ".SHA256SUMS.not-bundled").is_file()
    assert sums.read_text(encoding="utf-8").startswith("a" * 64)


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


def test_exact_validator_subprocess_uses_supplied_checkout() -> None:
    """The acceptance result is produced by the explicitly supplied checkout."""
    root = Path(__file__).parents[2].resolve()
    candidates = [root.parent / "release-fix-adaptive-algo-b1d5-20260825"]
    candidates = [
        path
        for path in candidates
        if path.resolve() != root
        and (path / "robot_sf/benchmark/release_acceptance.py").is_file()
        and (
            path
            / "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml"
        ).is_file()
    ]
    if not candidates:
        pytest.skip("requires a distinct reviewed validator checkout")
    validator = candidates[0]
    source_candidates = [root.parent / "release-exec-s30-h600-b1d5ab6de708-20260825"]
    source_candidates = [
        path
        for path in source_candidates
        if path.resolve() != root
        and (
            path
            / "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml"
        ).is_file()
    ]
    if not source_candidates:
        pytest.skip("requires a distinct frozen source checkout")
    source = source_candidates[0]
    manifest = source / (
        "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml"
    )
    result = recovery._run_exact_validator(
        validator_root=validator,
        source_root=source,
        acceptance_root=validator,
        manifest_path=manifest,
    )
    assert result["status"] == "not_applicable"


def test_exact_validator_runs_from_frozen_source_without_import_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relative source assets resolve from the frozen checkout, not validator tooling."""
    validator = tmp_path / "validator"
    source = tmp_path / "source"
    acceptance = tmp_path / "acceptance"
    manifest = source / "manifest.yaml"
    for path in (validator, source, acceptance):
        path.mkdir()
    _write(manifest, "manifest\n")
    _write(source / "maps/registry.yaml", "maps: {}\n")
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["command"] = command
        observed.update(kwargs)
        return SimpleNamespace(returncode=0, stdout='{"status":"valid"}\n', stderr="")

    monkeypatch.setattr(recovery.subprocess, "run", fake_run)
    result = recovery._run_exact_validator(
        validator_root=validator,
        source_root=source,
        acceptance_root=acceptance,
        manifest_path=manifest,
    )

    assert result["status"] == "valid"
    assert observed["cwd"] == source
    assert observed["env"]["PYTHONPATH"] == str(validator)  # type: ignore[index]
    assert observed["env"]["ROBOT_SF_MAP_REGISTRY"] == str(  # type: ignore[index]
        source / "maps/registry.yaml"
    )
    assert "sys.path.insert(0, str(validator_root))" in observed["command"][2]  # type: ignore[index]


def test_publication_projection_annotates_only_pinned_goal_timeout_boundary(
    tmp_path: Path,
) -> None:
    """A known terminal-boundary ambiguity gains a note without invented timing or metric drift."""
    campaign = tmp_path / "campaign"
    arm = "guarded_ppo__differential_drive"
    episode_id = "francis2023_parallel_traffic--132--2bf83ad03db6559e"
    episodes = campaign / "runs" / arm / "episodes.jsonl"
    row = {
        "episode_id": episode_id,
        "status": "success",
        "termination_reason": "success",
        "steps": 400,
        "metrics": {"success": 1.0, "time_to_goal": 39.9},
        "outcome": {"route_complete": True, "timeout_event": True},
        "event_ledger": {
            "software_commit": recovery.FROZEN_SOURCE_SHA,
            "exact_events": {"goal_reached": True, "timeout": True},
        },
    }
    _write(episodes, json.dumps(row, sort_keys=True) + "\n")
    original_digest = _sha256(episodes)
    sidecar = episodes.with_name("episodes.jsonl.provenance.json")
    _write(
        sidecar,
        json.dumps(
            {
                "raw_artifacts": [
                    {
                        "kind": "episodes_jsonl",
                        "path": f"runs/{arm}/episodes.jsonl",
                        "sha256": original_digest,
                    }
                ],
                "derived_artifacts": [],
                "rows": [{"raw_artifact": f"runs/{arm}/episodes.jsonl"}],
            }
        )
        + "\n",
    )
    _write(campaign / "run_meta.json", json.dumps({"repo": {"commit": recovery.FROZEN_SOURCE_SHA}}))

    evidence = recovery._annotate_publication_goal_timeout_boundaries(
        campaign,
        expected_rows={(arm, episode_id)},
    )
    sidecar_evidence = recovery._rebind_publication_sidecars(
        campaign,
        source_file_map={
            f"runs/{arm}/episodes.jsonl": {
                "sha256": original_digest,
                "bytes": len(json.dumps(row, sort_keys=True) + "\n"),
            },
            f"runs/{arm}/episodes.jsonl.provenance.json": {
                "sha256": _sha256(sidecar),
                "bytes": sidecar.stat().st_size,
            },
        },
        boundary_reconciliation=evidence,
        expected_arm_count=1,
        expected_row_count=1,
    )

    derived = json.loads(episodes.read_text(encoding="utf-8"))
    assert derived["goal_timeout_boundary_note"] == recovery.GOAL_TIMEOUT_BOUNDARY_NOTE
    assert "reached_goal_step" not in derived
    assert derived["status"] == row["status"]
    assert derived["termination_reason"] == row["termination_reason"]
    assert derived["outcome"] == row["outcome"]
    assert derived["metrics"] == row["metrics"]
    refreshed_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
    assert refreshed_sidecar["raw_artifacts"][0]["sha256"] == _sha256(episodes)
    projection = refreshed_sidecar["derived_artifacts"][-1]
    assert projection["producer_sha256"] == original_digest
    assert projection["goal_timeout_boundary_annotation"]["timing_evidence_fabricated"] is False
    assert evidence["annotated_row_count"] == 1
    assert sidecar_evidence["row_count"] == 1
    assert evidence["timing_evidence_fabricated"] is False
    run_meta = json.loads((campaign / "run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["goal_timeout_boundary"]["unresolved_rows"] == 0


def test_publication_projection_rejects_unexpected_goal_timeout_row_before_writing(
    tmp_path: Path,
) -> None:
    """An unreviewed ambiguous identity cannot be silently annotated by the recovery helper."""
    campaign = tmp_path / "campaign"
    episodes = campaign / "runs" / "guarded_ppo__differential_drive" / "episodes.jsonl"
    row = {
        "episode_id": "unexpected-row",
        "status": "success",
        "termination_reason": "success",
        "metrics": {"success": 1.0},
        "outcome": {"route_complete": True, "timeout_event": True},
        "event_ledger": {"exact_events": {"goal_reached": True, "timeout": True}},
    }
    original = json.dumps(row, sort_keys=True) + "\n"
    _write(episodes, original)

    with pytest.raises(recovery.DerivedReleaseError, match="reviewed boundary-row set"):
        recovery._annotate_publication_goal_timeout_boundaries(
            campaign,
            expected_rows={
                (
                    "guarded_ppo__differential_drive",
                    "francis2023_parallel_traffic--132--2bf83ad03db6559e",
                )
            },
        )
    assert episodes.read_text(encoding="utf-8") == original


def test_publication_projection_rejects_inconsistent_goal_timeout_semantics(
    tmp_path: Path,
) -> None:
    """The reviewed identity cannot receive a note when its scientific outcome is inconsistent."""
    campaign = tmp_path / "campaign"
    arm = "guarded_ppo__differential_drive"
    episode_id = "francis2023_parallel_traffic--132--2bf83ad03db6559e"
    episodes = campaign / "runs" / arm / "episodes.jsonl"
    row = {
        "episode_id": episode_id,
        "status": "failure",
        "termination_reason": "success",
        "metrics": {"success": 1.0},
        "outcome": {"route_complete": True, "timeout_event": True},
        "event_ledger": {"exact_events": {"goal_reached": True, "timeout": True}},
    }
    original = json.dumps(row, sort_keys=True) + "\n"
    _write(episodes, original)

    with pytest.raises(recovery.DerivedReleaseError, match="successful terminal semantics"):
        recovery._annotate_publication_goal_timeout_boundaries(
            campaign,
            expected_rows={(arm, episode_id)},
        )
    assert episodes.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("malformed_value", ["false", "true", 1, [], {}])
def test_publication_projection_requires_literal_boolean_boundary_events(
    malformed_value: object,
) -> None:
    """Truth-like malformed event values cannot authorize a publication annotation."""
    record = {
        "event_ledger": {
            "exact_events": {"goal_reached": malformed_value, "timeout": True},
        }
    }

    assert recovery._is_unresolved_goal_timeout_boundary(record) is False


def test_publication_projection_reconciles_snqi_ordering_as_advisory(
    tmp_path: Path,
) -> None:
    """Stored SNQI fields define diagnostics ordering but never gain ranking authority."""
    campaign = tmp_path / "campaign"
    root = Path(__file__).resolve().parents[2]
    weights = root / "configs/benchmarks/snqi_weights_camera_ready_v3.json"
    baseline = root / "configs/benchmarks/snqi_baseline_camera_ready_v3.json"
    rows = {
        "goal__differential_drive": _snqi_metrics(curvature_mean=0.05),
        "orca__differential_drive": _snqi_metrics(curvature_mean=0.5),
    }
    for arm, metrics in rows.items():
        _write(
            campaign / "runs" / arm / "episodes.jsonl",
            json.dumps({"episode_id": arm, "metrics": metrics}) + "\n",
        )
    diagnostics = {
        "contract_enabled": True,
        "contract_enforcement": "warn",
        "contract_status": "fail",
        "weights_sha256": _sha256(weights),
        "baseline_sha256": _sha256(baseline),
        "planner_ordering": [
            {
                "planner_key": "orca",
                "kinematics": "differential_drive",
                "episode_count": 1,
                "mean_snqi": rows["orca__differential_drive"]["snqi"],
                "rank": 1,
            },
            {
                "planner_key": "goal",
                "kinematics": "differential_drive",
                "episode_count": 1,
                "mean_snqi": rows["goal__differential_drive"]["snqi"],
                "rank": 2,
            },
        ],
        "positioning": {"planner_ordering_informative": True, "caveats": []},
    }
    _write(campaign / "reports" / "snqi_diagnostics.json", json.dumps(diagnostics) + "\n")

    evidence = recovery._reconcile_publication_snqi_diagnostics(
        campaign,
        expected_row_count=2,
        expected_arm_count=2,
    )

    reconciled = json.loads(
        (campaign / "reports" / "snqi_diagnostics.json").read_text(encoding="utf-8")
    )
    assert [row["planner_key"] for row in reconciled["planner_ordering"]] == ["goal", "orca"]
    assert reconciled["planner_ordering_basis"] == "stored_metrics.snqi"
    assert reconciled["contract_status"] == "fail"
    assert reconciled["contract_enforcement"] == "warn"
    assert reconciled["release_claim_boundary"]["ranking_authority"] is False
    assert reconciled["release_claim_boundary"]["ranking_claims_admitted"] is False
    assert reconciled["positioning"]["planner_ordering_informative"] is False
    assert reconciled["positioning"]["recommendation"] == "retain_as_advisory_only_not_for_ranking"
    assert evidence["verified_episode_rows"] == 2
    assert evidence["post_reconciliation_violation_count"] == 0
    markdown = (campaign / "reports" / "snqi_diagnostics.md").read_text(encoding="utf-8")
    assert recovery.SNQI_ADVISORY_BOUNDARY in markdown


def test_publication_projection_rejects_drifted_stored_snqi(tmp_path: Path) -> None:
    """Ordering repair cannot hide a per-episode scalarization mismatch."""
    campaign = tmp_path / "campaign"
    root = Path(__file__).resolve().parents[2]
    weights = root / "configs/benchmarks/snqi_weights_camera_ready_v3.json"
    baseline = root / "configs/benchmarks/snqi_baseline_camera_ready_v3.json"
    metrics = _snqi_metrics(curvature_mean=0.05)
    metrics["snqi"] += 0.25
    _write(
        campaign / "runs" / "goal__differential_drive" / "episodes.jsonl",
        json.dumps({"episode_id": "drifted", "metrics": metrics}) + "\n",
    )
    _write(
        campaign / "reports" / "snqi_diagnostics.json",
        json.dumps(
            {
                "contract_enabled": True,
                "contract_enforcement": "warn",
                "contract_status": "fail",
                "weights_sha256": _sha256(weights),
                "baseline_sha256": _sha256(baseline),
                "planner_ordering": [
                    {
                        "planner_key": "goal",
                        "kinematics": "differential_drive",
                        "episode_count": 1,
                        "mean_snqi": metrics["snqi"],
                        "rank": 1,
                    }
                ],
            }
        )
        + "\n",
    )

    with pytest.raises(recovery.DerivedReleaseError, match="drift beyond"):
        recovery._reconcile_publication_snqi_diagnostics(
            campaign,
            expected_row_count=1,
            expected_arm_count=1,
        )


def test_build_derived_release_cleans_partial_stage_on_bundle_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed export leaves neither a final target nor a hidden partial stage."""
    producer, _ = _make_verified_retrieval(tmp_path)
    source = tmp_path / "source"
    validator = tmp_path / "validator"
    source.mkdir()
    validator.mkdir()
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
    monkeypatch.setattr(
        recovery, "load_release_campaign_config", lambda *_a, **_k: SimpleNamespace()
    )
    monkeypatch.setattr(
        recovery,
        "_run_exact_validator",
        lambda **_kwargs: {
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
        "_verify_acceptance_campaign_subset",
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
    monkeypatch.setattr(
        recovery,
        "_assert_publication_inputs_from_manifest",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        recovery,
        "_annotate_publication_goal_timeout_boundaries",
        lambda *_args, **_kwargs: {"annotated_row_count": 1},
    )
    monkeypatch.setattr(
        recovery,
        "_rebind_publication_sidecars",
        lambda *_args, **_kwargs: {"arm_count": 14, "row_count": 20_160},
    )
    monkeypatch.setattr(
        recovery,
        "_reconcile_publication_snqi_diagnostics",
        lambda *_args, **_kwargs: {"verified_episode_rows": 20_160},
    )

    def fail_export(*_args, **_kwargs):
        raise recovery.PublicationPreflightError("test export failure")

    monkeypatch.setattr(recovery, "export_publication_bundle", fail_export)
    output_root = tmp_path / "output"
    with pytest.raises(recovery.PublicationPreflightError, match="test export failure"):
        recovery.build_derived_release(
            producer_root=producer,
            acceptance_root=producer,
            source_repository_root=source,
            validator_repository_root=validator,
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
    validator = tmp_path / "validator"
    source.mkdir()
    validator.mkdir()
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
        "_verify_acceptance_campaign_subset",
        lambda *_a, **_k: {"status": "verified", "file_map": fixture_file_map},
    )
    monkeypatch.setattr(
        recovery,
        "load_release_manifest",
        lambda _path: SimpleNamespace(canonical_campaign_config_path=config_path),
    )
    monkeypatch.setattr(
        recovery, "load_release_campaign_config", lambda *_a, **_k: SimpleNamespace()
    )
    monkeypatch.setattr(
        recovery,
        "validate_release_manifest",
        lambda *_a, **_k: {"status": "valid", "problems": []},
    )
    monkeypatch.setattr(
        recovery,
        "_run_exact_validator",
        lambda **_k: {
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
    monkeypatch.setattr(
        recovery,
        "_assert_publication_inputs_from_manifest",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        recovery,
        "_annotate_publication_goal_timeout_boundaries",
        lambda *_a, **_k: {"annotated_row_count": 1},
    )
    monkeypatch.setattr(
        recovery,
        "_rebind_publication_sidecars",
        lambda *_a, **_k: {"arm_count": 14, "row_count": 20_160},
    )
    monkeypatch.setattr(
        recovery,
        "_reconcile_publication_snqi_diagnostics",
        lambda *_a, **_k: {"verified_episode_rows": 20_160},
    )
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
        validator_repository_root=validator,
        expected_validator_commit="d" * 40,
        manifest_path=manifest,
        output_root=output_root,
        derived_name="derived",
    )
    final_campaign = output_root / "derived"
    assert result["status"] == "published_to_staging"
    assert final_campaign.is_dir()
    accepted_result = json.loads(
        (final_campaign / "release" / "release_result.json").read_text(encoding="utf-8")
    )
    assert accepted_result["publication_preflight_status"] == "pass"
    assert accepted_result["publication_preflight_violations"] == []
    assert (final_campaign / "derived_publication").is_dir()
    final_inventory = (final_campaign / "SHA256SUMS").read_text()
    assert "derived_publication/" in final_inventory
    assert "derived_publication/derived_publication_bundle.tar.gz" in final_inventory
    assert "derived_publication/publication_custody.json" in final_inventory
    assert not list(output_root.glob(".derived.staging-*"))
