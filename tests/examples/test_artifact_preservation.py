"""Focused contracts for the task-owned artifact-preservation example (#8909)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from scripts.tools import chunk_manifest

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "advanced" / "41_artifact_preservation.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location("artifact_preservation_example", EXAMPLE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
example = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(example)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    example.prepare_fixture_tree(example.DEFAULT_FIXTURE, source)
    return source


def _fixture_copy(tmp_path: Path) -> Path:
    payload = json.loads(
        example.DEFAULT_FIXTURE.joinpath("fixture.json").read_text(encoding="utf-8")
    )
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_end_to_end_manifest_and_cleanup_boundary(tmp_path: Path) -> None:
    output = tmp_path / "walkthrough"
    report = example.run_preservation(example.DEFAULT_FIXTURE, out_dir=output)
    manifest = json.loads((output / example.MANIFEST_NAME).read_text(encoding="utf-8"))

    assert report["status"] == "verified"
    assert manifest["schema_version"] == "chunk_manifest.v1"
    assert report["manifest"]["manifest_id"] == manifest["manifest_id"]
    assert report["verification"] == {"source": "ok", "destination": "ok"}
    assert report["cleanup_eligibility"]["before_destination_verification"]["outcome"] != "eligible"
    assert report["cleanup_eligibility"]["after_destination_verification"]["outcome"] == "eligible"
    assert report["boundary"]["upload"] == report["boundary"]["delete"] == "not_performed"
    assert report["boundary"]["evidence"] == "not_evidence"
    assert report["boundary"]["custody"] == "not_asserted"
    assert report["boundary"]["locality"] == "logical_fixture_only"
    assert {
        report["roles"][path]
        for path in ("config/run.yaml", "cache/index.tmp", "credentials/api_token.txt")
    } == {"required", "transient", "forbidden"}
    assert not (output / "destination" / "cache" / "index.tmp").exists()


@pytest.mark.parametrize(
    ("kind", "expected_code", "expected_path"),
    [
        ("unowned", "unowned_root", None),
        ("missing", "missing_required_member", "rows/results.jsonl"),
        ("credential", "credential_like_member", "credentials/api_token.txt"),
    ],
)
def test_fixture_rejections_fail_closed(
    tmp_path: Path, kind: str, expected_code: str, expected_path: str | None
) -> None:
    source = _source(tmp_path)
    if kind == "unowned":
        (source / example.MARKER_NAME).unlink()
    elif kind == "missing":
        (source / "rows" / "results.jsonl").unlink()
    else:
        secret = source / "credentials" / "api_token.txt"
        secret.parent.mkdir()
        secret.write_text("placeholder", encoding="utf-8")

    with pytest.raises(example.PreservationExampleError) as caught:
        example.build_fixture_manifest(source)
    assert caught.value.code == expected_code
    assert caught.value.path == expected_path


def test_symlink_escape_fails_closed(tmp_path: Path) -> None:
    source = _source(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (source / "escape.txt").symlink_to(outside)

    with pytest.raises(chunk_manifest.ChunkManifestError) as caught:
        example.build_fixture_manifest(source)
    assert caught.value.code == "symlink_rejected"


def test_changed_source_fails_manifest_verification(tmp_path: Path) -> None:
    source = _source(tmp_path)
    manifest = example.build_fixture_manifest(source)
    receipt = source / "source" / "source_receipt.json"
    receipt.write_bytes(
        receipt.read_bytes().replace(b"synthetic-source-v1", b"synthetic-source-v2")
    )

    result = chunk_manifest.verify_manifest(source, manifest=manifest)
    assert result["status"] == "failed"
    assert any(
        failure["file"] == "source/source_receipt.json"
        and failure["code"] == "full_digest_mismatch"
        for failure in result["failures"]
    )


def test_duplicate_semantic_id_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture_copy(tmp_path)
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    payload["members"][1]["semantic_id"] = payload["members"][0]["semantic_id"]
    fixture.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(example.PreservationExampleError) as caught:
        example.prepare_fixture_tree(fixture, tmp_path / "source")
    assert caught.value.code == "duplicate_semantic_id"


def test_partial_destination_fails_verification(tmp_path: Path) -> None:
    source, destination = _source(tmp_path), tmp_path / "destination"
    manifest = example.build_fixture_manifest(source)
    destination.mkdir()
    (destination / example.MARKER_NAME).write_bytes((source / example.MARKER_NAME).read_bytes())
    first = manifest["files"][0]["path"]
    target = destination.joinpath(*Path(first).parts)
    target.parent.mkdir(parents=True)
    target.write_bytes(source.joinpath(*Path(first).parts).read_bytes())

    result = chunk_manifest.verify_manifest(destination, manifest=manifest)
    assert result["status"] == "failed"
    assert any(failure["code"] == "missing_member" for failure in result["failures"])
