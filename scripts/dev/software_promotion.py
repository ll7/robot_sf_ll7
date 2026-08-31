#!/usr/bin/env python3
"""Verify and receipt-bind an immutable Robot SF package promotion.

This helper deliberately does not build or upload packages.  It verifies the
candidate bundle produced by ``software_candidate_manifest.py``, checks the
GitHub artifact identity supplied to the protected workflow, and creates the
small, credential-free receipts that let a later job prove it is promoting the
same bytes.  The workflow owns the actual trusted-publisher exchange.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "robot_sf.software_promotion.receipt.v1"
COLD_INSTALL_SCHEMA_VERSION = "robot_sf.software_promotion.cold_install.v1"
REPOSITORY = "ll7/robot_sf_ll7"
INDEX_URLS = {
    "testpypi": "https://test.pypi.org/legacy/",
    "pypi": "https://upload.pypi.org/legacy/",
}
SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
ARTIFACT_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
ARTIFACT_ID_PATTERN = re.compile(r"[1-9][0-9]*\Z")
RUN_ID_PATTERN = re.compile(r"[1-9][0-9]*\Z")
VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+!_-]*\Z")
ARTIFACT_NAME_PATTERN = re.compile(
    r"robot-sf-software-candidate-[0-9a-f]{40}-[1-9][0-9]*-[1-9][0-9]*\Z"
)
RECEIPT_ARTIFACT_NAME_PATTERN = re.compile(
    r"robot-sf-(?:testpypi|pypi)-(?:receipt|cold-install)-[1-9][0-9]*-[1-9][0-9]*\Z"
)
CANDIDATE_MEMBER_KINDS = ("wheel", "sdist", "sbom", "provenance")
PUBLISHED_MEMBER_KINDS = ("wheel", "sdist")
MANIFEST_NAME = "candidate-manifest.json"
PROVENANCE_NAME = "candidate-provenance.json"
VALIDATION_ROSTER = (
    ("version-alignment", "python scripts/dev/check_version_alignment.py"),
    ("metadata", "twine check --strict $DIST_DIR/*.whl $DIST_DIR/*.tar.gz"),
    ("archive-license", "python scripts/tools/check_distribution_licenses.py $DIST_DIR"),
    ("wheel-install", "bash scripts/validation/wheel_install_smoke.sh $DIST_DIR/robot_sf-*.whl"),
)


class PromotionError(ValueError):
    """Raised when promotion identity or receipt validation fails closed."""


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PromotionError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise PromotionError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=_reject_nonfinite,
        )
    except PromotionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionError(f"{label} is not valid UTF-8 JSON: {path}: {exc}") from exc


def _json_bytes(payload: Any) -> bytes:
    try:
        return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise PromotionError("promotion receipt cannot be serialised as finite JSON") from exc


def _write_new_json(path: Path, payload: Any, *, label: str) -> None:
    """Create one receipt without allowing a concurrent writer to replace it."""

    payload_bytes = _json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload_bytes)
    except FileExistsError as exc:
        raise PromotionError(f"refusing to overwrite existing {label}: {path}") from exc
    except OSError as exc:
        raise PromotionError(f"cannot write {label}: {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PromotionError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_real_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise PromotionError(f"{label} is not a regular file: {path}")


def _require_real_dir(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise PromotionError(f"{label} is not a regular directory: {path}")


def _positive_identity(value: str, *, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise PromotionError(f"{label} must be a positive decimal identity")
    return value


def _source_sha(value: str) -> str:
    if not isinstance(value, str) or not SHA_PATTERN.fullmatch(value):
        raise PromotionError("source SHA must be one exact lowercase 40-hex identity")
    return value


def _version(value: str) -> str:
    if not isinstance(value, str) or not VERSION_PATTERN.fullmatch(value):
        raise PromotionError("package version is unsafe or ambiguous")
    return value


def _artifact_digest(value: str) -> str:
    if not isinstance(value, str) or not ARTIFACT_DIGEST_PATTERN.fullmatch(value):
        raise PromotionError("artifact digest must be sha256:<64 lowercase hex characters>")
    return value


def _artifact_name(value: str, *, kind: str) -> str:
    pattern = ARTIFACT_NAME_PATTERN if kind == "candidate" else RECEIPT_ARTIFACT_NAME_PATTERN
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise PromotionError(f"{kind} artifact name is not a recognized immutable identity")
    return value


def _member_record(value: Any, *, expected_kind: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"filename", "kind", "sha256", "size"}:
        raise PromotionError("candidate member record is malformed")
    filename = value["filename"]
    if (
        not isinstance(filename, str)
        or not filename
        or filename != Path(filename).name
        or filename in {".", "..", MANIFEST_NAME}
    ):
        raise PromotionError(f"candidate member filename is unsafe: {filename!r}")
    kind = value["kind"]
    if expected_kind is not None and kind != expected_kind:
        raise PromotionError(f"candidate member kind drift: expected {expected_kind!r}")
    if kind not in CANDIDATE_MEMBER_KINDS:
        raise PromotionError(f"candidate member kind is unknown: {kind!r}")
    digest = value["sha256"]
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise PromotionError(f"candidate member {filename} has an invalid SHA-256")
    size = value["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise PromotionError(f"candidate member {filename} has an invalid size")
    return value


def _validate_validation_roster(validation: Any) -> None:
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise PromotionError("candidate validation envelope is not passed")
    checks = validation.get("checks")
    if not isinstance(checks, list) or any(not isinstance(item, dict) for item in checks):
        raise PromotionError("candidate validation roster is malformed")
    expected_checks = [
        {"command": command, "id": identifier, "status": "passed"}
        for identifier, command in VALIDATION_ROSTER
    ]
    if checks != expected_checks:
        raise PromotionError("candidate validation roster is incomplete or reordered")


def _validate_manifest_header(
    manifest: Any,
    *,
    expected_source_sha: str,
    expected_workflow_run_id: str,
    expected_version: str,
) -> tuple[str, str, dict[str, str]]:
    if not isinstance(manifest, dict):
        raise PromotionError("candidate manifest must be a JSON object")
    expected_keys = {
        "schema_version",
        "repository",
        "source_sha",
        "workflow",
        "package",
        "validation",
        "members",
    }
    if set(manifest) != expected_keys:
        raise PromotionError("candidate manifest has missing or unclassified fields")
    if manifest["schema_version"] != "robot_sf.software_candidate.v1":
        raise PromotionError("candidate manifest schema version is unsupported")
    if manifest["repository"] != REPOSITORY:
        raise PromotionError("candidate repository is not the canonical Robot SF repository")
    source_sha = _source_sha(expected_source_sha)
    if manifest["source_sha"] != source_sha:
        raise PromotionError("candidate source SHA does not match the dispatch identity")

    workflow = manifest["workflow"]
    if not isinstance(workflow, dict) or set(workflow) != {"run_id", "run_attempt"}:
        raise PromotionError("candidate workflow identity is malformed")
    run_id = _positive_identity(
        expected_workflow_run_id,
        label="candidate workflow run ID",
        pattern=RUN_ID_PATTERN,
    )
    if workflow["run_id"] != run_id:
        raise PromotionError("candidate workflow run ID does not match the dispatch identity")
    if (
        not isinstance(workflow["run_attempt"], int)
        or isinstance(workflow["run_attempt"], bool)
        or workflow["run_attempt"] < 1
    ):
        raise PromotionError("candidate workflow run attempt is invalid")

    package = manifest["package"]
    version = _version(expected_version)
    if package != {"name": "robot_sf", "version": version}:
        raise PromotionError("candidate package identity does not match the dispatch identity")
    _validate_validation_roster(manifest["validation"])
    return source_sha, run_id, package


def _validate_candidate_members(
    bundle_dir: Path,
    entries: list[Path],
    members: Any,
) -> list[dict[str, Any]]:
    if not isinstance(members, list) or len(members) != len(CANDIDATE_MEMBER_KINDS):
        raise PromotionError("candidate must bind exactly four members")
    validated_members = [
        _member_record(member, expected_kind=kind)
        for member, kind in zip(members, CANDIDATE_MEMBER_KINDS, strict=True)
    ]
    filenames = [member["filename"] for member in validated_members]
    if len(filenames) != len(set(filenames)):
        raise PromotionError("candidate manifest contains duplicate member filenames")
    expected_names = {MANIFEST_NAME, *(member["filename"] for member in validated_members)}
    if {path.name for path in entries} != expected_names:
        raise PromotionError("candidate bundle membership differs from its manifest")
    for member in validated_members:
        path = bundle_dir / member["filename"]
        if path.stat().st_size != member["size"] or _sha256(path) != member["sha256"]:
            raise PromotionError(f"candidate member bytes drifted: {member['filename']}")
    return validated_members


def _validate_candidate_provenance(
    bundle_dir: Path,
    member: dict[str, Any],
    *,
    source_sha: str,
    package: dict[str, str],
    members: list[dict[str, Any]],
) -> None:
    provenance = _load_json(bundle_dir / member["filename"], label="candidate provenance")
    if not isinstance(provenance, dict):
        raise PromotionError("candidate provenance must be a JSON object")
    if provenance.get("schema_version") != "robot_sf.software_candidate.provenance.v1":
        raise PromotionError("candidate provenance schema version is unsupported")
    build = provenance.get("build")
    if provenance.get("repository") != REPOSITORY or provenance.get("source_sha") != source_sha:
        raise PromotionError("candidate provenance is not bound to the candidate source")
    if not isinstance(build, dict):
        raise PromotionError("candidate provenance build binding is malformed")
    if provenance.get("package") != package or build.get("count") != 1:
        raise PromotionError("candidate provenance build/package binding is invalid")
    if build.get("source_role") != "disposable-exact-commit":
        raise PromotionError("candidate provenance does not identify the exact build source")
    if provenance.get("subjects") != members[:2]:
        raise PromotionError("candidate provenance subjects do not match candidate members")


def _load_candidate(
    bundle_dir: Path,
    *,
    expected_source_sha: str,
    expected_workflow_run_id: str,
    expected_version: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a candidate and return ``(manifest, identity)`` after byte checks."""

    _require_real_dir(bundle_dir, label="candidate bundle")
    entries = sorted(bundle_dir.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise PromotionError("candidate bundle contains a non-regular member")
    manifest_path = bundle_dir / MANIFEST_NAME
    _require_real_file(manifest_path, label="candidate manifest")
    manifest = _load_json(manifest_path, label="candidate manifest")
    source_sha, run_id, package = _validate_manifest_header(
        manifest,
        expected_source_sha=expected_source_sha,
        expected_workflow_run_id=expected_workflow_run_id,
        expected_version=expected_version,
    )

    validated_members = _validate_candidate_members(bundle_dir, entries, manifest["members"])
    provenance_member = validated_members[-1]
    _validate_candidate_provenance(
        bundle_dir,
        provenance_member,
        source_sha=source_sha,
        package=package,
        members=validated_members,
    )

    identity = {
        "source_sha": source_sha,
        "workflow_run_id": run_id,
        "package": package,
        "members": validated_members,
        "manifest_sha256": _sha256(manifest_path),
        "provenance_sha256": provenance_member["sha256"],
        "sbom_sha256": validated_members[2]["sha256"],
    }
    return manifest, identity


def _validate_artifact_source(
    workflow_run: dict[str, Any],
    *,
    kind: str,
    expected_source_sha: str | None,
) -> None:
    if kind == "candidate":
        if workflow_run.get("head_sha") != _source_sha(expected_source_sha or ""):
            raise PromotionError(
                "GitHub candidate artifact source SHA drifted from the dispatch identity"
            )
    elif expected_source_sha is not None:
        raise PromotionError("source SHA is only valid for candidate artifact metadata")


def _artifact_metadata(
    metadata_path: Path,
    *,
    expected_id: str,
    expected_name: str,
    expected_digest: str,
    expected_run_id: str,
    kind: str,
    expected_source_sha: str | None = None,
) -> None:
    metadata = _load_json(metadata_path, label="GitHub artifact metadata")
    if not isinstance(metadata, dict):
        raise PromotionError("GitHub artifact metadata must be an object")
    artifact_id = metadata.get("id")
    if isinstance(artifact_id, bool) or not isinstance(artifact_id, (int, str)):
        raise PromotionError("GitHub artifact metadata has no valid artifact ID")
    if str(artifact_id) != _positive_identity(
        expected_id, label="artifact ID", pattern=ARTIFACT_ID_PATTERN
    ):
        raise PromotionError("GitHub artifact ID drifted from the dispatch identity")
    actual_name = metadata.get("name")
    if actual_name != _artifact_name(expected_name, kind=kind):
        raise PromotionError("GitHub artifact name drifted from the dispatch identity")
    if metadata.get("expired") is not False:
        raise PromotionError("GitHub artifact is expired or has no explicit unexpired status")
    digest = metadata.get("digest")
    if digest != _artifact_digest(expected_digest):
        raise PromotionError("GitHub artifact archive digest drifted from the dispatch identity")
    workflow_run = metadata.get("workflow_run")
    if not isinstance(workflow_run, dict):
        raise PromotionError("GitHub artifact has no workflow-run binding")
    if str(workflow_run.get("id")) != _positive_identity(
        expected_run_id,
        label="artifact workflow run ID",
        pattern=RUN_ID_PATTERN,
    ):
        raise PromotionError("GitHub artifact workflow run drifted from the dispatch identity")
    _validate_artifact_source(
        workflow_run,
        kind=kind,
        expected_source_sha=expected_source_sha,
    )
    archive_url = metadata.get("archive_download_url")
    expected_archive_url = (
        f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{expected_id}/zip"
    )
    if archive_url != expected_archive_url:
        raise PromotionError("GitHub artifact archive download URL is not canonical")


def _published_members(identity: dict[str, Any]) -> list[dict[str, Any]]:
    members = identity["members"]
    return [member for member in members if member["kind"] in PUBLISHED_MEMBER_KINDS]


def _candidate_identity_from_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    candidate = receipt.get("candidate")
    if not isinstance(candidate, dict):
        raise PromotionError("promotion receipt has no candidate identity")
    required = {
        "artifact_digest",
        "artifact_id",
        "artifact_name",
        "manifest_sha256",
        "members",
        "package",
        "provenance_sha256",
        "sbom_sha256",
        "source_sha",
        "workflow_run_id",
    }
    if set(candidate) != required:
        raise PromotionError("promotion receipt candidate identity is incomplete")
    return candidate


def _assert_receipt_candidate(
    receipt: dict[str, Any],
    identity: dict[str, Any],
    *,
    expected_channel: str,
) -> None:
    expected_receipt_keys = {
        "candidate",
        "channel",
        "index_url",
        "package",
        "promotion",
        "published",
        "schema_version",
        "status",
    }
    if set(receipt) != expected_receipt_keys:
        raise PromotionError("promotion receipt has missing or unclassified fields")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise PromotionError("promotion receipt schema version is unsupported")
    if receipt.get("channel") != expected_channel:
        raise PromotionError("promotion receipt channel is not the requested channel")
    if receipt.get("index_url") != INDEX_URLS[expected_channel]:
        raise PromotionError("promotion receipt index URL is not canonical for its channel")
    if receipt.get("status") != "accepted":
        raise PromotionError("promotion receipt is not an accepted publication")
    candidate = _candidate_identity_from_receipt(receipt)
    expected_candidate = {
        "artifact_digest": identity["artifact_digest"],
        "artifact_id": identity["artifact_id"],
        "artifact_name": identity["artifact_name"],
        "manifest_sha256": identity["manifest_sha256"],
        "members": identity["members"],
        "package": identity["package"],
        "provenance_sha256": identity["provenance_sha256"],
        "sbom_sha256": identity["sbom_sha256"],
        "source_sha": identity["source_sha"],
        "workflow_run_id": identity["workflow_run_id"],
    }
    if candidate != expected_candidate:
        raise PromotionError("promotion receipt is bound to a different candidate")
    published = receipt.get("published")
    if (
        not isinstance(published, dict)
        or set(published) != {"files"}
        or published.get("files") != _published_members(identity)
    ):
        raise PromotionError("promotion receipt published file hashes do not match the candidate")
    promotion = receipt.get("promotion")
    if (
        not isinstance(promotion, dict)
        or set(promotion) != {"workflow_run_id", "run_attempt"}
        or not isinstance(promotion.get("workflow_run_id"), str)
        or not RUN_ID_PATTERN.fullmatch(promotion["workflow_run_id"])
        or not isinstance(promotion.get("run_attempt"), int)
        or isinstance(promotion["run_attempt"], bool)
        or promotion["run_attempt"] < 1
    ):
        raise PromotionError("promotion receipt has no valid publisher workflow identity")


def _receipt_identity(
    *,
    identity: dict[str, Any],
    artifact_id: str,
    artifact_name: str,
    artifact_digest: str,
) -> dict[str, Any]:
    return {
        "artifact_digest": _artifact_digest(artifact_digest),
        "artifact_id": _positive_identity(
            artifact_id, label="artifact ID", pattern=ARTIFACT_ID_PATTERN
        ),
        "artifact_name": _artifact_name(artifact_name, kind="candidate"),
        "manifest_sha256": identity["manifest_sha256"],
        "members": identity["members"],
        "package": identity["package"],
        "provenance_sha256": identity["provenance_sha256"],
        "sbom_sha256": identity["sbom_sha256"],
        "source_sha": identity["source_sha"],
        "workflow_run_id": identity["workflow_run_id"],
    }


def _load_and_bind_candidate(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    _source_sha(args.source_sha)
    _version(args.version)
    _positive_identity(
        args.candidate_run_id, label="candidate workflow run ID", pattern=RUN_ID_PATTERN
    )
    _positive_identity(
        args.candidate_artifact_id, label="candidate artifact ID", pattern=ARTIFACT_ID_PATTERN
    )
    _artifact_name(args.candidate_artifact_name, kind="candidate")
    _artifact_digest(args.candidate_artifact_digest)
    manifest, identity = _load_candidate(
        args.candidate_dir,
        expected_source_sha=args.source_sha,
        expected_workflow_run_id=args.candidate_run_id,
        expected_version=args.version,
    )
    identity["artifact_digest"] = args.candidate_artifact_digest
    identity["artifact_id"] = args.candidate_artifact_id
    identity["artifact_name"] = args.candidate_artifact_name
    return manifest, identity


def _check_artifact(args: argparse.Namespace) -> None:
    _artifact_metadata(
        args.metadata,
        expected_id=args.artifact_id,
        expected_name=args.artifact_name,
        expected_digest=args.artifact_digest,
        expected_run_id=args.run_id,
        kind=args.kind,
        expected_source_sha=args.source_sha,
    )
    print(f"PASS: {args.kind} artifact metadata is bound to run {args.run_id}")


def _verify_candidate(args: argparse.Namespace) -> None:
    _manifest, identity = _load_and_bind_candidate(args)
    print(
        "PASS: immutable candidate verified "
        f"version={identity['package']['version']} source={identity['source_sha']} "
        f"artifact={identity['artifact_id']}"
    )


def _stage_packages(args: argparse.Namespace) -> None:
    _manifest, identity = _load_and_bind_candidate(args)
    candidate_root = args.candidate_dir.resolve()
    output_root = args.output_dir.resolve()
    if output_root == candidate_root or candidate_root in output_root.parents:
        raise PromotionError("package staging directory must be outside the candidate bundle")
    if args.output_dir.exists():
        _require_real_dir(args.output_dir, label="package staging directory")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise PromotionError(f"package staging directory must be empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for member in _published_members(identity):
        source = args.candidate_dir / member["filename"]
        target = args.output_dir / member["filename"]
        shutil.copyfile(source, target)
    print(f"PASS: staged {len(PUBLISHED_MEMBER_KINDS)} exact package files")


def _write_receipt(args: argparse.Namespace) -> None:
    if args.channel not in INDEX_URLS:
        raise PromotionError(f"unsupported promotion channel: {args.channel}")
    _manifest, identity = _load_and_bind_candidate(args)
    promotion_run_id = _positive_identity(
        args.promotion_run_id,
        label="promotion workflow run ID",
        pattern=RUN_ID_PATTERN,
    )
    if not isinstance(args.promotion_run_attempt, int) or args.promotion_run_attempt < 1:
        raise PromotionError("promotion workflow run attempt must be positive")
    receipt = {
        "candidate": _receipt_identity(
            identity=identity,
            artifact_id=args.candidate_artifact_id,
            artifact_name=args.candidate_artifact_name,
            artifact_digest=args.candidate_artifact_digest,
        ),
        "channel": args.channel,
        "index_url": INDEX_URLS[args.channel],
        "package": identity["package"],
        "promotion": {
            "run_attempt": args.promotion_run_attempt,
            "workflow_run_id": promotion_run_id,
        },
        "published": {"files": _published_members(identity)},
        "schema_version": SCHEMA_VERSION,
        "status": "accepted",
    }
    _assert_receipt_candidate(receipt, identity, expected_channel=args.channel)
    _write_new_json(args.receipt, receipt, label="promotion receipt")
    print(f"PASS: wrote {args.channel} promotion receipt without credentials")


def _load_receipt(path: Path) -> dict[str, Any]:
    _require_real_file(path, label="promotion receipt")
    receipt = _load_json(path, label="promotion receipt")
    if not isinstance(receipt, dict):
        raise PromotionError("promotion receipt must be an object")
    return receipt


def _verify_receipt(args: argparse.Namespace) -> None:
    _manifest, identity = _load_and_bind_candidate(args)
    receipt = _load_receipt(args.receipt)
    _assert_receipt_candidate(receipt, identity, expected_channel=args.channel)
    print(f"PASS: {args.channel} receipt reuses the exact candidate bytes")


def _report_digest(report_path: Path) -> str:
    _require_real_file(report_path, label="cold-install report")
    report = _load_json(report_path, label="cold-install report")
    if not isinstance(report, dict) or report.get("status") != "passed":
        raise PromotionError("cold-install report is not passed")
    if report.get("source_checkout_import") is not False:
        raise PromotionError("cold-install report does not prove a non-checkout import")
    failed_scripts = report.get("console_scripts_failed")
    if (
        isinstance(failed_scripts, bool)
        or not isinstance(failed_scripts, int)
        or failed_scripts != 0
    ):
        raise PromotionError("cold-install report contains failed console scripts")
    return _sha256(report_path)


def _find_package_member(identity: dict[str, Any], kind: str) -> dict[str, Any]:
    return next(member for member in identity["members"] if member["kind"] == kind)


def _verify_index_artifacts(args: argparse.Namespace) -> None:
    _manifest, identity = _load_and_bind_candidate(args)
    _require_real_dir(args.download_dir, label="index download directory")
    files = [
        path for path in args.download_dir.iterdir() if path.is_file() and not path.is_symlink()
    ]
    if len(files) != 2:
        raise PromotionError("index download must contain exactly one wheel and one sdist")
    expected = {member["filename"]: member for member in _published_members(identity)}
    if {path.name for path in files} != set(expected):
        raise PromotionError("index returned package filenames different from the candidate")
    for path in files:
        member = expected[path.name]
        if path.stat().st_size != member["size"] or _sha256(path) != member["sha256"]:
            raise PromotionError(f"public index bytes differ from candidate: {path.name}")
    print("PASS: public index returned byte-identical wheel and sdist")


def _write_cold_install_receipt(args: argparse.Namespace) -> None:
    if args.index_url != "https://test.pypi.org/simple":
        raise PromotionError("cold-install receipt must use the canonical TestPyPI simple index")
    _manifest, identity = _load_and_bind_candidate(args)
    test_receipt = _load_receipt(args.test_receipt)
    _assert_receipt_candidate(test_receipt, identity, expected_channel="testpypi")
    wheel_member = _find_package_member(identity, "wheel")
    _require_real_file(args.downloaded_wheel, label="downloaded TestPyPI wheel")
    if args.downloaded_wheel.name != wheel_member["filename"]:
        raise PromotionError("downloaded TestPyPI wheel filename differs from candidate")
    if (
        args.downloaded_wheel.stat().st_size != wheel_member["size"]
        or _sha256(args.downloaded_wheel) != wheel_member["sha256"]
    ):
        raise PromotionError("downloaded TestPyPI wheel bytes differ from candidate")
    report_sha256 = _report_digest(args.report)
    receipt = {
        "candidate": _receipt_identity(
            identity=identity,
            artifact_id=args.candidate_artifact_id,
            artifact_name=args.candidate_artifact_name,
            artifact_digest=args.candidate_artifact_digest,
        ),
        "index_url": args.index_url,
        "report": {"filename": args.report.name, "sha256": report_sha256},
        "schema_version": COLD_INSTALL_SCHEMA_VERSION,
        "status": "passed",
        "testpypi_receipt_sha256": _sha256(args.test_receipt),
        "wheel": wheel_member,
    }
    _write_new_json(args.receipt, receipt, label="cold-install receipt")
    print("PASS: wrote TestPyPI cold-install receipt")


def _verify_cold_install(args: argparse.Namespace) -> None:
    _manifest, identity = _load_and_bind_candidate(args)
    test_receipt = _load_receipt(args.test_receipt)
    _assert_receipt_candidate(test_receipt, identity, expected_channel="testpypi")
    _require_real_file(args.report, label="cold-install report")
    _require_real_file(args.receipt, label="cold-install receipt")
    cold = _load_json(args.receipt, label="cold-install receipt")
    if not isinstance(cold, dict) or cold.get("schema_version") != COLD_INSTALL_SCHEMA_VERSION:
        raise PromotionError("cold-install receipt schema version is unsupported")
    if set(cold) != {
        "candidate",
        "index_url",
        "report",
        "schema_version",
        "status",
        "testpypi_receipt_sha256",
        "wheel",
    }:
        raise PromotionError("cold-install receipt has missing or unclassified fields")
    if cold.get("status") != "passed" or cold.get("index_url") != "https://test.pypi.org/simple":
        raise PromotionError("cold-install receipt is not a passed TestPyPI result")
    if cold.get("candidate") != _receipt_identity(
        identity=identity,
        artifact_id=args.candidate_artifact_id,
        artifact_name=args.candidate_artifact_name,
        artifact_digest=args.candidate_artifact_digest,
    ):
        raise PromotionError("cold-install receipt is bound to a different candidate")
    if cold.get("testpypi_receipt_sha256") != _sha256(args.test_receipt):
        raise PromotionError("cold-install receipt is bound to a different TestPyPI receipt")
    report = cold.get("report")
    if not isinstance(report, dict) or report != {
        "filename": args.report.name,
        "sha256": _sha256(args.report),
    }:
        raise PromotionError("cold-install report bytes differ from its receipt")
    wheel = _find_package_member(identity, "wheel")
    if cold.get("wheel") != wheel:
        raise PromotionError("cold-install wheel binding differs from candidate")
    _report_digest(args.report)
    print("PASS: TestPyPI cold-install receipt is bound to the exact candidate")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    artifact = subparsers.add_parser("check-artifact", help="verify GitHub artifact metadata")
    artifact.add_argument("--metadata", type=Path, required=True)
    artifact.add_argument("--artifact-id", required=True)
    artifact.add_argument("--artifact-name", required=True)
    artifact.add_argument("--artifact-digest", required=True)
    artifact.add_argument("--run-id", required=True)
    artifact.add_argument("--kind", choices=("candidate", "receipt"), required=True)
    artifact.add_argument("--source-sha")

    def add_candidate_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--candidate-dir", type=Path, required=True)
        command.add_argument("--source-sha", required=True)
        command.add_argument("--candidate-run-id", required=True)
        command.add_argument("--candidate-artifact-id", required=True)
        command.add_argument("--candidate-artifact-name", required=True)
        command.add_argument("--candidate-artifact-digest", required=True)
        command.add_argument("--version", required=True)

    verify_candidate = subparsers.add_parser("verify-candidate", help="verify candidate bytes")
    add_candidate_arguments(verify_candidate)

    stage = subparsers.add_parser("stage-packages", help="stage exact package bytes for upload")
    add_candidate_arguments(stage)
    stage.add_argument("--output-dir", type=Path, required=True)

    write_receipt = subparsers.add_parser("write-receipt", help="write an accepted upload receipt")
    add_candidate_arguments(write_receipt)
    write_receipt.add_argument("--channel", choices=tuple(INDEX_URLS), required=True)
    write_receipt.add_argument("--promotion-run-id", required=True)
    write_receipt.add_argument("--promotion-run-attempt", type=int, required=True)
    write_receipt.add_argument("--receipt", type=Path, required=True)

    verify_receipt = subparsers.add_parser("verify-receipt", help="verify an upload receipt")
    add_candidate_arguments(verify_receipt)
    verify_receipt.add_argument("--channel", choices=tuple(INDEX_URLS), required=True)
    verify_receipt.add_argument("--receipt", type=Path, required=True)

    index = subparsers.add_parser(
        "verify-index-artifacts", help="verify public-index package bytes"
    )
    add_candidate_arguments(index)
    index.add_argument("--download-dir", type=Path, required=True)

    cold = subparsers.add_parser("write-cold-install-receipt", help="write a TestPyPI cold receipt")
    add_candidate_arguments(cold)
    cold.add_argument("--test-receipt", type=Path, required=True)
    cold.add_argument("--downloaded-wheel", type=Path, required=True)
    cold.add_argument("--index-url", required=True)
    cold.add_argument("--report", type=Path, required=True)
    cold.add_argument("--receipt", type=Path, required=True)

    verify_cold = subparsers.add_parser(
        "verify-cold-install", help="verify a TestPyPI cold-install receipt"
    )
    add_candidate_arguments(verify_cold)
    verify_cold.add_argument("--test-receipt", type=Path, required=True)
    verify_cold.add_argument("--report", type=Path, required=True)
    verify_cold.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one fail-closed promotion verification or receipt command."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "check-artifact":
            _check_artifact(args)
        elif args.command == "verify-candidate":
            _verify_candidate(args)
        elif args.command == "stage-packages":
            _stage_packages(args)
        elif args.command == "write-receipt":
            _write_receipt(args)
        elif args.command == "verify-receipt":
            _verify_receipt(args)
        elif args.command == "verify-index-artifacts":
            _verify_index_artifacts(args)
        elif args.command == "write-cold-install-receipt":
            _write_cold_install_receipt(args)
        elif args.command == "verify-cold-install":
            _verify_cold_install(args)
        else:  # pragma: no cover - argparse enforces the command choices.
            raise PromotionError(f"unsupported command: {args.command}")
    except PromotionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
