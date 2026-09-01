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
import tarfile
import zipfile
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "robot_sf.software_promotion.receipt.v1"
COLD_INSTALL_SCHEMA_VERSION = "robot_sf.software_promotion.cold_install.v1"
INDEX_VERIFICATION_SCHEMA_VERSION = "robot_sf.software_promotion.index_verification.v1"
RIGHTS_ADMISSION_SCHEMA_VERSION = "robot_sf.software_rights_admission.v1"
REPOSITORY = "ll7/robot_sf_ll7"
INDEX_URLS = {
    "testpypi": "https://test.pypi.org/legacy/",
    "pypi": "https://upload.pypi.org/legacy/",
}
PUBLIC_INDEX_URLS = {
    "testpypi": "https://test.pypi.org/simple",
    "pypi": "https://pypi.org/simple",
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
RIGHTS_ARTIFACT_NAME_PATTERN = re.compile(
    r"robot-sf-software-rights-admission-[0-9a-f]{40}-[1-9][0-9]*-[1-9][0-9]*\Z"
)
CANDIDATE_MEMBER_KINDS = ("wheel", "sdist", "sbom", "provenance")
CANDIDATE_MATERIALIZATION_FIELDS = frozenset(
    {
        "candidate_commit_sha",
        "candidate_tree_sha",
        "policy_path",
        "policy_sha256",
        "source_inventory_path",
        "source_inventory_sha256",
        "candidate_inventory_path",
        "candidate_metadata_path",
    }
)
CANDIDATE_MATERIALIZATION_PATH_FIELDS = frozenset(
    {
        "policy_path",
        "source_inventory_path",
        "candidate_inventory_path",
        "candidate_metadata_path",
    }
)
CANDIDATE_MATERIALIZATION_SHA_FIELDS = frozenset({"policy_sha256", "source_inventory_sha256"})
PUBLISHED_MEMBER_KINDS = ("wheel", "sdist")
MANIFEST_NAME = "candidate-manifest.json"
PROVENANCE_NAME = "candidate-provenance.json"
RIGHTS_RECEIPT_NAME = "rights-admission.json"
RIGHTS_POLICY_ID = "robot_sf.software_release_rights_policy.v1"
RIGHTS_POLICY_PATH = "scripts/validation/software_release_rights_policy.v1.json"
SANITIZED_CANDIDATE_SCHEMA = "robot_sf.software_sanitized_candidate.v1"
RIGHTS_GATE_ID = "strict-distribution-rights"
RIGHTS_WORKFLOW_PATH = ".github/workflows/software-candidate.yml"
PROMOTION_WORKFLOW_PATH = ".github/workflows/software-promotion.yml"
# The current producer contract is a direct, reviewed manual dispatch.  A
# reusable caller would need its own explicit identity binding before it could
# be admitted; accepting ``workflow_call`` based only on the workflow path
# would not identify that caller.
RIGHTS_WORKFLOW_EVENTS = frozenset({"workflow_dispatch"})
PRODUCER_WORKFLOW_EVENTS = frozenset({"workflow_dispatch"})
RIGHTS_GATE_COMMAND = (
    "python scripts/tools/check_distribution_licenses.py $DIST_DIR "
    "--strict-asset-rights --repo-root $BUILD_SOURCE --source-tree-ref $SOURCE_SHA"
)
SUPPORTED_DEPENDENCY_SCHEMA_VERSION = "robot-sf.dependency-license-inventory.v1"
SUPPORTED_DEPENDENCY_POLICY_SCHEMA_VERSION = "robot-sf.dependency-license-policy.v1"
SUPPORTED_DEPENDENCY_PROFILE_SCHEMA_VERSION = "robot-sf.dependency-license-profiles.v1"
SUPPORTED_RELEASE_EXTRAS = frozenset(
    {
        "viz",
        "maps",
        "benchmark",
        "training",
        "gpu",
        "recurrent",
        "progress",
        "analytics",
        "browser",
        "sacadrl",
        "socnav",
        "criticality",
    }
)
# ``all`` is the supported aggregator published by the sanitized candidate;
# standalone ``rllib`` remains intentionally outside the release surface.
SUPPORTED_RELEASE_DISTRIBUTION_EXTRAS = SUPPORTED_RELEASE_EXTRAS | {"all"}
# Preserve the complete checked-in profile-manifest order. The selected
# release surface is still ``all``; the canonical manifest intentionally also
# records non-release profiles as visible context.
SUPPORTED_RELEASE_PROFILE_ROSTER = (
    "core",
    "viz",
    "maps",
    "benchmark",
    "gpu",
    "training",
    "recurrent",
    "rllib",
    "progress",
    "analytics",
    "browser",
    "sacadrl",
    "socnav",
    "criticality",
    "all",
    "fast-pysf",
    "socnavbench",
)
SUPPORTED_DEPENDENCY_GATE_ID = "strict-supported-dependency-surface"
SUPPORTED_DEPENDENCY_REPORT_NAME = "dependency-license-inventory.json"
SUPPORTED_DEPENDENCY_POLICY_PATH = "scripts/validation/dependency_license_policy.v1.json"
SUPPORTED_DEPENDENCY_PROFILE_PATH = "scripts/validation/dependency_license_profiles.v1.json"
SUPPORTED_DEPENDENCY_GATE_COMMAND = (
    "python scripts/tools/check_dependency_license_inventory.py "
    "--repo-root $BUILD_SOURCE --output $DEPENDENCY_REPORT "
    "--candidate-bundle $CANDIDATE_BUNDLE --fail-on-unresolved"
)
VALIDATION_ROSTER = (
    ("version-alignment", "python scripts/dev/check_version_alignment.py"),
    ("metadata", "twine check --strict $DIST_DIR/*.whl $DIST_DIR/*.tar.gz"),
    (
        "archive-license",
        "cd $BUILD_SOURCE && python scripts/tools/check_distribution_licenses.py "
        "$DIST_DIR --strict-asset-rights --repo-root $BUILD_SOURCE "
        "--inventory $BUILD_SOURCE/scripts/validation/software_candidate_asset_rights.v1.json "
        "--source-tree-ref HEAD",
    ),
    (
        "wheel-install",
        "cd $BUILD_SOURCE && bash scripts/validation/wheel_install_smoke.sh "
        "$DIST_DIR/robot_sf-*.whl",
    ),
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
    pattern = {
        "candidate": ARTIFACT_NAME_PATTERN,
        "receipt": RECEIPT_ARTIFACT_NAME_PATTERN,
        "rights": RIGHTS_ARTIFACT_NAME_PATTERN,
    }.get(kind)
    if pattern is None:
        raise PromotionError(f"unsupported artifact kind: {kind}")
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


def _validate_validation_roster(validation: Any) -> None:  # noqa: C901 - versioned roster gate
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise PromotionError("candidate validation envelope is not passed")
    checks = validation.get("checks")
    if not isinstance(checks, list) or any(not isinstance(item, dict) for item in checks):
        raise PromotionError("candidate validation roster is malformed")
    if len({item.get("id") for item in checks}) != len(checks):
        raise PromotionError("candidate validation roster contains duplicate IDs")
    required = {identifier for identifier, _command in VALIDATION_ROSTER}
    observed = {item.get("id") for item in checks}
    if not required.issubset(observed):
        raise PromotionError("candidate validation roster is missing a required base check")
    for item in checks:
        if set(item) != {"command", "id", "status"}:
            raise PromotionError("candidate validation roster contains unclassified fields")
        if not isinstance(item["id"], str) or not item["id"]:
            raise PromotionError("candidate validation roster contains an invalid check ID")
        if (
            item["status"] != "passed"
            or not isinstance(item["command"], str)
            or not item["command"]
        ):
            raise PromotionError("candidate validation roster contains an unpassed check")
    expected_commands = dict(VALIDATION_ROSTER)
    for identifier, command in expected_commands.items():
        matching = next(item for item in checks if item["id"] == identifier)
        if identifier == "archive-license":
            # The sanitized candidate workflow may replace the broad candidate gate with its strict
            # rights-clean command.  The separate rights admission receipt
            # below remains mandatory, so accepting that additive upgrade
            # does not permit the current unresolved tree to publish.
            accepted_strict_command = matching["command"] == RIGHTS_GATE_COMMAND or (
                "scripts/tools/check_distribution_licenses.py" in matching["command"]
                and "--strict-asset-rights" in matching["command"]
            )
            if matching["command"] != command and not accepted_strict_command:
                raise PromotionError("candidate archive-license check command is unsupported")
        elif matching["command"] != command:
            raise PromotionError(f"candidate validation command drifted for {identifier}")


def _validate_candidate_materialization(value: Any) -> dict[str, Any]:
    """Validate the optional rights-scoped source identity envelope."""

    if not isinstance(value, dict) or set(value) != CANDIDATE_MATERIALIZATION_FIELDS:
        raise PromotionError("candidate materialization identity is missing or unclassified")
    for field in ("candidate_commit_sha", "candidate_tree_sha"):
        identity = value.get(field)
        if not isinstance(identity, str) or not SHA_PATTERN.fullmatch(identity):
            raise PromotionError(f"candidate materialization {field} is invalid")
    for field in CANDIDATE_MATERIALIZATION_SHA_FIELDS:
        digest = value.get(field)
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise PromotionError(f"candidate materialization {field} is invalid")

    paths: list[str] = []
    for field in CANDIDATE_MATERIALIZATION_PATH_FIELDS:
        path = value.get(field)
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or "\\" in path
            or "\x00" in path
            or PurePosixPath(path).as_posix() != path
        ):
            raise PromotionError(f"candidate materialization {field} is invalid")
        parts = PurePosixPath(path).parts
        if (
            not parts
            or parts[0] == ".git"
            or any(part in {"", ".", ".."} for part in parts)
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in path)
        ):
            raise PromotionError(f"candidate materialization {field} is invalid")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise PromotionError("candidate materialization paths must be distinct")
    return value


def _validate_optional_candidate_materialization(
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    """Validate and return a manifest's optional materialization envelope."""

    if "materialization" not in manifest:
        return None
    return _validate_candidate_materialization(manifest["materialization"])


def _validate_manifest_header(  # noqa: C901 - closed manifest contract
    manifest: Any,
    *,
    expected_source_sha: str,
    expected_workflow_run_id: str,
    expected_workflow_run_attempt: int,
    expected_version: str,
) -> tuple[str, str, dict[str, str], dict[str, Any] | None]:
    if not isinstance(manifest, dict):
        raise PromotionError("candidate manifest must be a JSON object")
    required_keys = {
        "schema_version",
        "repository",
        "source_sha",
        "workflow",
        "package",
        "validation",
        "members",
    }
    if set(manifest) not in (required_keys, required_keys | {"materialization"}):
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
    if workflow["run_attempt"] != expected_workflow_run_attempt:
        raise PromotionError("candidate workflow run attempt does not match the dispatch identity")

    package = manifest["package"]
    version = _version(expected_version)
    if package != {"name": "robot_sf", "version": version}:
        raise PromotionError("candidate package identity does not match the dispatch identity")
    _validate_validation_roster(manifest["validation"])
    return source_sha, run_id, package, _validate_optional_candidate_materialization(manifest)


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
    materialization: dict[str, Any] | None,
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
    _validate_provenance_materialization(provenance, materialization)


def _validate_provenance_materialization(
    provenance: dict[str, Any], materialization: dict[str, Any] | None
) -> None:
    """Require provenance to mirror the manifest's optional materialization envelope."""

    if materialization is None:
        if "materialization" in provenance:
            raise PromotionError(
                "candidate provenance materialization is not bound by the manifest"
            )
    elif provenance.get("materialization") != materialization:
        raise PromotionError("candidate provenance materialization differs from the manifest")


def _distribution_extras(path: Path, *, kind: str) -> frozenset[str]:
    """Read case-insensitive extras from one exact wheel or sdist metadata file."""

    _require_real_file(path, label=f"candidate {kind}")
    try:
        if kind == "wheel":
            with zipfile.ZipFile(path) as archive:
                metadata_names = [
                    name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
                ]
                if len(metadata_names) != 1:
                    raise PromotionError("candidate wheel must contain exactly one METADATA file")
                metadata_bytes = archive.read(metadata_names[0])
        elif kind == "sdist":
            with tarfile.open(path, "r:*") as archive:
                metadata_members = [
                    member for member in archive.getmembers() if member.name.endswith("/PKG-INFO")
                ]
                if len(metadata_members) != 1:
                    raise PromotionError("candidate sdist must contain exactly one PKG-INFO file")
                extracted = archive.extractfile(metadata_members[0])
                if extracted is None:
                    raise PromotionError("candidate sdist PKG-INFO is unreadable")
                metadata_bytes = extracted.read()
        else:
            raise PromotionError(f"unsupported candidate distribution kind: {kind}")
        metadata = BytesParser(policy=email_policy).parsebytes(metadata_bytes)
    except (OSError, UnicodeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise PromotionError(f"candidate {kind} metadata is unreadable: {path}") from exc
    extras = frozenset(
        value.strip().lower()
        for value in metadata.get_all("Provides-Extra", [])
        if isinstance(value, str) and value.strip()
    )
    if extras != SUPPORTED_RELEASE_DISTRIBUTION_EXTRAS:
        missing = sorted(SUPPORTED_RELEASE_DISTRIBUTION_EXTRAS - extras)
        unsupported = sorted(extras - SUPPORTED_RELEASE_DISTRIBUTION_EXTRAS)
        raise PromotionError(
            f"candidate {kind} Provides-Extra values differ from the closed supported surface: "
            f"missing={missing}, unsupported={unsupported}"
        )
    return extras


def _load_candidate(
    bundle_dir: Path,
    *,
    expected_source_sha: str,
    expected_workflow_run_id: str,
    expected_workflow_run_attempt: int,
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
    source_sha, run_id, package, materialization = _validate_manifest_header(
        manifest,
        expected_source_sha=expected_source_sha,
        expected_workflow_run_id=expected_workflow_run_id,
        expected_workflow_run_attempt=expected_workflow_run_attempt,
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
        materialization=materialization,
    )

    identity = {
        "source_sha": source_sha,
        "workflow_run_id": run_id,
        "workflow_run_attempt": manifest["workflow"]["run_attempt"],
        "package": package,
        "members": validated_members,
        "manifest_sha256": _sha256(manifest_path),
        "provenance_sha256": provenance_member["sha256"],
        "sbom_sha256": validated_members[2]["sha256"],
        "materialization": materialization,
        "distribution_extras": _distribution_extras(
            bundle_dir / validated_members[0]["filename"], kind="wheel"
        ),
    }
    sdist_extras = _distribution_extras(bundle_dir / validated_members[1]["filename"], kind="sdist")
    if sdist_extras != identity["distribution_extras"]:
        raise PromotionError("candidate wheel and sdist advertise different optional extras")
    return manifest, identity


def _validate_artifact_source(
    workflow_run: dict[str, Any],
    *,
    kind: str,
    expected_source_sha: str | None,
) -> None:
    if kind in {"candidate", "rights"}:
        if workflow_run.get("head_sha") != _source_sha(expected_source_sha or ""):
            raise PromotionError(
                f"GitHub {kind} artifact source SHA drifted from the dispatch identity"
            )
    elif expected_source_sha is not None:
        raise PromotionError("source SHA is only valid for candidate artifact metadata")


def _artifact_metadata(  # noqa: C901, PLR0912 - identity branches fail closed
    metadata_path: Path,
    *,
    expected_id: str,
    expected_name: str,
    expected_digest: str,
    expected_run_id: str,
    kind: str,
    expected_source_sha: str | None = None,
    expected_run_attempt: int | None = None,
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
    expected_run = _positive_identity(
        expected_run_id,
        label="artifact workflow run ID",
        pattern=RUN_ID_PATTERN,
    )
    if expected_run_attempt is not None and (
        isinstance(expected_run_attempt, bool) or expected_run_attempt < 1
    ):
        raise PromotionError("artifact workflow run attempt must be positive")
    if kind == "candidate":
        expected_prefix = f"robot-sf-software-candidate-{_source_sha(expected_source_sha or '')}-"
        if not actual_name.startswith(expected_prefix) or not actual_name[
            len(expected_prefix) :
        ].startswith(f"{expected_run}-"):
            raise PromotionError("GitHub candidate artifact name is not bound to source/run")
    elif kind == "rights":
        expected_prefix = (
            f"robot-sf-software-rights-admission-{_source_sha(expected_source_sha or '')}-"
        )
        if not actual_name.startswith(expected_prefix) or not actual_name[
            len(expected_prefix) :
        ].startswith(f"{expected_run}-"):
            raise PromotionError("GitHub rights artifact name is not bound to source/run")
    elif actual_name.split("-")[-2] != expected_run:
        raise PromotionError("GitHub receipt artifact name is not bound to its workflow run")
    if expected_run_attempt is not None and actual_name.rsplit("-", 1)[-1] != str(
        expected_run_attempt
    ):
        raise PromotionError("GitHub artifact name is not bound to its workflow run attempt")
    if metadata.get("expired") is not False:
        raise PromotionError("GitHub artifact is expired or has no explicit unexpired status")
    digest = metadata.get("digest")
    if digest != _artifact_digest(expected_digest):
        raise PromotionError("GitHub artifact archive digest drifted from the dispatch identity")
    workflow_run = metadata.get("workflow_run")
    if not isinstance(workflow_run, dict):
        raise PromotionError("GitHub artifact has no workflow-run binding")
    if str(workflow_run.get("id")) != expected_run:
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


def _validate_workflow_run_metadata(  # noqa: C901 - closed metadata contract
    metadata: Any,
    *,
    run_id: str,
    run_attempt: int | None,
    source_sha: str,
    kind: str,
) -> int:
    """Validate one canonical, successful producer or promotion run."""

    if not isinstance(metadata, dict):
        raise PromotionError(f"{kind} workflow-run metadata must be an object")
    expected_run_id = _positive_identity(
        run_id,
        label=f"{kind} workflow run ID",
        pattern=RUN_ID_PATTERN,
    )
    expected_source = _source_sha(source_sha)
    paths = {
        "candidate": RIGHTS_WORKFLOW_PATH,
        "rights": RIGHTS_WORKFLOW_PATH,
        "promotion": PROMOTION_WORKFLOW_PATH,
    }
    if kind not in paths:
        raise PromotionError(f"unsupported workflow-run kind: {kind}")
    if str(metadata.get("id")) != expected_run_id:
        raise PromotionError(f"{kind} workflow-run ID drifted from the dispatch identity")
    if metadata.get("path") != paths[kind]:
        raise PromotionError(f"{kind} admission was not produced by the sanctioned workflow")
    if metadata.get("event") not in PRODUCER_WORKFLOW_EVENTS:
        allowed = ", ".join(sorted(PRODUCER_WORKFLOW_EVENTS))
        raise PromotionError(f"{kind} workflow event is not sanctioned ({allowed})")
    if metadata.get("head_sha") != expected_source:
        raise PromotionError(f"{kind} workflow-run source SHA drifted from the dispatch identity")
    if metadata.get("status") != "completed" or metadata.get("conclusion") != "success":
        raise PromotionError(f"{kind} workflow-run did not complete successfully")
    repository = metadata.get("repository")
    if not isinstance(repository, dict) or repository.get("full_name") != REPOSITORY:
        raise PromotionError(f"{kind} workflow-run repository is not canonical")
    workflow_id = metadata.get("workflow_id")
    if isinstance(workflow_id, bool) or not isinstance(workflow_id, int) or workflow_id < 1:
        raise PromotionError(f"{kind} workflow-run has no valid workflow identity")
    observed_attempt = metadata.get("run_attempt")
    if (
        isinstance(observed_attempt, bool)
        or not isinstance(observed_attempt, int)
        or observed_attempt < 1
    ):
        raise PromotionError(f"{kind} workflow-run has no valid run attempt")
    if run_attempt is not None and observed_attempt != run_attempt:
        raise PromotionError(f"{kind} workflow-run attempt drifted from the dispatch identity")
    return observed_attempt


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


def _assert_receipt_candidate(  # noqa: C901 - closed receipt contract
    receipt: dict[str, Any],
    identity: dict[str, Any],
    *,
    expected_channel: str,
    expected_promotion_run_id: str | None = None,
    expected_promotion_run_attempt: int | None = None,
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
    if receipt.get("package") != identity["package"]:
        raise PromotionError("promotion receipt package identity differs from candidate")
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
    if expected_promotion_run_id is not None:
        expected_run_id = _positive_identity(
            expected_promotion_run_id,
            label="expected promotion workflow run ID",
            pattern=RUN_ID_PATTERN,
        )
        if promotion["workflow_run_id"] != expected_run_id:
            raise PromotionError("promotion receipt is bound to a different publisher run")
    if expected_promotion_run_attempt is not None:
        if (
            isinstance(expected_promotion_run_attempt, bool)
            or expected_promotion_run_attempt < 1
            or promotion["run_attempt"] != expected_promotion_run_attempt
        ):
            raise PromotionError("promotion receipt is bound to a different publisher attempt")


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


def _dependency_input_digest(report: dict[str, Any], path: str, *, label: str) -> str:
    """Return the uniquely recorded digest for one canonical dependency input."""

    inputs = report.get("repository_inputs")
    if not isinstance(inputs, list):
        raise PromotionError("supported dependency report has no repository input digest list")
    matches = [item for item in inputs if isinstance(item, dict) and item.get("path") == path]
    if len(matches) != 1:
        raise PromotionError(f"supported dependency report must bind exactly one {label} input")
    digest = matches[0].get("sha256")
    if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise PromotionError(f"supported dependency report {label} digest is invalid")
    return digest


def _validate_canonical_dependency_inputs(
    report: dict[str, Any],
    report_path: Path,
    *,
    repo_root: Path,
) -> None:
    """Bind report inputs and the full profile roster to the trusted checkout."""

    try:
        from scripts.tools.check_dependency_license_inventory import check_report_freshness

        freshness_issues = check_report_freshness(repo_root, report_path)
    except (OSError, ValueError, ImportError) as exc:
        raise PromotionError("canonical dependency report freshness could not be checked") from exc
    # The consumer independently validates candidate/package bindings below;
    # the canonical helper's candidate-bundle requirement is therefore not an
    # input-freshness failure at this boundary.
    actionable = [
        issue
        for issue in freshness_issues
        if issue != "candidate-bound report freshness requires --candidate-bundle"
    ]
    if actionable:
        raise PromotionError(f"canonical dependency report is stale: {actionable}")

    canonical_path = repo_root / SUPPORTED_DEPENDENCY_PROFILE_PATH
    canonical = _load_json(canonical_path, label="canonical dependency profile manifest")
    profiles = canonical.get("profiles")
    if not isinstance(profiles, list):
        raise PromotionError("canonical dependency profile manifest has no profile list")
    canonical_ids = [profile.get("id") for profile in profiles if isinstance(profile, dict)]
    if canonical_ids != list(SUPPORTED_RELEASE_PROFILE_ROSTER):
        raise PromotionError("trusted checkout profile manifest roster is not the closed contract")
    embedded = report.get("profile_manifest")
    if not isinstance(embedded, dict) or embedded.get("profile_ids") != canonical_ids:
        raise PromotionError("dependency report profile roster differs from trusted checkout")


def _validate_supported_dependency_report(  # noqa: C901, PLR0912, PLR0915 - closed report contract
    path: Path,
    *,
    identity: dict[str, Any],
    tree_sha256: str,
    expected_gate: dict[str, Any],
    repo_root: Path,
) -> None:
    """Validate the transported dependency report, not only its receipt digest."""

    _require_real_file(path, label="supported dependency report")
    if path.name != SUPPORTED_DEPENDENCY_REPORT_NAME:
        raise PromotionError(
            f"supported dependency report must be named {SUPPORTED_DEPENDENCY_REPORT_NAME}"
        )
    report = _load_json(path, label="supported dependency report")
    if not isinstance(report, dict):
        raise PromotionError("supported dependency report must be a JSON object")
    if report.get("schema_version") != SUPPORTED_DEPENDENCY_SCHEMA_VERSION:
        raise PromotionError("supported dependency report schema version is unsupported")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise PromotionError("supported dependency report has no summary")
    if summary.get("status") != "complete" or summary.get("candidate_bound") is not True:
        raise PromotionError("supported dependency report is not a complete candidate binding")
    unresolved = summary.get("unresolved_count")
    if isinstance(unresolved, bool) or not isinstance(unresolved, int) or unresolved != 0:
        raise PromotionError("supported dependency report contains unresolved rows")
    if report.get("failures") != [] or report.get("structural_issues") != []:
        raise PromotionError("supported dependency report contains failures")
    _validate_canonical_dependency_inputs(report, path, repo_root=repo_root)

    profile_manifest = report.get("profile_manifest")
    if not isinstance(profile_manifest, dict) or profile_manifest.get("path") != (
        SUPPORTED_DEPENDENCY_PROFILE_PATH
    ):
        raise PromotionError("supported dependency report profile manifest path is unsupported")
    if profile_manifest.get("schema_version") != SUPPORTED_DEPENDENCY_PROFILE_SCHEMA_VERSION:
        raise PromotionError("supported dependency report profile schema is unsupported")
    surface = report.get("surface")
    profile_ids = surface.get("profile_ids") if isinstance(surface, dict) else None
    if (
        not isinstance(profile_ids, list)
        or not profile_ids
        or any(not isinstance(value, str) or not value for value in profile_ids)
    ):
        raise PromotionError("supported dependency report selected profile surface is invalid")
    # #8146's release surface is selected through the closed ``all`` profile;
    # that profile expands to the twelve supported extras.  A core-only report
    # is therefore not a software release admission.
    if profile_ids != ["all"]:
        raise PromotionError(
            "supported dependency report profile roster differs from the v0.0.6 supported surface"
        )
    embedded_profile_ids = profile_manifest.get("profile_ids")
    if embedded_profile_ids != list(SUPPORTED_RELEASE_PROFILE_ROSTER):
        raise PromotionError(
            "supported dependency report embedded profile roster differs from the trusted "
            "canonical profile manifest"
        )
    if identity["distribution_extras"] != SUPPORTED_RELEASE_DISTRIBUTION_EXTRAS:
        raise PromotionError("candidate archive does not advertise the supported profile surface")
    policy = report.get("policy")
    if not isinstance(policy, dict) or policy.get("path") != SUPPORTED_DEPENDENCY_POLICY_PATH:
        raise PromotionError("supported dependency report policy path is unsupported")
    if policy.get("schema_version") != SUPPORTED_DEPENDENCY_POLICY_SCHEMA_VERSION:
        raise PromotionError("supported dependency report policy schema is unsupported")

    binding = report.get("candidate_binding")
    expected_binding = {
        "status": "bound",
        "repository": REPOSITORY,
        "source_sha": identity["source_sha"],
        "workflow": {
            "run_id": identity["workflow_run_id"],
            "run_attempt": identity["workflow_run_attempt"],
        },
        "package": identity["package"],
    }
    if not isinstance(binding, dict) or any(
        binding.get(key) != value for key, value in expected_binding.items()
    ):
        raise PromotionError("supported dependency report candidate binding differs from candidate")
    if binding.get("profile_ids") != ["all"]:
        raise PromotionError("supported dependency report candidate profile binding is not all")
    if binding.get("manifest_sha256") != identity["manifest_sha256"]:
        raise PromotionError("supported dependency report manifest binding differs from candidate")
    members = binding.get("members")
    expected_members = {member["kind"]: member for member in identity["members"]}
    if (
        not isinstance(members, list)
        or {member.get("kind"): member for member in members if isinstance(member, dict)}
        != expected_members
    ):
        raise PromotionError("supported dependency report member binding differs from candidate")
    sbom = binding.get("sbom")
    if not isinstance(sbom, dict) or sbom.get("sha256") != identity["sbom_sha256"]:
        raise PromotionError("supported dependency report SBOM binding differs from candidate")
    _validate_dependency_materialization_binding(binding, identity)

    policy_sha256 = _dependency_input_digest(
        report, SUPPORTED_DEPENDENCY_POLICY_PATH, label="policy"
    )
    profile_sha256 = _dependency_input_digest(
        report, SUPPORTED_DEPENDENCY_PROFILE_PATH, label="profile manifest"
    )
    if (
        expected_gate["policy_sha256"] != policy_sha256
        or expected_gate["profile_manifest_sha256"] != profile_sha256
    ):
        raise PromotionError("supported dependency report input digests differ from receipt")
    if expected_gate["report_sha256"] != _sha256(path):
        raise PromotionError("supported dependency report bytes differ from receipt")
    if expected_gate["candidate_tree_sha256"] != tree_sha256:
        raise PromotionError("supported dependency report tree binding differs from receipt")


def _validate_dependency_materialization_binding(
    binding: dict[str, Any], identity: dict[str, Any]
) -> None:
    """Require a dependency report to mirror the candidate materialization envelope."""

    materialization = identity["materialization"]
    if materialization is None:
        if binding.get("materialization") is not None:
            raise PromotionError(
                "supported dependency report materialization binding is not bound by candidate"
            )
    elif binding.get("materialization") != materialization:
        raise PromotionError(
            "supported dependency report materialization binding differs from candidate"
        )


def _validate_rights_admission(  # noqa: C901, PLR0912, PLR0915 - closed rights receipt gate
    path: Path,
    *,
    identity: dict[str, Any],
    repo_root: Path,
) -> None:
    """Require the independent rights-clean admission before publication.

    The current build candidate is intentionally not rights-clean.  A separate
    receipt produced by the sanitized-candidate workflow must therefore bind
    the exact candidate artifact and manifest, its policy digest, and a passed
    strict archive/tree gate.  This contract is deliberately checked in this
    publisher, so protected-environment configuration alone cannot publish an
    unresolved development tree.
    """

    if path.name != RIGHTS_RECEIPT_NAME:
        raise PromotionError(f"rights admission receipt must be named {RIGHTS_RECEIPT_NAME}")
    _require_real_dir(path.parent, label="rights admission artifact directory")
    entries = sorted(path.parent.iterdir(), key=lambda entry: entry.name)
    expected_entries = sorted([RIGHTS_RECEIPT_NAME, SUPPORTED_DEPENDENCY_REPORT_NAME])
    if [entry.name for entry in entries] != expected_entries:
        raise PromotionError(
            "rights admission artifact must contain exactly the receipt and dependency report"
        )
    _require_real_file(path, label="rights admission receipt")
    receipt = _load_json(path, label="rights admission receipt")
    if not isinstance(receipt, dict):
        raise PromotionError("rights admission receipt must be a JSON object")
    if set(receipt) != {
        "candidate",
        "sanitized",
        "strict_gate",
        "status",
        "schema_version",
        "supported_dependency_gate",
    }:
        raise PromotionError("rights admission receipt has missing or unclassified fields")
    if receipt.get("schema_version") != RIGHTS_ADMISSION_SCHEMA_VERSION:
        raise PromotionError("rights admission receipt schema version is unsupported")
    if receipt.get("status") != "accepted":
        raise PromotionError("rights admission receipt is not accepted")

    expected_candidate = _receipt_identity(
        identity=identity,
        artifact_id=identity["artifact_id"],
        artifact_name=identity["artifact_name"],
        artifact_digest=identity["artifact_digest"],
    )
    if receipt.get("candidate") != expected_candidate:
        raise PromotionError("rights admission receipt is bound to a different candidate")

    sanitized = receipt.get("sanitized")
    if not isinstance(sanitized, dict) or set(sanitized) != {
        "policy_id",
        "policy_path",
        "policy_sha256",
        "schema_version",
        "source_sha",
        "tree_sha256",
    }:
        raise PromotionError("rights admission sanitized-candidate binding is incomplete")
    if sanitized.get("schema_version") != SANITIZED_CANDIDATE_SCHEMA:
        raise PromotionError("rights admission sanitized-candidate schema is unsupported")
    if sanitized.get("source_sha") != identity["source_sha"]:
        raise PromotionError("rights admission sanitized source SHA differs from candidate")
    if sanitized.get("policy_id") != RIGHTS_POLICY_ID:
        raise PromotionError("rights admission policy identity is unsupported")
    if sanitized.get("policy_path") != RIGHTS_POLICY_PATH:
        raise PromotionError("rights admission policy path is unsupported")
    policy_sha256 = sanitized.get("policy_sha256")
    if not isinstance(policy_sha256, str) or not SHA256_PATTERN.fullmatch(policy_sha256):
        raise PromotionError("rights admission policy digest is invalid")
    tree_sha256 = sanitized.get("tree_sha256")
    if not isinstance(tree_sha256, str) or not SHA256_PATTERN.fullmatch(tree_sha256):
        raise PromotionError("rights admission sanitized tree digest is invalid")

    strict_gate = receipt.get("strict_gate")
    if not isinstance(strict_gate, dict) or set(strict_gate) != {
        "command",
        "findings",
        "id",
        "policy_sha256",
        "source_sha",
        "status",
    }:
        raise PromotionError("rights admission strict-gate receipt is incomplete")
    command = strict_gate.get("command")
    if command != RIGHTS_GATE_COMMAND:
        raise PromotionError("rights admission strict gate command is not the canonical check")
    if strict_gate.get("id") != RIGHTS_GATE_ID or strict_gate.get("status") != "passed":
        raise PromotionError("rights admission strict gate did not pass")
    if strict_gate.get("source_sha") != identity["source_sha"]:
        raise PromotionError("rights admission strict gate source SHA differs from candidate")
    if strict_gate.get("policy_sha256") != policy_sha256:
        raise PromotionError("rights admission strict gate policy digest differs from binding")
    findings = strict_gate.get("findings")
    if isinstance(findings, bool) or not isinstance(findings, int) or findings != 0:
        raise PromotionError("rights admission strict gate reports unresolved findings")

    dependency_gate = receipt.get("supported_dependency_gate")
    if not isinstance(dependency_gate, dict) or set(dependency_gate) != {
        "candidate_manifest_sha256",
        "candidate_tree_sha256",
        "command",
        "id",
        "policy_path",
        "policy_sha256",
        "profile_manifest_path",
        "profile_manifest_sha256",
        "report_filename",
        "report_sha256",
        "schema_version",
        "source_sha",
        "status",
        "unresolved_count",
    }:
        raise PromotionError(
            "rights admission supported-dependency gate is missing or unclassified"
        )
    if dependency_gate.get("schema_version") != SUPPORTED_DEPENDENCY_SCHEMA_VERSION:
        raise PromotionError("supported-dependency report schema version is unsupported")
    if dependency_gate.get("id") != SUPPORTED_DEPENDENCY_GATE_ID:
        raise PromotionError("rights admission supported-dependency gate identity is unsupported")
    if dependency_gate.get("status") != "passed":
        raise PromotionError("rights admission supported-dependency gate did not pass")
    if dependency_gate.get("source_sha") != identity["source_sha"]:
        raise PromotionError(
            "rights admission supported-dependency source SHA differs from candidate"
        )
    if dependency_gate.get("candidate_manifest_sha256") != expected_candidate["manifest_sha256"]:
        raise PromotionError(
            "rights admission supported-dependency report is bound to a different manifest"
        )
    if dependency_gate.get("candidate_tree_sha256") != tree_sha256:
        raise PromotionError(
            "rights admission supported-dependency report is bound to a different source tree"
        )
    if dependency_gate.get("policy_path") != SUPPORTED_DEPENDENCY_POLICY_PATH:
        raise PromotionError("rights admission supported-dependency policy path is unsupported")
    if dependency_gate.get("profile_manifest_path") != SUPPORTED_DEPENDENCY_PROFILE_PATH:
        raise PromotionError("rights admission supported-dependency profile path is unsupported")
    for field in ("policy_sha256", "profile_manifest_sha256", "report_sha256"):
        value = dependency_gate.get(field)
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise PromotionError(f"rights admission supported-dependency {field} is invalid")
    if dependency_gate.get("report_filename") != SUPPORTED_DEPENDENCY_REPORT_NAME:
        raise PromotionError("rights admission supported-dependency report filename is unsupported")
    if dependency_gate.get("command") != SUPPORTED_DEPENDENCY_GATE_COMMAND:
        raise PromotionError(
            "rights admission supported-dependency command is not the canonical strict check"
        )
    unresolved_count = dependency_gate.get("unresolved_count")
    if (
        isinstance(unresolved_count, bool)
        or not isinstance(unresolved_count, int)
        or unresolved_count != 0
    ):
        raise PromotionError("rights admission supported-dependency gate reports unresolved rows")
    _validate_supported_dependency_report(
        path.parent / SUPPORTED_DEPENDENCY_REPORT_NAME,
        identity=identity,
        tree_sha256=tree_sha256,
        expected_gate=dependency_gate,
        repo_root=repo_root,
    )


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
        expected_workflow_run_attempt=args.candidate_run_attempt,
        expected_version=args.version,
    )
    identity["artifact_digest"] = args.candidate_artifact_digest
    identity["artifact_id"] = args.candidate_artifact_id
    identity["artifact_name"] = args.candidate_artifact_name
    _validate_rights_admission(args.rights_receipt, identity=identity, repo_root=Path.cwd())
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
        expected_run_attempt=args.run_attempt,
    )
    print(f"PASS: {args.kind} artifact metadata is bound to run {args.run_id}")


def _check_rights_workflow_run(args: argparse.Namespace) -> None:
    """Verify that the rights receipt came from the sanctioned successful workflow run."""

    metadata = _load_json(args.metadata, label="rights workflow-run metadata")
    run_attempt = _validate_workflow_run_metadata(
        metadata,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
        source_sha=args.source_sha,
        kind="rights",
    )
    print(f"PASS: sanctioned rights workflow run {args.run_id} attempt {run_attempt} succeeded")


def _check_workflow_run(args: argparse.Namespace) -> None:
    """Verify a candidate, rights, or promotion run using one closed contract."""

    metadata = _load_json(args.metadata, label=f"{args.kind} workflow-run metadata")
    run_attempt = _validate_workflow_run_metadata(
        metadata,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
        source_sha=args.source_sha,
        kind=args.kind,
    )
    print(
        f"PASS: sanctioned {args.kind} workflow run {args.run_id} attempt {run_attempt} succeeded"
    )


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
    _assert_receipt_candidate(
        receipt,
        identity,
        expected_channel=args.channel,
        expected_promotion_run_id=args.promotion_run_id,
        expected_promotion_run_attempt=args.promotion_run_attempt,
    )
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


def _write_index_verification_receipt(args: argparse.Namespace) -> None:
    """Record a byte-identical public-index download for a published candidate."""

    if args.channel != "pypi":
        raise PromotionError("production index verification is only accepted for PyPI")
    _manifest, identity = _load_and_bind_candidate(args)
    production_receipt = _load_receipt(args.production_receipt)
    _assert_receipt_candidate(
        production_receipt,
        identity,
        expected_channel="pypi",
        expected_promotion_run_id=args.production_run_id,
        expected_promotion_run_attempt=args.production_run_attempt,
    )
    _verify_index_artifacts(args)
    receipt = {
        "candidate": _receipt_identity(
            identity=identity,
            artifact_id=args.candidate_artifact_id,
            artifact_name=args.candidate_artifact_name,
            artifact_digest=args.candidate_artifact_digest,
        ),
        "channel": args.channel,
        "files": _published_members(identity),
        "index_url": PUBLIC_INDEX_URLS[args.channel],
        "production_receipt_sha256": _sha256(args.production_receipt),
        "schema_version": INDEX_VERIFICATION_SCHEMA_VERSION,
        "status": "passed",
    }
    _write_new_json(args.receipt, receipt, label="index verification receipt")
    print("PASS: wrote PyPI byte-verification receipt")


def _verify_index_verification_receipt(args: argparse.Namespace) -> None:
    """Verify a production receipt and its independent public-index proof."""

    if args.channel != "pypi":
        raise PromotionError("production index verification is only accepted for PyPI")
    _manifest, identity = _load_and_bind_candidate(args)
    production_receipt = _load_receipt(args.production_receipt)
    _assert_receipt_candidate(
        production_receipt,
        identity,
        expected_channel="pypi",
        expected_promotion_run_id=args.production_run_id,
        expected_promotion_run_attempt=args.production_run_attempt,
    )
    receipt = _load_json(args.receipt, label="index verification receipt")
    expected = {
        "candidate": _receipt_identity(
            identity=identity,
            artifact_id=args.candidate_artifact_id,
            artifact_name=args.candidate_artifact_name,
            artifact_digest=args.candidate_artifact_digest,
        ),
        "channel": "pypi",
        "files": _published_members(identity),
        "index_url": PUBLIC_INDEX_URLS["pypi"],
        "production_receipt_sha256": _sha256(args.production_receipt),
        "schema_version": INDEX_VERIFICATION_SCHEMA_VERSION,
        "status": "passed",
    }
    if receipt != expected:
        raise PromotionError("index verification receipt is not bound to the exact PyPI result")
    _verify_index_artifacts(args)
    print("PASS: PyPI index verification receipt is bound to exact candidate bytes")


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


def _parser() -> argparse.ArgumentParser:  # noqa: PLR0915 - one versioned CLI surface
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    artifact = subparsers.add_parser("check-artifact", help="verify GitHub artifact metadata")
    artifact.add_argument("--metadata", type=Path, required=True)
    artifact.add_argument("--artifact-id", required=True)
    artifact.add_argument("--artifact-name", required=True)
    artifact.add_argument("--artifact-digest", required=True)
    artifact.add_argument("--run-id", required=True)
    artifact.add_argument("--run-attempt", type=int)
    artifact.add_argument("--kind", choices=("candidate", "receipt", "rights"), required=True)
    artifact.add_argument("--source-sha")

    rights_run = subparsers.add_parser(
        "check-rights-run", help="verify the sanctioned rights-admission workflow run"
    )
    rights_run.add_argument("--metadata", type=Path, required=True)
    rights_run.add_argument("--run-id", required=True)
    rights_run.add_argument("--run-attempt", type=int)
    rights_run.add_argument("--source-sha", required=True)

    workflow_run = subparsers.add_parser(
        "check-workflow-run", help="verify a sanctioned candidate, rights, or promotion run"
    )
    workflow_run.add_argument("--metadata", type=Path, required=True)
    workflow_run.add_argument("--run-id", required=True)
    workflow_run.add_argument("--run-attempt", type=int, required=True)
    workflow_run.add_argument("--source-sha", required=True)
    workflow_run.add_argument("--kind", choices=("candidate", "rights", "promotion"), required=True)

    def add_candidate_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--candidate-dir", type=Path, required=True)
        command.add_argument(
            "--rights-receipt",
            type=Path,
            required=True,
            help="Independent rights-clean admission receipt",
        )
        command.add_argument("--source-sha", required=True)
        command.add_argument("--candidate-run-id", required=True)
        command.add_argument("--candidate-run-attempt", type=int, required=True)
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
    verify_receipt.add_argument("--promotion-run-id")
    verify_receipt.add_argument("--promotion-run-attempt", type=int)

    index = subparsers.add_parser(
        "verify-index-artifacts", help="verify public-index package bytes"
    )
    add_candidate_arguments(index)
    index.add_argument("--download-dir", type=Path, required=True)

    index_receipt = subparsers.add_parser(
        "write-index-verification-receipt",
        help="write a receipt for byte-identical production-index downloads",
    )
    add_candidate_arguments(index_receipt)
    index_receipt.add_argument("--channel", choices=("pypi",), required=True)
    index_receipt.add_argument("--production-receipt", type=Path, required=True)
    index_receipt.add_argument("--production-run-id", required=True)
    index_receipt.add_argument("--production-run-attempt", type=int, required=True)
    index_receipt.add_argument("--download-dir", type=Path, required=True)
    index_receipt.add_argument("--receipt", type=Path, required=True)

    verify_index_receipt = subparsers.add_parser(
        "verify-index-verification-receipt",
        help="verify a production-index byte-verification receipt",
    )
    add_candidate_arguments(verify_index_receipt)
    verify_index_receipt.add_argument("--channel", choices=("pypi",), required=True)
    verify_index_receipt.add_argument("--production-receipt", type=Path, required=True)
    verify_index_receipt.add_argument("--production-run-id", required=True)
    verify_index_receipt.add_argument("--production-run-attempt", type=int, required=True)
    verify_index_receipt.add_argument("--download-dir", type=Path, required=True)
    verify_index_receipt.add_argument("--receipt", type=Path, required=True)

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


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - bounded CLI dispatch
    """Run one fail-closed promotion verification or receipt command."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "check-artifact":
            _check_artifact(args)
        elif args.command == "check-rights-run":
            _check_rights_workflow_run(args)
        elif args.command == "check-workflow-run":
            _check_workflow_run(args)
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
        elif args.command == "write-index-verification-receipt":
            _write_index_verification_receipt(args)
        elif args.command == "verify-index-verification-receipt":
            _verify_index_verification_receipt(args)
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
