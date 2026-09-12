"""Convert a task-owned fixture tree into a verified preservation manifest.

This dependency-light walkthrough materializes a synthetic tree, uses existing ``chunk_manifest.v1``
and check-only ``cleanup_eligibility_record.v1`` owners, and verifies a logical second failure-domain
copy. The fixture-local marker is not a worktree lease or durable receipt.

Usage::

    uv run python examples/advanced/41_artifact_preservation.py --json

Use fresh roots. The example never deletes existing contents or uploads bytes, and output is not
benchmark evidence; failed runs leave partial fixtures for inspection.

References: ``docs/context/cleanup_eligibility.md`` and the canonical manifest/cleanup owners.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools import chunk_manifest  # noqa: E402
from scripts.validation import check_cleanup_eligibility  # noqa: E402

DEFAULT_FIXTURE = REPO_ROOT / "examples" / "fixtures" / "artifact_preservation" / "v1"
DEFAULT_OUTPUT = Path("output/example-fixtures/41_artifact_preservation")
MARKER_NAME = ".task-owner.json"
MANIFEST_NAME = "preservation_manifest.json"
REPORT_NAME = "preservation_walkthrough.json"
REPORT_SCHEMA = "artifact_preservation_walkthrough.v1"
REQUIRED, OPTIONAL, TRANSIENT, EXCLUDED, FORBIDDEN = (
    "required",
    "optional_diagnostic",
    "transient",
    "excluded",
    "forbidden",
)
INCLUDED_ROLES = frozenset({REQUIRED, OPTIONAL})
EXCLUDED_ROLES = frozenset({TRANSIENT, EXCLUDED})
CREDENTIAL_LIKE_RE = re.compile(
    r"(?i)(?:api[_-]?key|secret|password|passwd|token|credential|bearer)"
)


class PreservationExampleError(ValueError):
    """Fail-closed fixture or preservation error with a stable code."""

    def __init__(self, code: str, message: str, *, path: str | None = None) -> None:
        """Store a stable failure code and optional relative member path."""
        super().__init__(message)
        self.code, self.path = code, path

    def to_dict(self) -> dict[str, Any]:
        """Return the error without exposing local absolute paths."""
        result: dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.path is not None:
            result["path"] = self.path
        return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_fixture(path: Path) -> dict[str, Any]:  # noqa: C901
    fixture = Path(path).expanduser()
    if fixture.is_dir():
        fixture /= "fixture.json"
    try:
        payload = json.loads(fixture.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreservationExampleError("fixture_unreadable", f"cannot read fixture: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != "artifact_preservation_fixture.v1":
        raise PreservationExampleError(
            "fixture_schema", "unsupported artifact-preservation fixture"
        )
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict) or not artifact.get("artifact_id"):
        raise PreservationExampleError("fixture_schema", "fixture artifact identity is incomplete")
    patterns, raw_members = payload["exclude_patterns"], payload["members"]
    paths: set[str] = set()
    semantic_ids: set[str] = set()
    members: list[dict[str, Any]] = []
    for raw in raw_members:
        if not isinstance(raw, Mapping):
            raise PreservationExampleError("fixture_schema", "fixture member must be an object")
        relative, role = (
            chunk_manifest.normalize_relative_path(str(raw.get("path", ""))),
            raw.get("role"),
        )
        if (
            role not in INCLUDED_ROLES | EXCLUDED_ROLES
            or relative in paths
            or relative == MARKER_NAME
        ):
            raise PreservationExampleError(
                "fixture_schema", f"invalid or duplicate member: {relative}", path=relative
            )
        paths.add(relative)
        if role in INCLUDED_ROLES:
            semantic_id = raw.get("semantic_id")
            if not semantic_id:
                raise PreservationExampleError(
                    "fixture_schema", f"semantic id is missing: {relative}", path=relative
                )
            if semantic_id in semantic_ids:
                raise PreservationExampleError(
                    "duplicate_semantic_id", f"duplicate semantic id: {semantic_id}", path=relative
                )
            semantic_ids.add(semantic_id)
        members.append({**dict(raw), "path": relative, "materialize": True})
    normalized_forbidden = [
        {"path": chunk_manifest.normalize_relative_path(str(item["path"])), "role": FORBIDDEN}
        for item in payload.get("forbidden", [])
        if isinstance(item, Mapping) and item.get("role") == FORBIDDEN
    ]
    return {
        **payload,
        "artifact": artifact,
        "exclude_patterns": patterns,
        "members": members,
        "forbidden": normalized_forbidden,
    }


def _member_bytes(member: Mapping[str, Any]) -> bytes:
    if "content" in member:
        return str(member["content"]).encode("utf-8")
    try:
        return base64.b64decode(str(member["content_base64"]), validate=True)
    except (ValueError, TypeError) as exc:
        raise PreservationExampleError(
            "fixture_schema", f"invalid binary content: {member['path']}"
        ) from exc


def _fresh(path: Path, label: str) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise PreservationExampleError("not_fresh", f"{label} is not a directory")
    if path.exists() and any(path.iterdir()):
        raise PreservationExampleError("not_fresh", f"{label} must be empty")
    path.mkdir(parents=True, exist_ok=True)


def _target(root: Path, relative: str) -> Path:
    target = root.joinpath(*PurePosixPath(relative).parts)
    try:
        target.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise PreservationExampleError(
            "path_escape", f"member escapes root: {relative}", path=relative
        ) from exc
    return target


def _marker(spec: Mapping[str, Any]) -> dict[str, str]:
    artifact = spec["artifact"]
    return {
        "artifact_id": artifact["artifact_id"],
        "artifact_version": artifact["artifact_version"],
        "output_owner": artifact["owner_id"],
        "output_owner_state": "closed",
        "root_identity": artifact["root_identity"],
        "retention_class": artifact["retention_class"],
        "private_projection": "public_safe",
    }


def _owned(root: Path, spec: Mapping[str, Any]) -> None:
    if root.is_symlink() or not root.is_dir():
        raise PreservationExampleError("unowned_root", "output root must be an existing directory")
    marker = root / MARKER_NAME
    if marker.is_symlink() or not marker.is_file():
        raise PreservationExampleError("unowned_root", "output root has no ownership marker")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreservationExampleError(
            "unowned_root", f"ownership marker is unreadable: {exc}"
        ) from exc
    if payload != _marker(spec):
        raise PreservationExampleError(
            "ownership_mismatch", "ownership marker does not match fixture identity"
        )


def _safe_members(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        text = (
            ""
            if path.is_symlink() or not path.is_file()
            else path.read_bytes().decode("utf-8", errors="ignore")
        )
        if CREDENTIAL_LIKE_RE.search(relative) or CREDENTIAL_LIKE_RE.search(text):
            raise PreservationExampleError(
                "credential_like_member", "credential-like content rejected", path=relative
            )


def _materialize(spec: Mapping[str, Any], root: Path) -> None:
    _fresh(root, "fixture root")
    for member in spec["members"]:
        data, relative = _member_bytes(member), member["path"]
        target = _target(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    _write_json(root / MARKER_NAME, _marker(spec))


def prepare_fixture_tree(fixture: Path, root: Path) -> dict[str, Any]:
    """Materialize a validated fixture recipe for inspection or focused tests."""
    spec = _load_fixture(Path(fixture))
    _materialize(spec, Path(root))
    return spec


def build_fixture_manifest(root: Path, fixture: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    """Build the canonical versioned manifest after ownership and safety checks."""
    spec = _load_fixture(Path(fixture))
    _owned(Path(root), spec)
    _safe_members(Path(root))
    manifest = chunk_manifest.build_manifest(
        Path(root),
        artifact=chunk_manifest.ArtifactIdentity(
            spec["artifact"]["artifact_id"],
            spec["artifact"]["artifact_version"],
            spec["artifact"]["root_identity"],
            spec["artifact"]["retention_role"],
        ),
        policy=chunk_manifest.ChunkingPolicy(exclude_patterns=tuple(spec["exclude_patterns"])),
        workers=1,
    )
    expected = {member["path"] for member in spec["members"] if member["role"] in INCLUDED_ROLES}
    actual = {record["path"] for record in manifest["files"]}
    if missing := sorted(expected - actual):
        required = {member["path"] for member in spec["members"] if member["role"] == REQUIRED}
        raise PreservationExampleError(
            "missing_required_member" if missing[0] in required else "missing_member",
            "manifest is missing declared members",
            path=missing[0],
        )
    if unexpected := sorted(actual - expected):
        raise PreservationExampleError(
            "unexpected_member", "manifest contains an undeclared member", path=unexpected[0]
        )
    excluded = {entry["path"] for entry in manifest["excluded"]}
    wanted = {MARKER_NAME} | {
        member["path"] for member in spec["members"] if member["role"] in EXCLUDED_ROLES
    }
    if missing := sorted(wanted - excluded):
        raise PreservationExampleError(
            "exclusion_missing", "declared exclusion was not recorded", path=missing[0]
        )
    return manifest


def _copy_members(source: Path, destination: Path, manifest: Mapping[str, Any]) -> None:
    for record in manifest["files"]:
        relative = record["path"]
        target = _target(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_target(source, relative), target)
    shutil.copyfile(source / MARKER_NAME, destination / MARKER_NAME)


def _copy_record(copy_id: str, storage_class: str, domain: str, digest: str) -> dict[str, str]:
    return {
        "copy_id": copy_id,
        "storage_class": storage_class,
        "failure_domain": domain,
        "byte_digest": digest,
        "verification": "verified",
        "verification_basis": "manifest_verification",
    }


def _cleanup(
    out: Path, phase: str, spec: Mapping[str, Any], digest: str, copies: list[dict[str, str]]
) -> dict[str, Any]:
    artifact = spec["artifact"]
    record = {
        "schema": check_cleanup_eligibility.SCHEMA,
        "artifacts": [
            {
                "artifact_id": artifact["artifact_id"],
                "semantic_digest": digest,
                "retention_class": artifact["retention_class"],
                "output_owner": artifact["owner_id"],
                "output_owner_state": "closed",
                "private_projection": "public_safe",
                "retention_hold": False,
                "copies": copies,
                "writers": [],
                "consumers": [],
            }
        ],
    }
    path = out / f"cleanup_{phase}.json"
    _write_json(path, record)
    return check_cleanup_eligibility.check_artifact_file(path, artifact["artifact_id"]).to_dict()


def _layout(out: Path, destination: Path | None) -> tuple[Path, Path, Path]:
    output, source = Path(out).expanduser().resolve(), Path(out).expanduser().resolve() / "source"
    target = (
        output / "destination" if destination is None else Path(destination).expanduser().resolve()
    )
    source_resolved, target_resolved = source.resolve(strict=False), target.resolve(strict=False)
    if source_resolved == target_resolved or any(
        child.is_relative_to(parent)
        for parent, child in (
            (source_resolved, target_resolved),
            (target_resolved, source_resolved),
        )
    ):
        raise PreservationExampleError(
            "root_overlap", "source and destination roots must be disjoint"
        )
    return output, source, target


def run_preservation(
    fixture: Path = DEFAULT_FIXTURE,
    *,
    out_dir: Path = DEFAULT_OUTPUT,
    destination: Path | None = None,
) -> dict[str, Any]:
    """Run the local preservation walkthrough and return its diagnostic report."""
    spec = _load_fixture(Path(fixture))
    output, source, target = _layout(Path(out_dir), destination)
    _fresh(output, "walkthrough output")
    _fresh(source, "source root")
    _fresh(target, "destination root")
    _materialize(spec, source)
    manifest = build_fixture_manifest(source, Path(fixture))
    _write_json(output / MANIFEST_NAME, manifest)
    source_result = chunk_manifest.verify_manifest(source, manifest=manifest)
    if source_result["status"] != "ok":
        raise PreservationExampleError("source_verification_failed", "source verification failed")
    source_copy = _copy_record(
        "fixture-source-copy", "local_scratch", "fixture-source-domain", manifest["manifest_id"]
    )
    before = _cleanup(output, "before_destination", spec, manifest["manifest_id"], [source_copy])
    if before["outcome"] == "eligible":
        raise PreservationExampleError(
            "cleanup_guard_regression", "local-only copy became eligible"
        )
    _copy_members(source, target, manifest)
    _owned(target, spec)
    destination_result = chunk_manifest.verify_manifest(target, manifest=manifest)
    if destination_result["status"] != "ok":
        raise PreservationExampleError(
            "destination_verification_failed", "destination verification failed"
        )
    destination_copy = _copy_record(
        "fixture-destination-copy",
        "personal_durable",
        "fixture-destination-domain",
        manifest["manifest_id"],
    )
    after = _cleanup(
        output, "after_destination", spec, manifest["manifest_id"], [source_copy, destination_copy]
    )
    if after["outcome"] != "eligible":
        raise PreservationExampleError(
            "cleanup_guard_failed", "verified destination did not become eligible"
        )
    artifact = spec["artifact"]
    identity = {
        key: artifact[key]
        for key in (
            "artifact_id",
            "artifact_version",
            "root_identity",
            "source_id",
            "config_id",
            "campaign_id",
        )
    }
    return {
        "schema": REPORT_SCHEMA,
        "status": "verified",
        "manifest": {
            "schema_version": manifest["schema_version"],
            "manifest_id": manifest["manifest_id"],
            "tree_sha256": manifest["tree_sha256"],
            "member_count": len(manifest["files"]),
            "excluded_member_count": len(manifest["excluded"]),
            "byte_count": sum(record["size_bytes"] for record in manifest["files"]),
            "retention_class": artifact["retention_class"],
        },
        "identity": identity,
        "ownership": {
            "marker": MARKER_NAME,
            "owner_id": artifact["owner_id"],
            "verified": True,
        },
        "roles": {
            member["path"]: member["role"] for member in [*spec["members"], *spec["forbidden"]]
        },
        "verification": {
            "source": source_result["status"],
            "destination": destination_result["status"],
        },
        "copies": [
            {"copy_id": copy["copy_id"], "failure_domain": copy["failure_domain"]}
            for copy in (source_copy, destination_copy)
        ],
        "cleanup_eligibility": {
            "before_destination_verification": before,
            "after_destination_verification": after,
        },
        "boundary": {
            "upload": "not_performed",
            "delete": "not_performed",
            "evidence": "not_evidence",
            "custody": "not_asserted",
            "locality": "logical_fixture_only",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fixture",
        nargs="?",
        const=DEFAULT_FIXTURE,
        default=DEFAULT_FIXTURE,
        type=Path,
        help="Fixture recipe directory or fixture.json path.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUTPUT, help="Fresh walkthrough output root."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
        help="Fresh second fixture root; defaults to <out-dir>/destination.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full walkthrough report as JSON."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the example and return a shell-friendly status."""
    args = _parser().parse_args(argv)
    try:
        report = run_preservation(args.fixture, out_dir=args.out_dir, destination=args.destination)
    except (PreservationExampleError, chunk_manifest.ChunkManifestError) as exc:
        payload = {"status": "failed", "error": exc.to_dict()}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"artifact-preservation: {payload['error']['code']}: {exc}", file=sys.stderr)
        return 2
    _write_json(Path(args.out_dir).expanduser().resolve() / REPORT_NAME, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"status={report['status']} manifest_id={report['manifest']['manifest_id']} cleanup_after={report['cleanup_eligibility']['after_destination_verification']['outcome']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
