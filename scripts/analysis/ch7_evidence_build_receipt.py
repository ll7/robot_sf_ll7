"""Build and verify a non-admission receipt for the Chapter 7 v2 package.

The receipt binds the reproducible package build inputs and environment.  It is
deliberately separate from the package and from the future admission receipt:
adding or checking it cannot change package bytes or authorize evidence.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from scripts.analysis import build_ch7_evidence_package_v2 as package_builder
from scripts.analysis import verify_ch7_evidence_admission as package_admission
from scripts.analysis import verify_ch7_evidence_admission_v2 as package_diagnostic

# evidence-writer-exempt: this receipt is a hash-bound canonical JSON artifact;
# shared write_json would change its byte representation and self-hash. Its same-PR
# review sidecar carries the required AI-GENERATED NEEDS-REVIEW marker.

RECEIPT_SCHEMA_VERSION = "ch7-evidence-build-receipt.v1"
ISSUE = 7410
PACKAGE_ISSUE = 7087
DEFAULT_RECEIPT = Path("docs/context/evidence/issue_7322_ch7_build_receipt.v1.json")
PACKAGE_PATH = Path("docs/context/evidence/issue_7322_ch7_evidence_package_v2")
SOURCE_PACKAGE_PATH = Path("docs/context/evidence/issue_6792_ch7_evidence_package_v1")
CONFIG_PATH = Path("configs/analysis/ch7_evidence_package.v2.yaml")
PORTFOLIO_PATH = Path("configs/analysis/ch7_worked_example_portfolio.v2.yaml")
PACKAGE_BUILDER_PATH = Path("scripts/analysis/build_ch7_evidence_package_v2.py")
PACKAGE_VERIFIER_PATH = Path("scripts/analysis/verify_ch7_evidence_admission_v2.py")
PACKAGE_SCHEMA_PATH = Path("robot_sf/benchmark/schemas/ch7-evidence-package.v2.json")
ADMISSION_SCHEMA_PATH = Path("robot_sf/benchmark/schemas/ch7-evidence-admission.v2.json")
PYPROJECT_PATH = Path("pyproject.toml")
LOCK_PATH = Path("uv.lock")
V1_MANIFEST_PATH = SOURCE_PACKAGE_PATH / "manifest.json"
V1_SUMS_PATH = SOURCE_PACKAGE_PATH / "SHA256SUMS"
V1_AUDIT_PATH = SOURCE_PACKAGE_PATH / package_builder.SOURCE_AUDIT_MEMBER
V1_REDUCED_ATLAS_PATH = SOURCE_PACKAGE_PATH / package_builder.SOURCE_REDUCED_ATLAS_MEMBER

TREE_HASH_ALGORITHM = "payload-tree-sha256.v1"
CANONICALIZATION = "json-sort-keys-utf8-newline.v1"
INTEGRITY_SCOPE = "top-level-object-with-integrity.receipt_payload_sha256-removed.v1"
CLAIM_BOUNDARY = (
    "Build provenance only: this receipt records reproducibility inputs and checks for the "
    "collision-excluded Chapter 7 v2 package. It is not an admission receipt, domain approval, "
    "publication authorization, benchmark result, or paper-facing evidence."
)
PYTHON_COMMAND = ["python", "--version"]
UV_COMMAND = ["uv", "--version"]
UV_EXPORT_COMMAND = [
    "uv",
    "export",
    "--frozen",
    "--no-dev",
    "--no-emit-project",
    "--format",
    "requirements-txt",
]


class Ch7EvidenceBuildReceiptError(ValueError):
    """Raised when build provenance cannot be verified fail-closed."""


def canonical_bytes(payload: Any) -> bytes:
    """Serialize a receipt payload with the repository's deterministic JSON contract."""

    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    """Return a SHA-256 digest for bytes."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Ch7EvidenceBuildReceiptError(f"cannot read {path}") from exc
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    """Read a JSON object and convert parse failures to the receipt error type."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Ch7EvidenceBuildReceiptError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise Ch7EvidenceBuildReceiptError(f"{label} must be a JSON object: {path}")
    return dict(payload)


def _repo_path(repository: Path, relative: Path | str) -> Path:
    """Resolve a repository-relative path without allowing path escape."""

    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise Ch7EvidenceBuildReceiptError(f"path is not repository-relative: {path}")
    return repository / path


def _run(
    argv: Sequence[str], *, cwd: Path, label: str, text: bool = True
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    """Run a deterministic provenance command and retain its failure boundary."""

    result = subprocess.run(
        list(argv),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=text,
    )
    if result.returncode != 0:
        stderr = (
            result.stderr.decode("utf-8", "replace")
            if isinstance(result.stderr, bytes)
            else result.stderr
        )
        raise Ch7EvidenceBuildReceiptError(
            f"{label} failed with exit code {result.returncode}: {stderr.strip()}"
        )
    return result


def _git_text(repository: Path, *argv: str) -> str:
    result = _run(["git", "-C", str(repository), *argv], cwd=repository, label="git command")
    assert isinstance(result.stdout, str)
    return result.stdout.strip()


def _git_blob(repository: Path, commit: str, relative: Path) -> bytes:
    result = _run(
        ["git", "-C", str(repository), "cat-file", "blob", f"{commit}:{relative.as_posix()}"],
        cwd=repository,
        label=f"git source lookup for {relative}",
        text=False,
    )
    assert isinstance(result.stdout, bytes)
    return result.stdout


def _git_commit_tree(repository: Path, commit: str) -> tuple[str, str]:
    resolved = _git_text(repository, "rev-parse", f"{commit}^{{commit}}")
    tree = _git_text(repository, "rev-parse", f"{resolved}^{{tree}}")
    if len(resolved) != 40 or len(tree) != 40:
        raise Ch7EvidenceBuildReceiptError("source commit/tree is not a full Git object id")
    return resolved, tree


def _source_digest(repository: Path, commit: str, relative: Path) -> str:
    return sha256_bytes(_git_blob(repository, commit, relative))


def _current_matches_source(repository: Path, commit: str, relative: Path) -> None:
    current = _repo_path(repository, relative)
    if not current.is_file():
        raise Ch7EvidenceBuildReceiptError(f"source file is missing from the checkout: {relative}")
    expected = _source_digest(repository, commit, relative)
    actual = sha256_file(current)
    if actual != expected:
        raise Ch7EvidenceBuildReceiptError(
            f"working-tree source differs from pinned commit for {relative}"
        )


def _validate_schema(payload: Mapping[str, Any], schema_path: Path, label: str) -> None:
    schema = read_json(schema_path, f"{label} schema")
    try:
        errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=str)
    except (TypeError, ValidationError) as exc:
        raise Ch7EvidenceBuildReceiptError(f"{label} schema is invalid") from exc
    if errors:
        details = "; ".join(error.message for error in errors[:3])
        raise Ch7EvidenceBuildReceiptError(f"{label} validation failed: {details}")


def payload_tree_sha256(root: Path) -> str:
    """Hash SHA256SUMS-listed payload members and the checksum manifest.

    Durable evidence packages also carry review-only sidecars that are not
    package payload.  Hashing the listed member digests makes a generated
    package and its sidecar-bearing tracked copy comparable without weakening
    the package checksum verification.
    """

    try:
        entries = package_admission._parse_sums(root / "SHA256SUMS")
    except (OSError, package_admission.Ch7EvidenceAdmissionError) as exc:
        raise Ch7EvidenceBuildReceiptError(f"cannot parse package checksums: {root}") from exc
    files: list[dict[str, str]] = []
    for _expected, relative in sorted(entries, key=lambda item: item[1]):
        member = root / relative
        files.append({"path": relative, "sha256": sha256_file(member)})
    projection = {
        "algorithm": TREE_HASH_ALGORITHM,
        "files": files,
        "sha256sums_sha256": sha256_file(root / "SHA256SUMS"),
    }
    return sha256_bytes(canonical_bytes(projection))


def _verify_package(root: Path, *, review_sidecars: bool) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        sums_sha, _listed = package_admission._verify_members(
            root,
            label="Chapter 7 v2 package",
            require_review_sidecars=review_sidecars,
        )
    except package_admission.Ch7EvidenceAdmissionError as exc:
        raise Ch7EvidenceBuildReceiptError(f"package checksum verification failed: {exc}") from exc
    manifest = read_json(root / "manifest.json", "Chapter 7 v2 manifest")
    return manifest, {
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "sha256sums_sha256": sums_sha,
        "tree_sha256": payload_tree_sha256(root),
    }


def _environment(repository: Path, source_files: Mapping[str, str]) -> dict[str, Any]:
    uv_result = _run(UV_COMMAND, cwd=repository, label="uv version")
    export_result = _run(UV_EXPORT_COMMAND, cwd=repository, label="uv dependency export")
    assert isinstance(uv_result.stdout, str)
    assert isinstance(export_result.stdout, str)
    export_bytes = export_result.stdout.encode("utf-8")
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "command": list(PYTHON_COMMAND),
        },
        "uv": {"version": uv_result.stdout.strip(), "command": list(UV_COMMAND)},
        "project": {
            "path": PYPROJECT_PATH.as_posix(),
            "sha256": source_files[PYPROJECT_PATH.as_posix()],
        },
        "lock": {"path": LOCK_PATH.as_posix(), "sha256": source_files[LOCK_PATH.as_posix()]},
        "resolved_dependencies": {
            "command": list(UV_EXPORT_COMMAND),
            "sha256": sha256_bytes(export_bytes),
            "line_count": len(export_result.stdout.splitlines()),
        },
    }


def _expected_commands() -> dict[str, dict[str, Any]]:
    build = [
        "uv",
        "run",
        "python",
        PACKAGE_BUILDER_PATH.as_posix(),
        "--source-package",
        SOURCE_PACKAGE_PATH.as_posix(),
        "--config",
        CONFIG_PATH.as_posix(),
        "--output",
        "<fresh-output>",
    ]
    return {
        "build": {"argv": build, "cwd": "."},
        "build_determinism": {"argv": [*build, "--check-determinism"], "cwd": "."},
        "verify_check_only": {
            "argv": [
                "uv",
                "run",
                "python",
                PACKAGE_VERIFIER_PATH.as_posix(),
                "--package",
                "<fresh-output>",
                "--check-only",
            ],
            "cwd": ".",
        },
    }


def _build_independent(
    *, repository: Path, source_package: Path, config: Path, label: str
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix=f"ch7-v2-receipt-{label}-") as root:
        output = Path(root) / "package"
        manifest = package_builder.build_ch7_evidence_package_v2(
            source_package=source_package,
            output=output,
            config_path=config,
            check_determinism=False,
        )
        checked_manifest, hashes = _verify_package(output, review_sidecars=False)
        if checked_manifest != manifest:
            raise Ch7EvidenceBuildReceiptError(f"{label} manifest changed after validation")
        diagnostic = package_diagnostic.diagnose_v2_package(output)
        return manifest, hashes, diagnostic


def _diagnose_payload_copy(package: Path) -> dict[str, Any]:
    """Run the existing check-only verifier after omitting review-only sidecars."""

    with tempfile.TemporaryDirectory(prefix="ch7-v2-receipt-check-") as root:
        output = Path(root) / "package"
        output.mkdir()
        try:
            entries = package_admission._parse_sums(package / "SHA256SUMS")
        except (OSError, package_admission.Ch7EvidenceAdmissionError) as exc:
            raise Ch7EvidenceBuildReceiptError("cannot prepare check-only package copy") from exc
        for _expected, relative in entries:
            member = package / relative
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(member, target)
        shutil.copyfile(package / "SHA256SUMS", output / "SHA256SUMS")
        return package_diagnostic.diagnose_v2_package(output)


def _source_file_paths() -> tuple[tuple[str, Path], ...]:
    return (
        ("builder", PACKAGE_BUILDER_PATH),
        ("verifier", PACKAGE_VERIFIER_PATH),
        ("package_schema", PACKAGE_SCHEMA_PATH),
        ("admission_schema", ADMISSION_SCHEMA_PATH),
        ("config", CONFIG_PATH),
        ("portfolio", PORTFOLIO_PATH),
        ("project", PYPROJECT_PATH),
        ("lock", LOCK_PATH),
        ("v1_manifest", V1_MANIFEST_PATH),
        ("v1_sha256sums", V1_SUMS_PATH),
        ("v1_audit", V1_AUDIT_PATH),
        ("v1_reduced_atlas", V1_REDUCED_ATLAS_PATH),
    )


def _source_records(repository: Path, commit: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    records: list[dict[str, str]] = []
    by_path: dict[str, str] = {}
    for role, relative in _source_file_paths():
        path = relative.as_posix()
        digest = _source_digest(repository, commit, relative)
        records.append({"role": role, "path": path, "sha256": digest})
        by_path[path] = digest
    return records, by_path


def _receipt_payload_hash(receipt: Mapping[str, Any]) -> str:
    projection = json.loads(json.dumps(receipt))
    integrity = projection.get("integrity")
    if not isinstance(integrity, dict) or "receipt_payload_sha256" not in integrity:
        raise Ch7EvidenceBuildReceiptError("receipt integrity field is missing")
    del integrity["receipt_payload_sha256"]
    return sha256_bytes(canonical_bytes(projection))


def build_receipt(
    *,
    repository: Path,
    output: Path = DEFAULT_RECEIPT,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Build a durable receipt after two independent package builds."""

    repository = repository.resolve()
    resolved_commit, tree = _git_commit_tree(repository, source_commit or "HEAD")
    source_records, source_files = _source_records(repository, resolved_commit)
    for _role, relative in _source_file_paths():
        _current_matches_source(repository, resolved_commit, relative)

    source_package = _repo_path(repository, SOURCE_PACKAGE_PATH)
    config = _repo_path(repository, CONFIG_PATH)
    package = _repo_path(repository, PACKAGE_PATH)
    if not source_package.is_dir() or not config.is_file() or not package.is_dir():
        raise Ch7EvidenceBuildReceiptError("Chapter 7 v2 build inputs are missing")

    source = package_builder.verify_v1_source_package(source_package)
    first_manifest, first_hashes, first_diagnostic = _build_independent(
        repository=repository, source_package=source_package, config=config, label="a"
    )
    second_manifest, second_hashes, second_diagnostic = _build_independent(
        repository=repository, source_package=source_package, config=config, label="b"
    )
    if (
        first_manifest != second_manifest
        or first_hashes != second_hashes
        or first_diagnostic != second_diagnostic
    ):
        raise Ch7EvidenceBuildReceiptError("independent Chapter 7 v2 builds differ")

    tracked_manifest, tracked_hashes = _verify_package(package, review_sidecars=True)
    if tracked_manifest != first_manifest:
        raise Ch7EvidenceBuildReceiptError("tracked package manifest differs from rebuilt package")
    if tracked_hashes != first_hashes:
        raise Ch7EvidenceBuildReceiptError("tracked package payload differs from rebuilt package")
    diagnostic = first_diagnostic
    if diagnostic["diagnostics"]["receipt_created"] is not False:
        raise Ch7EvidenceBuildReceiptError("package diagnostic crossed the admission boundary")

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "issue": ISSUE,
        "package_issue": PACKAGE_ISSUE,
        "status": "verified_build_provenance",
        "claim_boundary": CLAIM_BOUNDARY,
        "repository": {
            "commit": resolved_commit,
            "tree": tree,
            "source_tree_contract": "git-commit-tree.v1",
            "receipt_excluded_from_source_commit": True,
        },
        "source": {
            "builder": next(record for record in source_records if record["role"] == "builder"),
            "verifier": next(record for record in source_records if record["role"] == "verifier"),
            "package_schema": next(
                record for record in source_records if record["role"] == "package_schema"
            ),
            "admission_schema": next(
                record for record in source_records if record["role"] == "admission_schema"
            ),
            "files": source_records,
        },
        "inputs": {
            "source_package": {
                "path": SOURCE_PACKAGE_PATH.as_posix(),
                "sha256sums_sha256": source["package_sha256sums_sha256"],
                "manifest_sha256": source["manifest_sha256"],
                "members": [
                    {"path": source["audit_member"], "sha256": source["audit_member_sha256"]},
                    {
                        "path": source["reduced_atlas_member"],
                        "sha256": source["reduced_atlas_member_sha256"],
                    },
                ],
            },
            "config": {
                "path": CONFIG_PATH.as_posix(),
                "sha256": source_files[CONFIG_PATH.as_posix()],
            },
            "portfolio": {
                "path": PORTFOLIO_PATH.as_posix(),
                "sha256": source_files[PORTFOLIO_PATH.as_posix()],
            },
        },
        "environment": _environment(repository, source_files),
        "commands": _expected_commands(),
        "package": {
            "path": PACKAGE_PATH.as_posix(),
            "manifest_sha256": tracked_hashes["manifest_sha256"],
            "sha256sums_sha256": tracked_hashes["sha256sums_sha256"],
            "tree_sha256": tracked_hashes["tree_sha256"],
            "tree_hash_algorithm": TREE_HASH_ALGORITHM,
            "status": tracked_manifest["status"],
            "admission_status": tracked_manifest["admission_status"],
        },
        "rebuilds": {
            "tree_hash_algorithm": TREE_HASH_ALGORITHM,
            "byte_identical": True,
            "matches_tracked_package": True,
            "build_a": first_hashes,
            "build_b": second_hashes,
        },
        "check_only": {
            "command": _expected_commands()["verify_check_only"],
            "exit_code": 0,
            "status": diagnostic["status"],
            "admission_status": diagnostic["admission_status"],
            "diagnostic": diagnostic,
        },
        "admission_boundary": {
            "admission_receipt_created": False,
            "domain_approval_recorded": False,
            "publication_authorized": False,
            "benchmark_result_admitted": False,
        },
        "integrity": {
            "canonicalization": CANONICALIZATION,
            "scope": INTEGRITY_SCOPE,
            "receipt_payload_sha256": "0" * 64,
        },
    }
    schema_path = _repo_path(
        repository, Path("robot_sf/benchmark/schemas/ch7-evidence-build-receipt.v1.json")
    )
    _validate_schema(receipt, schema_path, "build receipt")
    receipt["integrity"]["receipt_payload_sha256"] = _receipt_payload_hash(receipt)
    _validate_schema(receipt, schema_path, "build receipt")
    output = output if output.is_absolute() else repository / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(receipt))
    return receipt


def verify_receipt(*, repository: Path, receipt_path: Path, rebuild: bool = True) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    """Verify every recorded input, package hash, diagnostic, and optional rebuild."""

    repository = repository.resolve()
    receipt_path = receipt_path if receipt_path.is_absolute() else repository / receipt_path
    raw = receipt_path.read_bytes()
    receipt = read_json(receipt_path, "build receipt")
    schema_path = _repo_path(
        repository, Path("robot_sf/benchmark/schemas/ch7-evidence-build-receipt.v1.json")
    )
    _validate_schema(receipt, schema_path, "build receipt")
    if raw != canonical_bytes(receipt):
        raise Ch7EvidenceBuildReceiptError("build receipt is not canonically serialized")
    if receipt["integrity"]["receipt_payload_sha256"] != _receipt_payload_hash(receipt):
        raise Ch7EvidenceBuildReceiptError("build receipt payload hash mismatch")
    if receipt["claim_boundary"] != CLAIM_BOUNDARY:
        raise Ch7EvidenceBuildReceiptError("build receipt claim boundary changed")
    if receipt["commands"] != _expected_commands():
        raise Ch7EvidenceBuildReceiptError("canonical reproduction commands changed")

    commit, tree = _git_commit_tree(repository, receipt["repository"]["commit"])
    if commit != receipt["repository"]["commit"] or tree != receipt["repository"]["tree"]:
        raise Ch7EvidenceBuildReceiptError("pinned repository commit/tree is unavailable")
    source_files = receipt["source"]["files"]
    by_path: dict[str, str] = {}
    for record in source_files:
        path = Path(record["path"])
        digest = _source_digest(repository, commit, path)
        if digest != record["sha256"]:
            raise Ch7EvidenceBuildReceiptError(f"pinned source hash mismatch: {path}")
        _current_matches_source(repository, commit, path)
        by_path[path.as_posix()] = digest
    for role in ("builder", "verifier", "package_schema", "admission_schema"):
        record = receipt["source"][role]
        if by_path.get(record["path"]) != record["sha256"]:
            raise Ch7EvidenceBuildReceiptError(f"source role is not bound: {role}")

    environment = _environment(repository, by_path)
    if environment != receipt["environment"]:
        raise Ch7EvidenceBuildReceiptError("Python, uv, lock, or dependency identity changed")

    source_package = _repo_path(repository, SOURCE_PACKAGE_PATH)
    config = _repo_path(repository, CONFIG_PATH)
    package = _repo_path(repository, Path(receipt["package"]["path"]))
    source = package_builder.verify_v1_source_package(source_package)
    expected_source = receipt["inputs"]["source_package"]
    if source["package_sha256sums_sha256"] != expected_source["sha256sums_sha256"]:
        raise Ch7EvidenceBuildReceiptError("frozen v1 SHA256SUMS hash changed")
    if source["manifest_sha256"] != expected_source["manifest_sha256"]:
        raise Ch7EvidenceBuildReceiptError("frozen v1 manifest hash changed")
    for member in expected_source["members"]:
        if sha256_file(source_package / member["path"]) != member["sha256"]:
            raise Ch7EvidenceBuildReceiptError(f"frozen v1 member hash changed: {member['path']}")
    if sha256_file(config) != receipt["inputs"]["config"]["sha256"]:
        raise Ch7EvidenceBuildReceiptError("v2 config hash changed")
    if (
        sha256_file(_repo_path(repository, PORTFOLIO_PATH))
        != receipt["inputs"]["portfolio"]["sha256"]
    ):
        raise Ch7EvidenceBuildReceiptError("v2 portfolio hash changed")

    manifest, hashes = _verify_package(package, review_sidecars=True)
    if hashes != {
        "manifest_sha256": receipt["package"]["manifest_sha256"],
        "sha256sums_sha256": receipt["package"]["sha256sums_sha256"],
        "tree_sha256": receipt["package"]["tree_sha256"],
    }:
        raise Ch7EvidenceBuildReceiptError("tracked package hash changed")
    if (
        manifest["status"] != "blocked_pending_domain_approval"
        or manifest["admission_status"] != "not_admitted"
    ):
        raise Ch7EvidenceBuildReceiptError("package admission boundary changed")
    diagnostic = _diagnose_payload_copy(package)
    check_only = receipt["check_only"]
    if diagnostic != check_only["diagnostic"] or check_only["exit_code"] != 0:
        raise Ch7EvidenceBuildReceiptError("check-only diagnostic changed")

    recorded_a = receipt["rebuilds"]["build_a"]
    recorded_b = receipt["rebuilds"]["build_b"]
    if recorded_a != recorded_b or recorded_a != hashes:
        raise Ch7EvidenceBuildReceiptError("recorded rebuild hashes are inconsistent")
    if rebuild:
        first_manifest, first_hashes, first_diagnostic = _build_independent(
            repository=repository,
            source_package=source_package,
            config=config,
            label="verify-a",
        )
        second_manifest, second_hashes, second_diagnostic = _build_independent(
            repository=repository,
            source_package=source_package,
            config=config,
            label="verify-b",
        )
        if first_manifest != manifest or second_manifest != manifest:
            raise Ch7EvidenceBuildReceiptError("rebuilt manifest differs from tracked package")
        if (
            first_hashes != recorded_a
            or second_hashes != recorded_b
            or first_diagnostic != check_only["diagnostic"]
            or second_diagnostic != check_only["diagnostic"]
        ):
            raise Ch7EvidenceBuildReceiptError("rebuilt package hash differs from receipt")
    return {
        "status": "verified",
        "receipt_payload_sha256": receipt["integrity"]["receipt_payload_sha256"],
        "repository_commit": commit,
        "package_tree_sha256": hashes["tree_sha256"],
        "rebuilds_verified": rebuild,
        "admission_status": manifest["admission_status"],
    }
