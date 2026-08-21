"""Build and verify the Chapter 7 v2 evidence build receipt.

The receipt records reproducible build provenance for the collision-excluded v2
package. It is deliberately separate from the package and from the future
``ch7-evidence-admission.v2`` receipt.
"""

# evidence-writer-exempt: the receipt is an immutable, self-hashed one-line JSON artifact;
# shared write_json would change its exact serialization and invalidate the integrity contract.
# write_review_sidecar binds the exact bytes to the required generated-evidence marker.

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from robot_sf.evidence.writers import write_review_sidecar
from scripts.analysis import build_ch7_evidence_package_v2 as builder
from scripts.analysis import verify_ch7_evidence_admission as admission
from scripts.analysis import verify_ch7_evidence_admission_v2 as admission_v2

ROOT = Path(__file__).parents[2]
RECEIPT_SCHEMA = ROOT / "robot_sf/benchmark/schemas/ch7-evidence-build-receipt.v1.json"
DEFAULT_RECEIPT = Path("docs/context/evidence/issue_7410_ch7_evidence_build_receipt.v1.json")
PACKAGE = Path("docs/context/evidence/issue_7322_ch7_evidence_package_v2")
SOURCE_PACKAGE = Path("docs/context/evidence/issue_6792_ch7_evidence_package_v1")
CONFIG = Path("configs/analysis/ch7_evidence_package.v2.yaml")
PORTFOLIO = Path("configs/analysis/ch7_worked_example_portfolio.v2.yaml")
PAYLOAD_TREE_SCOPE = (
    "SHA256SUMS-listed payload files plus SHA256SUMS; *.review.json sidecars excluded"
)
TREE_HASH_ALGORITHM = "relative-path-bytes-then-SHA256SUMS.v1"
SERIALIZATION = "strict-json-sort-keys-utf8-newline.v1"
CLAIM_BOUNDARY = (
    "Build provenance only; this receipt is not an admission receipt, domain approval, "
    "publication authorization, benchmark result, or paper-facing evidence."
)
DIAGNOSTIC_SCHEMA_VERSION = "ch7-evidence-admission-diagnostic.v1"
EXPECTED_BLOCKERS = [
    "domain_approval_pending",
    "external_admission_receipt_required",
    "metric_semantics_excluded_issue_7042",
]


class Ch7EvidenceBuildReceiptError(ValueError):
    """Raised when build receipt generation or verification fails closed."""


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Ch7EvidenceBuildReceiptError(f"cannot hash file: {path}") from exc
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Ch7EvidenceBuildReceiptError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise Ch7EvidenceBuildReceiptError(f"{label} must be a JSON object: {path}")
    return dict(payload)


def _validate(payload: Mapping[str, Any], label: str) -> None:
    schema = (
        _read_object(RECEIPT_SCHEMA, "build receipt schema") if label == "build receipt" else None
    )
    if schema is None:
        raise Ch7EvidenceBuildReceiptError(f"unsupported schema validation target: {label}")
    try:
        errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=str)
    except (TypeError, ValidationError) as exc:
        raise Ch7EvidenceBuildReceiptError(f"{label} schema is invalid") from exc
    if errors:
        details = "; ".join(error.message for error in errors[:3])
        raise Ch7EvidenceBuildReceiptError(f"{label} validation failed: {details}")


def _repo_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise Ch7EvidenceBuildReceiptError(f"unsafe repository path: {relative}")
    resolved = (ROOT / path).resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise Ch7EvidenceBuildReceiptError(f"repository path escapes checkout: {relative}")
    return resolved


def _hash_repo_path(relative: str) -> str:
    path = _repo_path(relative)
    if not path.is_file():
        raise Ch7EvidenceBuildReceiptError(f"repository file is missing: {relative}")
    return _sha256_file(path)


def _run_text(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise Ch7EvidenceBuildReceiptError(f"command is unavailable: {' '.join(command)}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise Ch7EvidenceBuildReceiptError(
            f"command failed ({result.returncode}): {' '.join(command)}: {detail}"
        )
    return result.stdout.strip()


def _git_value(*arguments: str) -> str:
    return _run_text(("git", *arguments))


def _uv_version() -> str:
    raw = _run_text(("uv", "--version"))
    fields = raw.split()
    if len(fields) < 2 or fields[0] != "uv":
        raise Ch7EvidenceBuildReceiptError(f"unexpected uv version output: {raw}")
    return " ".join(fields[:2])


def _check_uv_lock() -> None:
    _run_text(("uv", "lock", "--check"))


def _tree_hash(root: Path, *, exclude_review_sidecars: bool = False) -> str:
    if not root.is_dir():
        raise Ch7EvidenceBuildReceiptError(f"package output is missing: {root}")
    digest = hashlib.sha256()
    paths = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(root).as_posix()
        if exclude_review_sidecars and relative.endswith(".review.json"):
            continue
        paths.append((relative, path))
    for relative, path in sorted(paths):
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    sums_path = root / "SHA256SUMS"
    if not sums_path.is_file():
        raise Ch7EvidenceBuildReceiptError(f"package output is missing SHA256SUMS: {root}")
    digest.update(sums_path.read_bytes())
    return digest.hexdigest()


def _source_snapshot() -> dict[str, Any]:
    source_root = ROOT / SOURCE_PACKAGE
    snapshot = {
        "v1_package_path": SOURCE_PACKAGE.as_posix(),
        "v1_package_sha256sums_sha256": _sha256_file(source_root / "SHA256SUMS"),
        "v1_manifest_sha256": _sha256_file(source_root / "manifest.json"),
        "v1_audit_member_path": builder.SOURCE_AUDIT_MEMBER,
        "v1_audit_member_sha256": _sha256_file(source_root / builder.SOURCE_AUDIT_MEMBER),
        "v1_reduced_atlas_member_path": builder.SOURCE_REDUCED_ATLAS_MEMBER,
        "v1_reduced_atlas_member_sha256": _sha256_file(
            source_root / builder.SOURCE_REDUCED_ATLAS_MEMBER
        ),
        "v2_config_path": CONFIG.as_posix(),
        "v2_config_sha256": _sha256_file(ROOT / CONFIG),
        "v2_portfolio_config_path": PORTFOLIO.as_posix(),
        "v2_portfolio_config_sha256": _sha256_file(ROOT / PORTFOLIO),
    }
    if snapshot["v1_package_sha256sums_sha256"] != builder.SOURCE_PACKAGE_SHA256SUMS:
        raise Ch7EvidenceBuildReceiptError(
            "v1 package SHA256SUMS digest differs from builder binding"
        )
    if snapshot["v1_audit_member_sha256"] != builder.SOURCE_AUDIT_SHA256:
        raise Ch7EvidenceBuildReceiptError("v1 audit member digest differs from builder binding")
    if snapshot["v1_reduced_atlas_member_sha256"] != builder.SOURCE_REDUCED_ATLAS_SHA256:
        raise Ch7EvidenceBuildReceiptError(
            "v1 reduced atlas member digest differs from builder binding"
        )
    if snapshot["v2_portfolio_config_sha256"] != builder.PORTFOLIO_CONFIG_SHA256:
        raise Ch7EvidenceBuildReceiptError("v2 portfolio digest differs from builder binding")
    return snapshot


def _package_snapshot() -> dict[str, Any]:
    package_root = ROOT / PACKAGE
    try:
        sums_sha, _listed = admission._verify_members(
            package_root,
            label="durable Chapter 7 v2 evidence package",
            require_review_sidecars=True,
        )
    except admission.Ch7EvidenceAdmissionError as exc:
        raise Ch7EvidenceBuildReceiptError(f"durable package verification failed: {exc}") from exc
    manifest = _read_object(package_root / "manifest.json", "durable v2 package manifest")
    try:
        package_schema = _read_object(
            ROOT / "robot_sf/benchmark/schemas/ch7-evidence-package.v2.json",
            "v2 package schema",
        )
        Draft202012Validator(package_schema).validate(manifest)
    except (ValidationError, TypeError) as exc:
        raise Ch7EvidenceBuildReceiptError(
            "durable v2 package manifest schema validation failed"
        ) from exc
    if manifest.get("status") != "blocked_pending_domain_approval":
        raise Ch7EvidenceBuildReceiptError("durable v2 package is not blocked pending approval")
    if manifest.get("admission_status") != "not_admitted":
        raise Ch7EvidenceBuildReceiptError("durable v2 package is not not-admitted")
    return {
        "path": PACKAGE.as_posix(),
        "status": manifest["status"],
        "admission_status": manifest["admission_status"],
        "manifest_sha256": _sha256_file(package_root / "manifest.json"),
        "sha256sums_sha256": sums_sha,
        "payload_tree_sha256": _tree_hash(package_root, exclude_review_sidecars=True),
        "payload_tree_scope": PAYLOAD_TREE_SCOPE,
    }


def _tool_sources() -> dict[str, dict[str, str]]:
    paths = {
        "receipt_tool": "scripts/analysis/build_ch7_evidence_build_receipt_v1.py",
        "builder": "scripts/analysis/build_ch7_evidence_package_v2.py",
        "verifier": "scripts/analysis/verify_ch7_evidence_admission_v2.py",
        "package_schema": "robot_sf/benchmark/schemas/ch7-evidence-package.v2.json",
        "atlas_schema": "robot_sf/benchmark/schemas/ch7-reduced-publication-atlas.v3.json",
        "admission_schema": "robot_sf/benchmark/schemas/ch7-evidence-admission.v2.json",
        "receipt_schema": "robot_sf/benchmark/schemas/ch7-evidence-build-receipt.v1.json",
    }
    return {key: {"path": path, "sha256": _hash_repo_path(path)} for key, path in paths.items()}


def _dependency_identity() -> dict[str, Any]:
    pyproject_path = ROOT / "pyproject.toml"
    try:
        project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]
    except (KeyError, OSError, tomllib.TOMLDecodeError) as exc:
        raise Ch7EvidenceBuildReceiptError("pyproject.toml project metadata is unreadable") from exc
    if project.get("name") != "robot_sf" or project.get("requires-python") != ">=3.11":
        raise Ch7EvidenceBuildReceiptError("project identity differs from the receipt contract")
    _check_uv_lock()
    return {
        "contract": "pyproject-and-uv-lock-digests.v1",
        "project_name": project["name"],
        "requires_python": project["requires-python"],
        "pyproject_path": "pyproject.toml",
        "pyproject_sha256": _sha256_file(pyproject_path),
        "lockfile_path": "uv.lock",
        "lockfile_sha256": _sha256_file(ROOT / "uv.lock"),
        "uv_lock_check": "passed",
    }


def _environment_snapshot() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "uv_version": _uv_version(),
        "dependency_identity": _dependency_identity(),
    }


def _commands() -> dict[str, str]:
    return {
        "build": (
            "uv run python scripts/analysis/build_ch7_evidence_package_v2.py "
            "--source-package docs/context/evidence/issue_6792_ch7_evidence_package_v1 "
            "--output <fresh-output> --config configs/analysis/ch7_evidence_package.v2.yaml "
            "--check-determinism"
        ),
        "verify": (
            "uv run python scripts/analysis/verify_ch7_evidence_admission_v2.py "
            "--package <fresh-output> --check-only"
        ),
        "receipt_generate": (
            "uv run python scripts/analysis/build_ch7_evidence_build_receipt_v1.py "
            "--receipt docs/context/evidence/issue_7410_ch7_evidence_build_receipt.v1.json "
            "--generate"
        ),
        "receipt_check": (
            "uv run python scripts/analysis/build_ch7_evidence_build_receipt_v1.py "
            "--receipt docs/context/evidence/issue_7410_ch7_evidence_build_receipt.v1.json "
            "--check-only"
        ),
    }


def _verification_snapshot(diagnostic: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = diagnostic.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise Ch7EvidenceBuildReceiptError("v2 check-only diagnostic lacks diagnostics")
    blockers = diagnostics.get("blockers")
    if not isinstance(blockers, list):
        raise Ch7EvidenceBuildReceiptError("v2 check-only diagnostic lacks blockers")
    blocker_codes = [blocker.get("code") for blocker in blockers if isinstance(blocker, Mapping)]
    return {
        "command": _commands()["verify"],
        "exit_code": 0,
        "diagnostic_schema_version": diagnostic.get("schema_version"),
        "package_status": diagnostic.get("status"),
        "admission_status": diagnostic.get("admission_status"),
        "admission_authorized": diagnostics.get("admission_authorized"),
        "empirical_outcomes_admitted": diagnostics.get("empirical_outcomes_admitted"),
        "receipt_created": diagnostics.get("receipt_created"),
        "blocker_codes": blocker_codes,
    }


def _probe_builds(scratch_root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    first = scratch_root / "build-a"
    second = scratch_root / "build-b"
    for output in (first, second):
        builder.build_ch7_evidence_package_v2(
            source_package=ROOT / SOURCE_PACKAGE,
            output=output,
            config_path=ROOT / CONFIG,
            check_determinism=True,
        )
    snapshots = [
        {
            "ordinal": ordinal,
            "output_tree_sha256": builder._tree_hash(output),
            "manifest_sha256": _sha256_file(output / "manifest.json"),
            "sha256sums_sha256": _sha256_file(output / "SHA256SUMS"),
        }
        for ordinal, output in enumerate((first, second), 1)
    ]
    if snapshots[0] != {
        **snapshots[1],
        "ordinal": 1,
    }:
        raise Ch7EvidenceBuildReceiptError("independent v2 builds are not byte-identical")
    diagnostic = admission_v2.diagnose_v2_package(first)
    verification = _verification_snapshot(diagnostic)
    if verification["blocker_codes"] != EXPECTED_BLOCKERS:
        raise Ch7EvidenceBuildReceiptError("v2 check-only blocker set changed")
    return snapshots, verification


def _repository_snapshot() -> dict[str, str]:
    commit = _git_value("rev-parse", "HEAD")
    tree = _git_value("rev-parse", "HEAD^{tree}")
    return {
        "commit": commit,
        "tree": tree,
        "tree_scope": "source checkout tree at the build commit, before the receipt commit",
    }


def _payload_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("integrity", None)
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(payload)
    sealed["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": SERIALIZATION,
        "excluded_json_pointer": "#/integrity",
        "receipt_sha256": _payload_hash(sealed),
    }
    return sealed


def _verify_integrity(payload: Mapping[str, Any]) -> None:
    integrity = payload.get("integrity")
    if not isinstance(integrity, Mapping):
        raise Ch7EvidenceBuildReceiptError("receipt integrity block is missing")
    expected = _payload_hash(payload)
    if integrity.get("receipt_sha256") != expected:
        raise Ch7EvidenceBuildReceiptError("receipt self-hash mismatch")


def generate_receipt(
    receipt_path: Path = DEFAULT_RECEIPT,
    *,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Generate a sealed receipt after two canonical, independent builds."""

    receipt_path = receipt_path if receipt_path.is_absolute() else ROOT / receipt_path
    if receipt_path.exists():
        raise Ch7EvidenceBuildReceiptError(f"refusing to overwrite receipt: {receipt_path}")
    source = _source_snapshot()
    package = _package_snapshot()
    environment = _environment_snapshot()
    if scratch_root is None:
        output_root = ROOT / "output"
        output_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="ch7-evidence-build-receipt-", dir=output_root
        )
    else:
        scratch_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="ch7-evidence-build-receipt-", dir=scratch_root
        )
    with temporary as scratch:
        builds, verification = _probe_builds(Path(scratch))
    if package["payload_tree_sha256"] != builds[0]["output_tree_sha256"]:
        raise Ch7EvidenceBuildReceiptError(
            "durable package payload tree differs from canonical build output"
        )
    payload = {
        "schema_version": "ch7-evidence-build-receipt.v1",
        "issue": 7410,
        "status": "verified",
        "claim_boundary": CLAIM_BOUNDARY,
        "package": package,
        "source": source,
        "build": {
            "repository": _repository_snapshot(),
            "tool_sources": _tool_sources(),
            "commands": _commands(),
            "environment": environment,
            "determinism": {
                "verified": True,
                "tree_hash_algorithm": TREE_HASH_ALGORITHM,
                "serialization": SERIALIZATION,
            },
            "independent_builds": builds,
        },
        "verification": verification,
    }
    receipt = _seal(payload)
    _validate(receipt, "build receipt")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(_canonical_bytes(receipt))
    write_review_sidecar(receipt_path, repo_root=ROOT)
    return receipt


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise Ch7EvidenceBuildReceiptError(f"receipt binding mismatch: {label}")


def _verify_repository(repository: Mapping[str, Any]) -> None:
    commit = repository["commit"]
    try:
        actual_tree = _git_value("rev-parse", f"{commit}^{{tree}}")
    except Ch7EvidenceBuildReceiptError as exc:
        raise Ch7EvidenceBuildReceiptError("recorded build commit is unavailable") from exc
    _require_equal(actual_tree, repository["tree"], "repository.tree")


def _verify_tool_sources(tool_sources: Mapping[str, Any]) -> None:
    for key, binding in tool_sources.items():
        if not isinstance(binding, Mapping):
            raise Ch7EvidenceBuildReceiptError(f"tool source binding is malformed: {key}")
        _require_equal(
            _hash_repo_path(binding["path"]), binding["sha256"], f"build.tool_sources.{key}"
        )


def _verify_durable_package(package: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    actual = _package_snapshot()
    _require_equal(actual, dict(package), "package")
    manifest = _read_object(ROOT / PACKAGE / "manifest.json", "durable v2 package manifest")
    expected_source = {
        "v1_package_sha256sums": source["v1_package_sha256sums_sha256"],
        "v1_manifest_sha256": source["v1_manifest_sha256"],
        "v1_audit_member": source["v1_audit_member_path"],
        "v1_audit_member_sha256": source["v1_audit_member_sha256"],
        "v1_reduced_atlas_member": source["v1_reduced_atlas_member_path"],
        "v1_reduced_atlas_member_sha256": source["v1_reduced_atlas_member_sha256"],
    }
    _require_equal(manifest["source"], expected_source, "package manifest source")
    _require_equal(
        manifest["inputs"]["portfolio_config"],
        {
            "path": source["v2_portfolio_config_path"],
            "sha256": source["v2_portfolio_config_sha256"],
        },
        "package manifest portfolio config",
    )


def _verify_source(source: Mapping[str, Any]) -> None:
    _require_equal(_source_snapshot(), dict(source), "source")


def _verify_environment(environment: Mapping[str, Any]) -> None:
    _require_equal(_environment_snapshot(), dict(environment), "build.environment")


def verify_receipt(
    receipt_path: Path = DEFAULT_RECEIPT,
    *,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Verify every receipt binding, including regenerated independent builds."""

    receipt_path = receipt_path if receipt_path.is_absolute() else ROOT / receipt_path
    receipt = _read_object(receipt_path, "build receipt")
    _validate(receipt, "build receipt")
    _verify_integrity(receipt)
    _require_equal(receipt["claim_boundary"], CLAIM_BOUNDARY, "claim_boundary")
    build = receipt["build"]
    _verify_repository(build["repository"])
    _verify_tool_sources(build["tool_sources"])
    _verify_source(receipt["source"])
    _verify_durable_package(receipt["package"], receipt["source"])
    _verify_environment(build["environment"])
    _require_equal(build["commands"], _commands(), "build.commands")
    _require_equal(
        build["determinism"],
        {
            "verified": True,
            "tree_hash_algorithm": TREE_HASH_ALGORITHM,
            "serialization": SERIALIZATION,
        },
        "build.determinism",
    )
    if scratch_root is None:
        output_root = ROOT / "output"
        output_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="ch7-evidence-build-receipt-check-", dir=output_root
        )
    else:
        scratch_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="ch7-evidence-build-receipt-check-", dir=scratch_root
        )
    with temporary as scratch:
        builds, verification = _probe_builds(Path(scratch))
    _require_equal(builds, build["independent_builds"], "build.independent_builds")
    _require_equal(verification, receipt["verification"], "verification")
    if builds[0]["output_tree_sha256"] != receipt["package"]["payload_tree_sha256"]:
        raise Ch7EvidenceBuildReceiptError("canonical build output differs from durable package")
    return {
        "status": "verified",
        "receipt_sha256": receipt["integrity"]["receipt_sha256"],
        "package_sha256sums_sha256": receipt["package"]["sha256sums_sha256"],
        "independent_output_tree_sha256": builds[0]["output_tree_sha256"],
        "admission_status": receipt["package"]["admission_status"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run receipt generation or fail-closed verification from the command line."""

    args = _parser().parse_args(argv)
    try:
        if args.generate:
            receipt = generate_receipt(args.receipt)
            print(f"ch7 v2 build receipt generated: {receipt['integrity']['receipt_sha256']}")
        else:
            result = verify_receipt(args.receipt)
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    except (Ch7EvidenceBuildReceiptError, OSError, ValidationError) as exc:
        print(f"ch7 v2 build receipt unavailable: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
