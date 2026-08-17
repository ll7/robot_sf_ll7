"""Create and verify the non-admission Chapter 7 v2 build receipt.

The receipt binds a reproducible build to its source commit, exact package
payload, dependency lock, and successful outcome-free admission diagnostic. Its
hash covers only the nested payload, so the receipt can be checked without a
self-referential digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from scripts.analysis import build_ch7_evidence_package_v2 as builder
from scripts.analysis import verify_ch7_evidence_admission as admission
from scripts.analysis import verify_ch7_evidence_admission_v2 as v2_admission

REPO_ROOT = Path(__file__).parents[2]
RECEIPT_SCHEMA_VERSION = "ch7-evidence-build-receipt.v1"
RECEIPT_SCHEMA_PATH = REPO_ROOT / "robot_sf/benchmark/schemas/ch7-evidence-build-receipt.v1.json"
BUILDER_PATH = "scripts/analysis/build_ch7_evidence_package_v2.py"
ADMISSION_VERIFIER_PATH = "scripts/analysis/verify_ch7_evidence_admission_v2.py"
RECEIPT_CHECKER_PATH = "scripts/analysis/verify_ch7_evidence_build_receipt_v1.py"
PACKAGE_SCHEMA_PATH = "robot_sf/benchmark/schemas/ch7-evidence-package.v2.json"
PYPROJECT_PATH = "pyproject.toml"
LOCK_PATH = "uv.lock"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class Ch7EvidenceBuildReceiptError(ValueError):
    """Raised when a build receipt cannot be created or verified."""


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Ch7EvidenceBuildReceiptError(f"file is unreadable: {path}") from exc
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Ch7EvidenceBuildReceiptError(f"{label} is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise Ch7EvidenceBuildReceiptError(f"{label} must be a JSON object")
    return dict(payload)


def _validate_schema(payload: Mapping[str, Any], label: str) -> None:
    try:
        schema = _read_object(RECEIPT_SCHEMA_PATH, "build receipt schema")
        errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=str)
    except (TypeError, ValidationError) as exc:
        raise Ch7EvidenceBuildReceiptError(f"{label} schema is invalid") from exc
    if errors:
        details = "; ".join(error.message for error in errors[:3])
        raise Ch7EvidenceBuildReceiptError(f"{label} validation failed: {details}")


def _repo_path(repo_root: Path, relative: str, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise Ch7EvidenceBuildReceiptError(f"{label} must be a safe repository-relative path")
    resolved_root = repo_root.resolve()
    resolved = (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise Ch7EvidenceBuildReceiptError(f"{label} escapes the repository") from exc
    return resolved


def _relative_path(repo_root: Path, path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise Ch7EvidenceBuildReceiptError(f"{label} must be inside the repository") from exc


def _run(
    command: Sequence[str], repo_root: Path, *, label: str
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command),
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise Ch7EvidenceBuildReceiptError(f"{label} could not start") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise Ch7EvidenceBuildReceiptError(
            f"{label} failed with exit code {result.returncode}: {detail}"
        )
    return result


def _git(repo_root: Path, *arguments: str) -> str:
    return _run(["git", *arguments], repo_root, label="git command").stdout.strip()


def _tracked_worktree_clean(repo_root: Path) -> bool:
    return (
        subprocess.run(["git", "diff", "--quiet"], cwd=repo_root, check=False).returncode == 0
        and subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=repo_root, check=False
        ).returncode
        == 0
    )


def _directory_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update((root / "SHA256SUMS").read_bytes())
    return digest.hexdigest()


def _payload_tree_hash(root: Path, listed_members: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(listed_members):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update((root / "SHA256SUMS").read_bytes())
    return digest.hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise Ch7EvidenceBuildReceiptError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise Ch7EvidenceBuildReceiptError(f"{label} is not a 40-character commit SHA")
    return value


def _source_binding(repo_root: Path, relative: str, label: str) -> dict[str, str]:
    path = _repo_path(repo_root, relative, label)
    if not path.is_file():
        raise Ch7EvidenceBuildReceiptError(f"{label} is missing: {relative}")
    return {"path": relative, "sha256": _sha256_file(path)}


def _uv_version(repo_root: Path) -> str:
    return _run(["uv", "--version"], repo_root, label="uv version query").stdout.strip()


def _build_command(source: str, config: str, output: Path) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        BUILDER_PATH,
        "--source-package",
        source,
        "--config",
        config,
        "--output",
        str(output),
        "--check-determinism",
    ]


def _build_command_template(source: str, config: str) -> list[str]:
    return _build_command(source, config, Path("<scratch-output>"))


def _check_only_command(package: str) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        ADMISSION_VERIFIER_PATH,
        "--package",
        package,
        "--check-only",
    ]


def _check_only_result(repo_root: Path, package: str) -> dict[str, Any]:
    command = _check_only_command(package)
    result = _run(command, repo_root, label="v2 check-only command")
    try:
        diagnostic = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise Ch7EvidenceBuildReceiptError("v2 check-only output is not JSON") from exc
    if not isinstance(diagnostic, Mapping):
        raise Ch7EvidenceBuildReceiptError("v2 check-only output is not an object")
    blockers = diagnostic.get("diagnostics", {}).get("blockers", [])
    if not isinstance(blockers, list) or not all(
        isinstance(item, Mapping) and isinstance(item.get("code"), str) for item in blockers
    ):
        raise Ch7EvidenceBuildReceiptError("v2 check-only blockers are malformed")
    if (
        diagnostic.get("status") != "blocked_pending_domain_approval"
        or diagnostic.get("admission_status") != "not_admitted"
        or diagnostic.get("diagnostics", {}).get("admission_authorized") is not False
        or diagnostic.get("diagnostics", {}).get("receipt_created") is not False
    ):
        raise Ch7EvidenceBuildReceiptError("v2 check-only did not prove the not-admitted boundary")
    return {
        "command": command,
        "exit_code": result.returncode,
        "result_schema": diagnostic.get("schema_version"),
        "status": diagnostic.get("status"),
        "admission_status": diagnostic.get("admission_status"),
        "admission_authorized": diagnostic["diagnostics"]["admission_authorized"],
        "empirical_outcomes_admitted": diagnostic["diagnostics"]["empirical_outcomes_admitted"],
        "receipt_created": diagnostic["diagnostics"]["receipt_created"],
        "blocker_codes": sorted(item["code"] for item in blockers),
        "result_sha256": _sha256_bytes(_canonical_bytes(diagnostic)),
    }


def _receipt_wrapper(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_hash": {
            "algorithm": "sha256",
            "canonicalization": "strict-json-sort-keys-utf8-newline.v1",
            "hashed_object": "payload",
            "sha256": _sha256_bytes(_canonical_bytes(payload)),
        },
        "payload": dict(payload),
    }


def create_receipt(
    *,
    source_package: Path,
    package: Path,
    config: Path,
    receipt: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Run two canonical deterministic builds and write one non-admission receipt."""

    repo_root = repo_root.resolve()
    if receipt.exists():
        raise Ch7EvidenceBuildReceiptError(f"refusing to overwrite receipt: {receipt}")
    source_rel = _relative_path(repo_root, source_package, "source package")
    package_rel = _relative_path(repo_root, package, "v2 package")
    config_rel = _relative_path(repo_root, config, "v2 config")
    source_path = _repo_path(repo_root, source_rel, "source package")
    package_path = _repo_path(repo_root, package_rel, "v2 package")
    config_path = _repo_path(repo_root, config_rel, "v2 config")
    if not package_path.is_dir():
        raise Ch7EvidenceBuildReceiptError(f"v2 package is missing: {package_rel}")

    source_commit = _require_commit(_git(repo_root, "rev-parse", "HEAD"), "source commit")
    source_tree = _require_commit(_git(repo_root, "rev-parse", "HEAD^{tree}"), "source tree")
    implementation = {
        "builder": _source_binding(repo_root, BUILDER_PATH, "builder"),
        "admission_verifier": _source_binding(
            repo_root, ADMISSION_VERIFIER_PATH, "admission verifier"
        ),
        "receipt_checker": _source_binding(repo_root, RECEIPT_CHECKER_PATH, "receipt checker"),
        "receipt_schema": _source_binding(
            repo_root,
            _relative_path(repo_root, RECEIPT_SCHEMA_PATH, "receipt schema"),
            "receipt schema",
        ),
        "package_schema": _source_binding(repo_root, PACKAGE_SCHEMA_PATH, "package schema"),
    }
    package_sums_sha, listed_members = admission._verify_members(
        package_path, label="durable v2 package", require_review_sidecars=True
    )
    package_manifest = _read_object(package_path / "manifest.json", "v2 package manifest")
    package_manifest_sha = _sha256_file(package_path / "manifest.json")
    package_payload_tree_sha = _payload_tree_hash(package_path, listed_members)
    package_directory_tree_sha = _directory_tree_hash(package_path)
    frozen_v1 = builder.verify_v1_source_package(source_path)
    portfolio_rel = builder.PORTFOLIO_CONFIG_PATH.as_posix()
    portfolio_path = _repo_path(repo_root, portfolio_rel, "v2 portfolio")
    if not config_path.is_file() or not portfolio_path.is_file():
        raise Ch7EvidenceBuildReceiptError("v2 config or portfolio is missing")

    with tempfile.TemporaryDirectory(prefix="ch7-v2-receipt-a-") as first_root:
        with tempfile.TemporaryDirectory(prefix="ch7-v2-receipt-b-") as second_root:
            generated = []
            for root in (Path(first_root), Path(second_root)):
                output = root / "package"
                _run(
                    _build_command(source_rel, config_rel, output),
                    repo_root,
                    label="canonical v2 builder",
                )
                generated.append(
                    {
                        "tree_sha256": _directory_tree_hash(output),
                        "manifest_sha256": _sha256_file(output / "manifest.json"),
                        "sha256sums_sha256": _sha256_file(output / "SHA256SUMS"),
                    }
                )

    if any(item["manifest_sha256"] != package_manifest_sha for item in generated):
        raise Ch7EvidenceBuildReceiptError("generated manifest differs from durable package")
    if any(item["sha256sums_sha256"] != package_sums_sha for item in generated):
        raise Ch7EvidenceBuildReceiptError("generated SHA256SUMS differs from durable package")
    if any(item["tree_sha256"] != package_payload_tree_sha for item in generated):
        raise Ch7EvidenceBuildReceiptError(
            "generated output tree differs from durable package payload"
        )

    check_only = _check_only_result(repo_root, package_rel)
    project = tomllib.loads((repo_root / PYPROJECT_PATH).read_text(encoding="utf-8"))["project"]
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "issue": 7410,
        "status": "build_provenance_verified",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "repository": {
            "source_commit": source_commit,
            "source_tree": source_tree,
            "source_tree_definition": "source-commit-tree-before-receipt.v1",
            "source_worktree_tracked_clean": _tracked_worktree_clean(repo_root),
        },
        "package": {
            "path": package_rel,
            "directory_tree_sha256": package_directory_tree_sha,
            "payload_tree_sha256": package_payload_tree_sha,
            "tree_hash_definition": "relative-path-bytes-plus-SHA256SUMS.v1",
            "manifest_sha256": package_manifest_sha,
            "sha256sums_sha256": package_sums_sha,
            "listed_members": listed_members,
            "manifest_status": package_manifest.get("status"),
            "admission_status": package_manifest.get("admission_status"),
        },
        "implementation": implementation,
        "inputs": {
            "v2_config": {"path": config_rel, "sha256": _sha256_file(config_path)},
            "v2_portfolio": {
                "path": portfolio_rel,
                "sha256": _sha256_file(portfolio_path),
            },
            "frozen_v1": {
                "path": source_rel,
                "sha256sums_sha256": frozen_v1["package_sha256sums_sha256"],
                "manifest_sha256": frozen_v1["manifest_sha256"],
                "audit_member": frozen_v1["audit_member"],
                "audit_member_sha256": frozen_v1["audit_member_sha256"],
                "reduced_atlas_member": frozen_v1["reduced_atlas_member"],
                "reduced_atlas_member_sha256": frozen_v1["reduced_atlas_member_sha256"],
            },
        },
        "environment": {
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
                "executable": Path(sys.executable).name,
            },
            "uv_version": _uv_version(repo_root),
            "project": {
                "name": project.get("name"),
                "requires_python": project.get("requires-python"),
                "pyproject": {
                    "path": PYPROJECT_PATH,
                    "sha256": _sha256_file(repo_root / PYPROJECT_PATH),
                },
                "lock": {
                    "path": LOCK_PATH,
                    "sha256": _sha256_file(repo_root / LOCK_PATH),
                },
            },
        },
        "commands": {
            "build": _build_command_template(source_rel, config_rel),
            "check_only": _check_only_command(package_rel),
        },
        "determinism": {
            "output_tree_hashes": [item["tree_sha256"] for item in generated],
            "output_manifest_hashes": [item["manifest_sha256"] for item in generated],
            "output_sha256sums_hashes": [item["sha256sums_sha256"] for item in generated],
            "outputs_match": generated[0] == generated[1],
        },
        "check_only": check_only,
        "admission_boundary": {
            "status": "not_admitted",
            "admission_receipt_created": False,
            "domain_approval_claimed": False,
            "paper_facing_use_authorized": False,
            "benchmark_result_claimed": False,
            "statement": (
                "This is build provenance only; it is not an admission receipt, domain approval, "
                "publication authorization, or benchmark result."
            ),
        },
    }
    wrapper = _receipt_wrapper(payload)
    _validate_schema(wrapper, "build receipt")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_bytes(_canonical_bytes(wrapper))
    return wrapper


def _verify_source_commit(repo_root: Path, repository: Mapping[str, Any]) -> None:
    source_commit = _require_commit(repository.get("source_commit"), "recorded source commit")
    source_tree = _require_commit(repository.get("source_tree"), "recorded source tree")
    if _git(repo_root, "rev-parse", f"{source_commit}^{{tree}}") != source_tree:
        raise Ch7EvidenceBuildReceiptError("recorded source tree does not match source commit")
    current_head = _git(repo_root, "rev-parse", "HEAD")
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, current_head],
            cwd=repo_root,
            check=False,
        ).returncode
        != 0
    ):
        raise Ch7EvidenceBuildReceiptError("current checkout is not descended from source commit")


def _verify_implementation(repo_root: Path, implementation: Mapping[str, Any]) -> None:
    for key, binding in implementation.items():
        path = _repo_path(repo_root, binding["path"], f"implementation.{key}")
        actual = _sha256_file(path)
        expected = _require_sha(binding["sha256"], f"implementation.{key}")
        if actual != expected:
            raise Ch7EvidenceBuildReceiptError(f"implementation hash mismatch: {key}")


def _verify_package(
    repo_root: Path, package_record: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    package = _repo_path(repo_root, package_record["path"], "v2 package")
    sums_sha, listed = admission._verify_members(
        package, label="durable v2 package", require_review_sidecars=True
    )
    if listed != package_record["listed_members"]:
        raise Ch7EvidenceBuildReceiptError("durable package member list changed")
    if sums_sha != package_record["sha256sums_sha256"]:
        raise Ch7EvidenceBuildReceiptError("durable package SHA256SUMS changed")
    if _sha256_file(package / "manifest.json") != package_record["manifest_sha256"]:
        raise Ch7EvidenceBuildReceiptError("durable package manifest changed")
    if _payload_tree_hash(package, listed) != package_record["payload_tree_sha256"]:
        raise Ch7EvidenceBuildReceiptError("durable package payload tree changed")
    if _directory_tree_hash(package) != package_record["directory_tree_sha256"]:
        raise Ch7EvidenceBuildReceiptError("durable package directory tree changed")
    manifest = _read_object(package / "manifest.json", "v2 package manifest")
    if (
        manifest.get("status") != "blocked_pending_domain_approval"
        or manifest.get("admission_status") != "not_admitted"
    ):
        raise Ch7EvidenceBuildReceiptError("package admission boundary changed")
    return package, manifest


def _verify_inputs(repo_root: Path, inputs: Mapping[str, Any]) -> None:
    for key in ("v2_config", "v2_portfolio"):
        binding = inputs[key]
        path = _repo_path(repo_root, binding["path"], f"inputs.{key}")
        actual = _sha256_file(path)
        expected = _require_sha(binding["sha256"], f"inputs.{key}")
        if actual != expected:
            raise Ch7EvidenceBuildReceiptError(f"input hash mismatch: {key}")
    frozen = inputs["frozen_v1"]
    frozen_package = _repo_path(repo_root, frozen["path"], "frozen v1 package")
    actual_frozen = builder.verify_v1_source_package(frozen_package)
    for field, actual in (
        ("sha256sums_sha256", actual_frozen["package_sha256sums_sha256"]),
        ("manifest_sha256", actual_frozen["manifest_sha256"]),
        ("audit_member_sha256", actual_frozen["audit_member_sha256"]),
        ("reduced_atlas_member_sha256", actual_frozen["reduced_atlas_member_sha256"]),
    ):
        if frozen[field] != actual:
            raise Ch7EvidenceBuildReceiptError(f"frozen v1 input hash mismatch: {field}")


def _verify_environment(repo_root: Path, environment: Mapping[str, Any]) -> None:
    project = environment["project"]
    for key in ("pyproject", "lock"):
        binding = project[key]
        path = _repo_path(repo_root, binding["path"], f"environment.project.{key}")
        actual = _sha256_file(path)
        expected = _require_sha(binding["sha256"], f"environment.project.{key}")
        if actual != expected:
            raise Ch7EvidenceBuildReceiptError(f"dependency identity changed: {key}")
    if project["name"] != "robot_sf" or project["requires_python"] != ">=3.11":
        raise Ch7EvidenceBuildReceiptError("project dependency identity changed")


def _verify_determinism(package_record: Mapping[str, Any], determinism: Mapping[str, Any]) -> None:
    generated_hashes = determinism["output_tree_hashes"]
    if (
        len(set(generated_hashes)) != 1
        or generated_hashes[0] != package_record["payload_tree_sha256"]
    ):
        raise Ch7EvidenceBuildReceiptError("independent build tree hashes do not agree")
    if determinism["outputs_match"] is not True:
        raise Ch7EvidenceBuildReceiptError("deterministic build result was not proven")


def _verify_check_only(package: Path, recorded: Mapping[str, Any]) -> None:
    diagnostic = v2_admission.diagnose_v2_package(package)
    if recorded["exit_code"] != 0 or recorded["result_schema"] != diagnostic["schema_version"]:
        raise Ch7EvidenceBuildReceiptError("recorded check-only result is not successful")
    if _sha256_bytes(_canonical_bytes(diagnostic)) != recorded["result_sha256"]:
        raise Ch7EvidenceBuildReceiptError("check-only result changed")
    actual_blockers = sorted(item["code"] for item in diagnostic["diagnostics"]["blockers"])
    if recorded["blocker_codes"] != actual_blockers:
        raise Ch7EvidenceBuildReceiptError("check-only blocker set changed")


def verify_receipt(receipt: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Verify every recorded hash and the non-admission boundary."""

    repo_root = repo_root.resolve()
    wrapper = _read_object(receipt, "build receipt")
    _validate_schema(wrapper, "build receipt")
    payload = wrapper["payload"]
    if _sha256_bytes(_canonical_bytes(payload)) != wrapper["receipt_hash"]["sha256"]:
        raise Ch7EvidenceBuildReceiptError("receipt payload hash mismatch")
    _verify_source_commit(repo_root, payload["repository"])
    if payload["repository"]["source_worktree_tracked_clean"] is not True:
        raise Ch7EvidenceBuildReceiptError(
            "receipt was not generated from a clean tracked worktree"
        )
    _verify_implementation(repo_root, payload["implementation"])
    package, _manifest = _verify_package(repo_root, payload["package"])
    _verify_inputs(repo_root, payload["inputs"])
    _verify_environment(repo_root, payload["environment"])
    _verify_determinism(payload["package"], payload["determinism"])
    _verify_check_only(package, payload["check_only"])
    if payload["admission_boundary"]["status"] != "not_admitted":
        raise Ch7EvidenceBuildReceiptError("receipt admission boundary changed")
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": payload["status"],
        "receipt_sha256": _sha256_file(receipt),
        "source_commit": payload["repository"]["source_commit"],
        "package_payload_tree_sha256": payload["package"]["payload_tree_sha256"],
        "admission_status": payload["admission_boundary"]["status"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--source-package", type=Path, required=True)
    create.add_argument("--package", type=Path, required=True)
    create.add_argument("--config", type=Path, required=True)
    create.add_argument("--receipt", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Create or verify one Chapter 7 v2 build provenance receipt."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            result = create_receipt(
                source_package=args.source_package,
                package=args.package,
                config=args.config,
                receipt=args.receipt,
            )
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        else:
            print(json.dumps(verify_receipt(args.receipt), sort_keys=True, separators=(",", ":")))
    except (
        Ch7EvidenceBuildReceiptError,
        OSError,
        ValidationError,
        KeyError,
        TypeError,
    ) as exc:
        print(f"ch7 evidence build receipt unavailable: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
