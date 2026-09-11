#!/usr/bin/env python3
"""Fail-closed cold restoration for a declared campaign capsule (issue #8895)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPSULE = REPO_ROOT / "tests/validation/fixtures/cold_restore/synthetic_capsule.json"
RECEIPT_SCHEMA = "cold_restore_receipt.v1"
CAPSULE_SCHEMA = "cold_restore_capsule.v1"
FIXTURE_SCHEMA = "campaign_capsule_fixture.v1"
RECEIPT_NAME = "restore_receipt.json"
MANIFEST_NAME = "artifact_manifest.json"
MARKER_NAME = ".cold_restore_owner.json"
CLAIM_BOUNDARY = "Synthetic offline capsule plumbing only; diagnostic-only, not simulation, benchmark evidence, or a scientific claim."  # fmt: skip
CACHE_ENV_VARS = ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME", "ROBOT_SF_ARTIFACT_ROOT", "ROBOT_SF_MODEL_CACHE")  # fmt: skip
BLOCKED_SOURCE_PARTS = {".cache", ".git", ".venv", "cache", "cluster", "gpfs", "institutional", "lustre", "output", "results", "scratch"}  # fmt: skip
EXPECTED_KEYS = {"campaign_id", "capsule_digest", "capsule_id", "config_id", "config_sha256", "episode_schema_path", "expected_row_count", "lineage_report_artifact_id", "lineage_row_artifact_ids", "row_ids", "seed", "source_commit"}  # fmt: skip
VALIDATION_COMMANDS = ("scripts/tools/chunk_manifest.py verify --json", "robot_sf.benchmark.aggregate.read_jsonl(strict=True)", "robot_sf.benchmark.schema_validator.validate_episode", "scripts/tools/lineage_index.py --check --format json", "bounded_consumer: episode-schema reader")  # fmt: skip


class RestoreError(ValueError):
    """Fail-closed error with a stable code and optional safe member name."""

    def __init__(
        self, code: str, *, member: str | None = None, unavailable: tuple[str, ...] = ()
    ) -> None:
        """Store machine-readable failure fields."""
        super().__init__(code)
        self.code, self.member = code, member
        self.unavailable = tuple(sorted(set(unavailable)))


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require(condition: bool, code: str, member: str | None = None) -> None:
    if not condition:
        raise RestoreError(code, member=member)


def _read_json(
    path: Path, code: str = "capsule_unreadable", member: str | None = None
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RestoreError(code, member=member) from exc
    if not isinstance(value, dict):
        raise RestoreError("unsupported_schema", member=member)
    return value


def _lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _reject_symlinks(path: Path, code: str) -> None:
    current = _lexical(path)
    if any(item.is_symlink() for item in (current, *current.parents)):
        raise RestoreError(code)


def _member_path(root: Path, raw: object) -> tuple[str, Path]:
    if not isinstance(raw, str):
        raise RestoreError("path_escape", member="declared_member")
    try:
        from scripts.tools import chunk_manifest
    except ImportError as exc:
        raise RestoreError("dependency_unavailable", unavailable=("chunk_manifest",)) from exc
    try:
        relative = chunk_manifest.normalize_relative_path(raw)
    except chunk_manifest.ChunkManifestError as exc:
        raise RestoreError("path_escape", member=raw[:159] or "declared_member") from exc
    target = (root / relative).resolve(strict=False)
    if not target.is_relative_to(root.resolve(strict=False)):
        raise RestoreError("path_escape", member=relative)
    return relative, target


def _source_path(capsule: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw or "://" in raw:
        raise RestoreError("capsule_schema_invalid", member="fixture_path")
    if Path(raw).is_absolute():
        raise RestoreError("undeclared_local_copy", member="fixture_path")
    if {part.casefold() for part in Path(raw).parts} & BLOCKED_SOURCE_PARTS:
        raise RestoreError("ambient_cache_hit", member="fixture_path")
    candidate = capsule.parent / raw
    _reject_symlinks(candidate, "path_escape")
    source = candidate.resolve(strict=False)
    if not source.is_relative_to(REPO_ROOT):
        raise RestoreError("path_escape", member="fixture_path")
    if not source.is_file():
        raise RestoreError("undeclared_local_copy", member="fixture_path")
    return source


def _payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if payload.get("schema") != FIXTURE_SCHEMA:
        raise RestoreError("unsupported_schema")
    evidence = payload.get("evidence")
    _require(isinstance(evidence, Mapping) and evidence.get("synthetic") is True and evidence.get("execution_status") == "not_run" and evidence.get("evidence_status") == "diagnostic_only" and evidence.get("promotable") is False and evidence.get("claim_boundary") == CLAIM_BOUNDARY, "fixture_boundary_invalid")  # fmt: skip
    expected, files, manifest = (
        payload.get(name) for name in ("expected", "files", "artifact_manifest")
    )
    _require(
        isinstance(expected, Mapping) and EXPECTED_KEYS <= set(expected), "capsule_identity_missing"
    )
    _require(isinstance(files, Mapping) and bool(files), "member_inventory_missing")
    _require(isinstance(manifest, Mapping), "manifest_missing")
    try:
        from scripts.tools import chunk_manifest
    except ImportError as exc:
        raise RestoreError("dependency_unavailable", unavailable=("chunk_manifest",)) from exc
    issues = chunk_manifest.validate_manifest(manifest)
    if issues:
        raise RestoreError("stale_pointer" if "digest" in issues[0]["code"] else "manifest_invalid")
    paths, folded = [], set()
    for raw in files:
        relative, _ = _member_path(Path("/"), raw)
        if relative.casefold() in folded:
            raise RestoreError("member_inventory_mismatch", member=relative)
        folded.add(relative.casefold())
        paths.append(relative)
    manifest_paths = [str(item["path"]) for item in manifest["files"]]
    if sorted(paths) != sorted(manifest_paths):
        missing = sorted(set(manifest_paths) - set(paths))
        raise RestoreError("member_inventory_mismatch", member=missing[0] if missing else None)
    return dict(expected), manifest


def _declaration(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    capsule = path.expanduser().resolve(strict=False)
    if not capsule.is_relative_to(REPO_ROOT):
        raise RestoreError("undeclared_local_copy", member="capsule")
    declaration = _read_json(capsule)
    required = {"schema", "mode", "capsule_id", "fixture_path", "artifact", "source", "environment", "cache_policy", "consumer"}  # fmt: skip
    if declaration.get("schema") != CAPSULE_SCHEMA:
        raise RestoreError("unsupported_schema")
    if set(declaration) != required:
        raise RestoreError("capsule_schema_invalid")
    payload = _read_json(_source_path(capsule, declaration["fixture_path"]))
    expected, manifest = _payload(payload)
    if declaration["mode"] != "synthetic":
        raise RestoreError("real_capsule_unavailable", unavailable=("durable_real_capsule",))
    artifact, source = declaration["artifact"], declaration["source"]
    environment, cache, consumer = (
        declaration["environment"],
        declaration["cache_policy"],
        declaration["consumer"],
    )
    if not all(
        isinstance(value, Mapping) for value in (artifact, source, environment, cache, consumer)
    ):
        raise RestoreError("capsule_schema_invalid")
    _require(
        isinstance(declaration["capsule_id"], str) and declaration["capsule_id"],
        "capsule_identity_missing",
    )
    _require(
        isinstance(artifact.get("version"), str)
        and artifact["version"] == manifest["artifact"]["artifact_version"],
        "artifact_version_mismatch",
        MANIFEST_NAME,
    )
    _require(artifact.get("digest") == manifest["manifest_id"], "stale_pointer", MANIFEST_NAME)
    for actual, wanted, code in (
        (source.get("commit"), expected["source_commit"], "source_identity_mismatch"),
        (source.get("config_id"), expected["config_id"], "config_identity_mismatch"),
        (source.get("seed"), expected["seed"], "seed_identity_mismatch"),
    ):
        _require(actual == wanted, code)
    _require(
        isinstance(source.get("seed"), int) and not isinstance(source["seed"], bool),
        "seed_identity_invalid",
    )
    _require(
        isinstance(source.get("checkpoint_id"), str) and source["checkpoint_id"],
        "checkpoint_identity_missing",
    )
    allowed_os, minimum_python = environment.get("allowed_os"), environment.get("minimum_python")
    _require(
        isinstance(allowed_os, list)
        and allowed_os
        and all(isinstance(item, str) for item in allowed_os),
        "environment_requirement_invalid",
    )
    _require(
        isinstance(minimum_python, list)
        and len(minimum_python) == 2
        and all(isinstance(item, int) for item in minimum_python),
        "environment_requirement_invalid",
    )
    _require(
        consumer.get("name") == "episode_schema_reader"
        and isinstance(consumer.get("max_rows"), int)
        and consumer["max_rows"] >= expected["expected_row_count"],
        "consumer_contract_invalid",
    )
    _require(cache == {"offline_required": True, "ambient_mode": "isolated"}, "ambient_cache_hit")
    return dict(declaration), payload, expected


def _observe(requirement: Mapping[str, Any]) -> dict[str, Any]:
    observed = {"os": platform.system().strip().lower(), "python": [sys.version_info.major, sys.version_info.minor]}  # fmt: skip
    if observed["os"] not in requirement["allowed_os"] or tuple(observed["python"]) < tuple(requirement["minimum_python"]):  # fmt: skip
        raise RestoreError("unsupported_environment", unavailable=("declared_environment",))
    return observed


@contextmanager
def _isolated_caches(root: Path):
    previous = {name: os.environ.get(name) for name in CACHE_ENV_VARS}
    for name in CACHE_ENV_VARS:
        os.environ[name] = str(root / ".isolated-cache" / name.lower())
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _destination_id(declaration: Mapping[str, Any]) -> str:
    return _digest(_canonical({"kind": "task_owned_temporary_root", "capsule_id": declaration.get("capsule_id"), "artifact": declaration.get("artifact")}).encode())  # fmt: skip


def _new_destination(path: Path, identity: str) -> Path:
    candidate = _lexical(path)
    blocked = bool({part.casefold() for part in candidate.parts} & BLOCKED_SOURCE_PARTS)
    if candidate == Path("/") or candidate.exists() or candidate.is_symlink() or blocked:
        raise RestoreError("undeclared_local_copy" if blocked else "destination_exists", member="destination" if blocked else None)  # fmt: skip
    _reject_symlinks(candidate.parent, "path_escape")
    root = candidate.resolve(strict=False)
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()
        (root / MARKER_NAME).write_text(json.dumps({"schema": "cold_restore_owner.v1", "identity": identity}, sort_keys=True) + "\n", encoding="utf-8")  # fmt: skip
    except OSError as exc:
        raise RestoreError("destination_unusable") from exc
    return root


def _materialize(payload: Mapping[str, Any], root: Path) -> int:
    try:
        (root / "artifact").mkdir()
    except OSError as exc:
        raise RestoreError("restore_io") from exc
    transferred = 0
    for raw, value in sorted(payload["files"].items(), key=lambda item: str(item[0])):
        relative, target = _member_path(root / "artifact", raw)
        if not isinstance(value, (str, Mapping)):
            raise RestoreError("fixture_member_invalid", member=relative)
        try:
            data = value.encode() if isinstance(value, str) else (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode()  # fmt: skip
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except (OSError, TypeError, ValueError) as exc:
            raise RestoreError("restore_io", member=relative) from exc
        transferred += len(data)
    try:
        (root / MANIFEST_NAME).write_text(json.dumps(payload["artifact_manifest"], ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")  # fmt: skip
    except (OSError, TypeError, ValueError) as exc:
        raise RestoreError("restore_io", member=MANIFEST_NAME) from exc
    return transferred


def _manifest_failure(exc: Any) -> RestoreError:
    code = getattr(exc, "code", "manifest_invalid")
    member = getattr(exc, "file", None) or MANIFEST_NAME
    if code in {"full_digest_mismatch", "chunk_digest_mismatch", "size_mismatch", "tree_digest_mismatch"}:  # fmt: skip
        code = "checksum_mismatch"
    elif code in {"manifest_digest_mismatch", "partial_manifest", "manifest_unreadable"}:
        code = "stale_pointer"
    elif code == "root_not_directory":
        code, member = "restore_incomplete", "artifact"
    return RestoreError(code, member=member)


def _verify_manifest(root: Path, expected: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:  # fmt: skip
    path = root / MANIFEST_NAME
    if path.is_symlink() or not path.is_file():
        raise RestoreError("restore_incomplete", member=MANIFEST_NAME)
    if (root / "artifact").is_symlink() or not (root / "artifact").is_dir():
        raise RestoreError("restore_incomplete", member="artifact")
    try:
        from scripts.tools import chunk_manifest
    except ImportError as exc:
        raise RestoreError("dependency_unavailable", unavailable=("chunk_manifest",)) from exc
    try:
        manifest = chunk_manifest.load_manifest_file(path)
        _require(
            manifest["artifact"]["artifact_version"] == expected["artifact_version"],
            "artifact_version_mismatch",
            MANIFEST_NAME,
        )
        _require(
            manifest["manifest_id"] == expected["artifact_digest"], "stale_pointer", MANIFEST_NAME
        )
        verification = chunk_manifest.verify_manifest(root / "artifact", manifest=manifest)
    except RestoreError:
        raise
    except chunk_manifest.ChunkManifestError as exc:
        raise _manifest_failure(exc) from exc
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise _manifest_failure(exc) from exc
    if verification["status"] != "ok":
        raise _manifest_failure(type("ManifestFailure", (), verification["failures"][0])())
    return manifest, verification


def _member(root: Path, name: str) -> Path:
    path = root / "artifact" / name
    if path.is_symlink() or not path.is_file():
        raise RestoreError("missing_member", member=name)
    return path


def _document(root: Path, name: str, code: str) -> dict[str, Any]:
    return _read_json(_member(root, name), code, name)


def _check(document: Mapping[str, Any], code: str, **fields: Any) -> None:
    if any(document.get(key) != value for key, value in fields.items()):
        raise RestoreError(code)


def _check_member(root: Path, name: str, code: str, fields: Mapping[str, Any]) -> None:
    _check(_document(root, name, code), code, **fields)


def _semantic(root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:  # noqa: C901
    try:
        from robot_sf.benchmark.aggregate import read_jsonl
        from robot_sf.benchmark.schema_validator import load_schema, validate_episode
        from scripts.tools import lineage_index
        from scripts.validation import cross_host_conformance_capsule as capsule_owner
    except ImportError as exc:
        raise RestoreError("dependency_unavailable", unavailable=("benchmark_validation",)) from exc
    try:
        spec = capsule_owner.load_capsule_spec(_member(root, "capsule.json"))
        digest = capsule_owner.capsule_digest(
            capsule_owner.resolve_capsule(spec, expected["source_commit"])
        )
    except (OSError, KeyError, TypeError, ValueError, capsule_owner.CapsuleContractError) as exc:
        raise RestoreError("unsupported_schema", member="capsule.json") from exc
    _check(
        spec,
        "capsule_identity_mismatch",
        capsule_id=expected["capsule_id"],
        source_commit=expected["source_commit"],
        seed=expected["seed"],
    )
    if digest != expected["capsule_digest"]:
        raise RestoreError("capsule_digest_mismatch", member="capsule.json")
    config_path = _member(root, "config.json")
    try:
        config_sha = _digest(config_path.read_bytes())
    except OSError as exc:
        raise RestoreError("missing_member", member="config.json") from exc
    if config_sha != expected["config_sha256"]:
        raise RestoreError("config_checksum_mismatch", member="config.json")
    _check_member(root, "config.json", "config_identity_mismatch", {"schema": "synthetic_config.v1", "config_id": expected["config_id"], "source_commit": expected["source_commit"], "seed": expected["seed"]})  # fmt: skip
    _check_member(root, "campaign.json", "campaign_identity_mismatch", {"schema": "synthetic_campaign_manifest.v1", "campaign_id": expected["campaign_id"], "capsule_id": expected["capsule_id"], "capsule_digest": expected["capsule_digest"], "config_id": expected["config_id"], "config_sha256": config_sha, "row_ids": expected["row_ids"], "source_commit": expected["source_commit"], "seed": expected["seed"]})  # fmt: skip
    schema_path = (REPO_ROOT / str(expected["episode_schema_path"])).resolve()
    if not schema_path.is_relative_to(REPO_ROOT) or not schema_path.is_file():
        raise RestoreError("unsupported_schema", member="episodes.jsonl")
    try:
        rows = read_jsonl(_member(root, "episodes.jsonl"), strict=True)
        schema = load_schema(schema_path)
        for row in rows:
            validate_episode(row, schema)
    except RestoreError:
        raise
    except Exception as exc:
        raise RestoreError("row_schema_invalid", member="episodes.jsonl") from exc
    ids = [row.get("episode_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise RestoreError("duplicate_row", member="episodes.jsonl")
    if len(rows) != expected["expected_row_count"] or ids != expected["row_ids"]:
        raise RestoreError("row_identity_mismatch", member="episodes.jsonl")
    for row in rows:
        _check(
            row,
            "row_identity_mismatch",
            campaign_id=expected["campaign_id"],
            config_id=expected["config_id"],
            source_commit=expected["source_commit"],
            seed=expected["seed"],
            config_sha256=config_sha,
            evidence_status="diagnostic_only",
            evidence_admissible=False,
        )
    _check_member(root, "missingness.json", "missingness_mismatch", {"schema": "synthetic_missingness_ledger.v1", "campaign_id": expected["campaign_id"], "expected_row_count": expected["expected_row_count"], "observed_row_count": len(rows), "missing_row_ids": [], "excluded_row_ids": [], "claim_boundary": CLAIM_BOUNDARY})  # fmt: skip
    report = _document(root, "report.json", "report_manifest_mismatch")
    _check(report, "report_manifest_mismatch", schema="synthetic_campaign_report.v1", campaign_id=expected["campaign_id"], capsule_id=expected["capsule_id"], config_id=expected["config_id"], config_sha256=config_sha, row_count=expected["expected_row_count"], row_ids=expected["row_ids"], source_commit=expected["source_commit"], seed=expected["seed"], claim_boundary=CLAIM_BOUNDARY, status="complete")  # fmt: skip
    try:
        index = lineage_index.build_index([_member(root, "lineage.json")])
    except Exception as exc:
        raise RestoreError("lineage_invalid", member="lineage.json") from exc
    matches = [
        item
        for item in index.get("rows", [])
        if expected["campaign_id"] in item["records"].get("campaign_ids", [])
    ]
    joined = matches[0]["records"].get("artifact_ids", []) if len(matches) == 1 else []
    if not index.get("ok") or len(matches) != 1 or sorted(item for item in joined if item.startswith("row-")) != sorted(expected["lineage_row_artifact_ids"]) or expected["lineage_report_artifact_id"] not in joined:  # fmt: skip
        raise RestoreError("lineage_identity_mismatch", member="lineage.json")
    return {"status": "verified", "schemas": {"capsule": capsule_owner.CAPSULE_SCHEMA_VERSION, "episode": "v1", "lineage": lineage_index.INDEX_SCHEMA}, "consumer": {"name": "episode_schema_reader", "status": "verified", "rows": len(rows)}, "rows": {"expected_count": expected["expected_row_count"], "observed_count": len(rows), "ids": ids}, "missingness": {"expected": expected["expected_row_count"], "observed": len(rows), "missing_ids": []}, "report": {"status": report["status"], "row_count": report["row_count"]}, "lineage": {"status": "verified", "row_artifact_count": len(expected["lineage_row_artifact_ids"])}, "contract": dict(expected)}  # fmt: skip


def _inventory(root: Path) -> list[dict[str, Any]]:
    artifact = root / "artifact"
    if not artifact.is_dir() or artifact.is_symlink():
        return []
    records = []
    for path in sorted(artifact.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and not path.is_symlink():
            data = path.read_bytes()
            records.append(
                {
                    "path": path.relative_to(artifact).as_posix(),
                    "size_bytes": len(data),
                    "content_sha256": _digest(data),
                }
            )
    return records


def _receipt(
    declaration: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    *,
    status: str,
    destination_id: str | None,
    restored: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    failure: RestoreError | None = None,
    elapsed: float = 0.0,
) -> dict[str, Any]:
    source = declaration.get("source", {}) if declaration else {}
    artifact = declaration.get("artifact", {}) if declaration else {}
    members = [{"path": str(item["path"]), "size_bytes": int(item["size_bytes"]), "content_sha256": str(item["content_sha256"])} for item in (restored or {}).get("members", [])]  # fmt: skip
    body = {"schema_version": RECEIPT_SCHEMA, "mode": declaration.get("mode", "synthetic") if declaration else "synthetic", "claim_boundary": CLAIM_BOUNDARY, "capsule": {"id": declaration.get("capsule_id") if declaration else None, "artifact_version": artifact.get("version"), "artifact_digest": artifact.get("digest"), "capsule_digest": expected.get("capsule_digest") if expected else None, "source_commit": source.get("commit"), "config_id": source.get("config_id"), "config_sha256": expected.get("config_sha256") if expected else None, "seed": source.get("seed"), "checkpoint_id": source.get("checkpoint_id")}, "environment": {"requirement": declaration.get("environment") if declaration else None, "observed": (validation or {}).get("environment")}, "cache": {"status": "isolated", "ambient_reads": 0, "variables": list(CACHE_ENV_VARS)}, "destination": {"kind": "task_owned_temporary_root", "identity": destination_id, "marker": MARKER_NAME}, "restored": {"status": (restored or {}).get("status", "not_started"), "members": members, "member_count": len(members), "byte_count": sum(item["size_bytes"] for item in members), "transferred_bytes": int((restored or {}).get("transferred_bytes", 0)), "artifact_manifest_id": (restored or {}).get("manifest_id"), "tree_sha256": (restored or {}).get("tree_sha256")}, "validation": {"status": "not_run", "commands": list(VALIDATION_COMMANDS), **dict(validation or {})}, "timing": {"elapsed_seconds": round(elapsed, 6), "normalization": "timestamps omitted; elapsed_seconds excluded from receipt_digest"}, "unavailable_dependencies": list(failure.unavailable if failure else ()), "result": {"status": status, "failure": {"code": failure.code, "member": failure.member} if failure else None}}  # fmt: skip
    body["receipt_digest"] = _receipt_digest(body)
    return body


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    if isinstance(body.get("timing"), Mapping):
        body["timing"] = {**body["timing"], "elapsed_seconds": 0.0}
    return _digest(_canonical(body).encode())


def normalize_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return a receipt with measured elapsed time normalized for comparison."""
    value = json.loads(json.dumps(receipt))
    value.setdefault("timing", {})["elapsed_seconds"] = 0.0
    return value


def _validate_receipt(receipt: Mapping[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise RestoreError("dependency_unavailable", unavailable=("jsonschema",)) from exc
    try:
        schema = _read_json(REPO_ROOT / "robot_sf/benchmark/schemas/cold_restore_receipt.v1.json")
        jsonschema.validate(receipt, schema)
    except (OSError, ValueError, jsonschema.ValidationError, jsonschema.SchemaError) as exc:
        raise RestoreError("invalid_receipt_schema") from exc
    if _receipt_digest(receipt) != receipt.get("receipt_digest"):
        raise RestoreError("receipt_digest_mismatch")


def _failed(
    declaration: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    root: Path | None,
    transferred: int,
    error: RestoreError,
    started: float,
) -> dict[str, Any]:
    try:
        members = _inventory(root) if root else []
    except OSError:
        members = []
    status = (
        "unavailable"
        if error.code.endswith("unavailable") or error.code == "offline_required"
        else "failed"
    )
    identity = (
        _destination_id(declaration) if declaration and declaration.get("capsule_id") else None
    )
    return _receipt(
        declaration,
        expected,
        status=status,
        destination_id=identity,
        restored={
            "status": "partial" if root else "not_started",
            "members": members,
            "transferred_bytes": transferred,
        },
        failure=error,
        elapsed=time.perf_counter() - started,
    )


def restore_capsule(
    capsule: str | Path, destination: str | Path, *, offline: bool, mode: str = "synthetic"
) -> dict[str, Any]:
    """Restore one declaration into a fresh root and retain its receipt."""
    started = time.perf_counter()
    declaration: dict[str, Any] | None = {"mode": mode}
    expected: dict[str, Any] | None = None
    root: Path | None = None
    transferred = 0
    try:
        if not offline:
            raise RestoreError("offline_required")
        if mode == "real":
            raise RestoreError("real_capsule_unavailable", unavailable=("durable_real_capsule",))
        if mode != "synthetic":
            raise RestoreError("unsupported_mode")
        declaration, payload, expected = _declaration(Path(capsule))
        expected = {
            **expected,
            "artifact_version": declaration["artifact"]["version"],
            "artifact_digest": declaration["artifact"]["digest"],
        }
        observed = _observe(declaration["environment"])
        root = _new_destination(Path(destination), _destination_id(declaration))
        with _isolated_caches(root):
            transferred = _materialize(payload, root)
            manifest, verification = _verify_manifest(root, expected)
            validation = {**_semantic(root, expected), "environment": observed}
        result = _receipt(
            declaration,
            expected,
            status="verified",
            destination_id=_destination_id(declaration),
            restored={
                "status": "verified",
                "members": manifest["files"],
                "transferred_bytes": transferred,
                "manifest_id": verification["manifest_id"],
                "tree_sha256": verification["tree_sha256"],
            },
            validation=validation,
            elapsed=time.perf_counter() - started,
        )
        _validate_receipt(result)
    except RestoreError as exc:
        result = _failed(declaration, expected, root, transferred, exc, started)
    except (OSError, KeyError, TypeError, ValueError):
        result = _failed(
            declaration,
            expected,
            root,
            transferred,
            RestoreError("restore_validation_error"),
            started,
        )
    if root:
        try:
            (root / RECEIPT_NAME).write_text(
                json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError):
            result = _failed(
                declaration,
                expected,
                root,
                transferred,
                RestoreError("receipt_write_failed", member=RECEIPT_NAME),
                started,
            )
    return result


def verify_destination(destination: str | Path) -> dict[str, Any]:
    """Re-verify a retained root without reading the declaration or mutating the root."""
    started = time.perf_counter()
    candidate = _lexical(Path(destination))
    root = candidate.resolve(strict=False)
    declaration: dict[str, Any] | None = None
    expected: dict[str, Any] | None = None
    try:
        if candidate.is_symlink() or not root.is_dir():
            raise RestoreError("destination_missing")
        _reject_symlinks(candidate, "destination_identity_mismatch")
        marker = _read_json(root / MARKER_NAME, "restore_incomplete", MARKER_NAME)
        receipt = _read_json(root / RECEIPT_NAME, "restore_incomplete", RECEIPT_NAME)
        _validate_receipt(receipt)
        _require(
            marker
            == {"schema": "cold_restore_owner.v1", "identity": receipt["destination"]["identity"]},
            "destination_identity_mismatch",
        )
        if receipt["result"]["status"] != "verified":
            raise RestoreError("restore_incomplete")
        capsule = receipt["capsule"]
        expected = {
            **receipt["validation"]["contract"],
            "artifact_version": capsule["artifact_version"],
            "artifact_digest": capsule["artifact_digest"],
        }
        declaration = {
            "mode": receipt["mode"],
            "capsule_id": capsule["id"],
            "artifact": {
                "version": capsule["artifact_version"],
                "digest": capsule["artifact_digest"],
            },
            "source": {
                "commit": capsule["source_commit"],
                "config_id": capsule["config_id"],
                "seed": capsule["seed"],
                "checkpoint_id": capsule["checkpoint_id"],
            },
            "environment": receipt["environment"]["requirement"]
            or {"allowed_os": ["linux"], "minimum_python": [3, 10]},
        }
        manifest, verification = _verify_manifest(root, expected)
        validation = {
            **_semantic(root, expected),
            "environment": receipt["environment"]["observed"],
        }
        result = _receipt(
            declaration,
            expected,
            status="verified",
            destination_id=receipt["destination"]["identity"],
            restored={
                "status": "verified",
                "members": manifest["files"],
                "transferred_bytes": receipt["restored"]["transferred_bytes"],
                "manifest_id": verification["manifest_id"],
                "tree_sha256": verification["tree_sha256"],
            },
            validation=validation,
            elapsed=time.perf_counter() - started,
        )
        _validate_receipt(result)
        return result
    except RestoreError as exc:
        return _failed(declaration, expected, root if declaration else None, 0, exc, started)
    except (OSError, KeyError, TypeError, ValueError):
        return _failed(
            declaration,
            expected,
            root if declaration else None,
            0,
            RestoreError("restore_validation_error"),
            started,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    restore = commands.add_parser("restore", help="restore and validate a declared capsule")
    restore.add_argument("--capsule", type=Path, default=DEFAULT_CAPSULE)
    restore.add_argument("--destination", type=Path, required=True)
    restore.add_argument(
        "--offline", action="store_true", help="require local declared inputs only"
    )
    restore.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    restore.add_argument("--json", action="store_true")
    verify = commands.add_parser("verify", help="re-verify a retained restore root")
    verify.add_argument("--destination", type=Path, required=True)
    verify.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the verifier and return 0, 1, or 2 for verified, failed, or unavailable."""
    args = _parser().parse_args(argv)
    result = (
        restore_capsule(args.capsule, args.destination, offline=args.offline, mode=args.mode)
        if args.command == "restore"
        else verify_destination(args.destination)
    )
    print(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
        if args.json
        else f"status={result['result']['status']}"
        + (f" code={result['result']['failure']['code']}" if result["result"]["failure"] else "")
    )
    status = result["result"]["status"]
    return 0 if status == "verified" else 2 if status == "unavailable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
