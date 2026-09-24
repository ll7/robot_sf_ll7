#!/usr/bin/env python3
"""Verify artifact and environment restoration on a surviving host (#8827).

Validates that transferred artifacts and runtime manifests can be reconstructed
in a clean temporary root, validating checksums and identities, running check-only
smoke assertions without neural network inference or source-host access.
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

REPORT_SCHEMA = "restore_verification_report.v1"
LABEL_RESTORATION_SMOKE = "restoration_smoke"

SUPPORTED_MANIFEST_SCHEMAS = frozenset(
    {"artifact_transfer_manifest.v1", "restored_environment_manifest.v1"}
)
SUPPORTED_ENV_SCHEMAS = frozenset({"environment_manifest.v1"})

PRIVATE_PATH_RE = re.compile(
    r"(?:^|[\s\"'=])(/(?:private|root/\.ssh|etc/(?:shadow|passwd|ssl)|var/run/secrets|opt/secrets)/|[a-zA-Z]:[/\\](?:secrets))",
    re.IGNORECASE,
)
STALE_PATH_RE = re.compile(
    r"(?:^|[\s\"'=])(/(?:home|tmp|var|scratch|work)/[a-zA-Z0-9_\-\.]+|[a-zA-Z]:[/\\]Users[/\\])",
    re.IGNORECASE,
)
CREDENTIAL_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|AWS_SECRET_ACCESS_KEY|bearer\s+[A-Za-z0-9_\-\.]+"
    r"|['\"]?(?:password|passwd|api_key|secret_key)['\"]?\s*[:=]\s*['\"][^'\"]{8,}['\"]",
    re.IGNORECASE,
)
SOURCE_HOST_ACCESS_RE = re.compile(
    r"\b(?:ssh|scp|rsync|sftp|ftp)://|\b(?:git@[a-zA-Z0-9_\-\.]+:)|\bhttps?://(?:cluster|slurm|compute|login|internal|source-host)\b",
    re.IGNORECASE,
)
EDITABLE_CHECKOUT_RE = re.compile(
    r"\.git\b|\bpip\s+install\s+-e\b|develop-eggs|sys\.path\.(?:insert|append)\(['\"](\.\./|[/\\])",
    re.IGNORECASE,
)
TEXT_EXTS = frozenset({".json", ".jsonl", ".txt", ".py", ".yaml", ".yml", ".sh", ".md", ".lock"})
BLOCKED_REASONS = frozenset(
    {
        "source_host_access_attempt",
        "symlink_forbidden",
        "credential_leak",
        "private_path_leak",
        "missing_environment_manifest",
    }
)


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a regular file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data, None
        return None, "content is not a JSON object"
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return None, str(exc)


def _hydrate_files(
    manifest: dict[str, Any],
    manifest_dir: Path,
    root: Path,
    disc: list[str],
    reasons: list[str],
) -> None:
    for item in manifest.get("files", []):
        if not isinstance(item, dict) or "path" not in item:
            disc.append("malformed file entry in manifest")
            continue
        rel = item["path"]
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            disc.append(f"forbidden path traversal: {rel}")
            continue
        target = root / rel
        if target.exists():
            continue
        if "content" in item:
            target.parent.mkdir(parents=True, exist_ok=True)
            content = item["content"]
            if isinstance(content, str):
                target.write_text(content, encoding="utf-8")
            elif isinstance(content, bytes):
                target.write_bytes(content)
        else:
            source = manifest_dir / rel
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            else:
                disc.append(f"missing_file: {rel}")
                if "missing_file" not in reasons:
                    reasons.append("missing_file")


def _inspect_security(
    root: Path,
    manifest: dict[str, Any],
    env: dict[str, Any] | None,
    disc: list[str],
    reasons: list[str],
) -> None:
    def _check(txt: str, src: str) -> None:
        checks = (
            (CREDENTIAL_RE, "credential_leak"),
            (PRIVATE_PATH_RE, "private_path_leak"),
            (SOURCE_HOST_ACCESS_RE, "source_host_access_attempt"),
            (EDITABLE_CHECKOUT_RE, "editable_checkout_dependency"),
            (STALE_PATH_RE, "stale_source_path"),
        )
        for pat, tag in checks:
            if pat.search(txt) and tag not in reasons:
                disc.append(f"{tag}: {src}")
                reasons.append(tag)

    _check(json.dumps(manifest), "manifest")
    if env:
        _check(json.dumps(env), "environment")
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_symlink():
            disc.append(f"symlink_forbidden: {rel}")
            if "symlink_forbidden" not in reasons:
                reasons.append("symlink_forbidden")
        elif p.is_file() and p.suffix in TEXT_EXTS:
            try:
                _check(p.read_text("utf-8", errors="ignore"), rel)
            except OSError as exc:
                disc.append(f"unreadable_file: {rel} ({exc})")


def _verify_rows(root: Path, identity: dict[str, Any], disc: list[str], reasons: list[str]) -> bool:
    f_rel = identity.get("file")
    if not f_rel or not (root / f_rel).is_file():
        disc.append(f"missing_file: {f_rel or 'none'}")
        return False
    rows: list[dict[str, Any]] = []
    try:
        with (root / f_rel).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        disc.append(f"row_parse_error: {f_rel} ({exc})")
        return False
    ok = True
    if "expected_count" in identity and len(rows) != identity["expected_count"]:
        disc.append(f"row_count_mismatch: expected {identity['expected_count']}, got {len(rows)}")
        ok = False
    id_key = identity.get("id_field", "id")
    row_ids = [str(r[id_key]) for r in rows if id_key in r or "row_id" in r]
    if len(set(row_ids)) != len(row_ids):
        disc.append(f"duplicate_row_ids: {f_rel}")
        ok = False
    if "row_ids" in identity and sorted(row_ids) != sorted(identity["row_ids"]):
        disc.append(f"row_ids_mismatch: {f_rel}")
        ok = False
    if not ok and "row_inspection_failed" not in reasons:
        reasons.append("row_inspection_failed")
    return ok


def _verify_checkpoint(
    root: Path, identity: dict[str, Any], disc: list[str], reasons: list[str]
) -> bool:
    f_rel = identity.get("file")
    if not f_rel or not (root / f_rel).is_file():
        disc.append(f"missing_file: {f_rel or 'none'}")
        return False
    if (root / f_rel).stat().st_size == 0:
        disc.append(f"empty_checkpoint: {f_rel}")
        return False
    meta = identity.get("metadata")
    if isinstance(meta, dict) and not meta.get("model_type"):
        disc.append(f"checkpoint_missing_model_type: {f_rel}")
        if "checkpoint_missing_model_type" not in reasons:
            reasons.append("checkpoint_missing_model_type")
        return False
    return True


def _verify_config(
    root: Path, identity: dict[str, Any], disc: list[str], reasons: list[str]
) -> bool:
    f_rel = identity.get("file")
    if not f_rel or not (root / f_rel).is_file():
        disc.append(f"missing_file: {f_rel or 'none'}")
        return False
    data, err = _read_json(root / f_rel)
    if err or not isinstance(data, dict):
        disc.append(f"config_parse_error: {f_rel} ({err})")
        return False
    ok = True
    if "seed" in identity and data.get("seed") != identity["seed"]:
        disc.append(f"seed_mismatch: expected {identity['seed']}, got {data.get('seed')}")
        ok = False
    for k, v in identity.get("parameters", {}).items():
        if data.get(k) != v:
            disc.append(f"config_param_mismatch: {k} (expected {v}, got {data.get(k)})")
            ok = False
    if not ok and "config_mismatch" not in reasons:
        reasons.append("config_mismatch")
    return ok


def _verify_file_checksums(
    files: list[dict[str, Any]],
    root: Path,
    disc: list[str],
    reasons: list[str],
    mismatch: set[str],
    sup: set[str],
) -> bool:
    if not files:
        disc.append("empty_files_manifest")
        reasons.append("empty_files_manifest")
        return False
    ok = True
    for item in files:
        if not isinstance(item, dict) or "path" not in item:
            continue
        target = root / item["path"]
        if not target.is_file():
            ok = False
            continue
        exp_h = item.get("sha256")
        if exp_h and compute_sha256(target) != exp_h:
            ok = False
            disc.append(f"checksum_mismatch: {item['path']}")
            if "checksum_mismatch" not in reasons:
                reasons.append("checksum_mismatch")
            mismatch.add("file_integrity")
        exp_sz = item.get("size_bytes")
        if exp_sz is not None and target.stat().st_size != exp_sz:
            ok = False
            disc.append(f"size_mismatch: {item['path']}")
            if "size_mismatch" not in reasons:
                reasons.append("size_mismatch")
            mismatch.add("file_integrity")
    if ok and not any("missing_file" in d or "checksum_mismatch" in d for d in disc):
        sup.add("file_integrity")
        return True
    return False


def _load_env_data(
    manifest: dict[str, Any],
    m_dir: Path,
    env_path: Path | None,
    disc: list[str],
    reasons: list[str],
) -> dict[str, Any] | None:
    candidate = env_path or manifest.get("environment")
    if isinstance(candidate, dict):
        return candidate
    if isinstance(candidate, (str, Path)):
        p = Path(candidate) if env_path else (m_dir / candidate)
        if p.is_file():
            data, err = _read_json(p)
            if err:
                disc.append(f"env_parse_error: {err}")
                reasons.append("environment_manifest_error")
            return data
        disc.append(f"missing_env_file: {candidate}")
    disc.append("missing_environment_manifest")
    reasons.append("missing_environment_manifest")
    return None


def _verify_environment(
    manifest: dict[str, Any],
    m_dir: Path,
    env_path: Path | None,
    disc: list[str],
    reasons: list[str],
    sup: set[str],
    unavail: set[str],
) -> tuple[dict[str, Any] | None, bool]:
    env_data = _load_env_data(manifest, m_dir, env_path, disc, reasons)
    if not env_data:
        return None, False
    if env_data.get("schema") not in SUPPORTED_ENV_SCHEMAS:
        disc.append(f"unsupported_env_schema: {env_data.get('schema')}")
        reasons.append("unsupported_schema")
        return env_data, False
    sup.add("environment_identity")
    for opt, meta in env_data.get("optional_dependencies", {}).items():
        avail = (
            meta.get("available", not meta.get("unavailable", False))
            if isinstance(meta, dict)
            else (meta != "unavailable")
        )
        (sup if avail else unavail).add(opt)
    return env_data, True


def _verify_manifest_header(
    manifest: dict[str, Any],
    disc: list[str],
    reasons: list[str],
    mismatch: set[str],
    sup: set[str],
) -> tuple[bool, str]:
    schema_ok = manifest.get("schema") in SUPPORTED_MANIFEST_SCHEMAS
    if schema_ok:
        sup.add("schema_compatibility")
    else:
        disc.append(f"unsupported_schema: {manifest.get('schema')}")
        reasons.append("unsupported_schema")
        mismatch.add("schema_compatibility")
    commit = manifest.get("source_commit", "")
    commit_ok = isinstance(commit, str) and bool(re.fullmatch(r"[0-9a-fA-F]{7,40}", commit))
    if commit_ok:
        sup.add("source_commit_tracking")
    else:
        disc.append(f"invalid_source_commit: {commit}")
        reasons.append("invalid_source_commit")
        mismatch.add("source_commit_tracking")
    return schema_ok and commit_ok, commit


def _make_report(
    verdict: str,
    m_id: str,
    commit: str,
    caps: dict[str, set[str]],
    identities: dict[str, bool],
    disc: list[str],
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "verdict": verdict,
        "label": LABEL_RESTORATION_SMOKE,
        "scientific_reproduction": False,
        "manifest_id": m_id,
        "source_commit": commit,
        "capabilities": {k: sorted(v) for k, v in caps.items()},
        "verified_identities": identities,
        "discrepancies": disc,
        "reasons": reasons,
    }


def verify_restoration(
    manifest_path: Path,
    env_path: Path | None,
    root: Path,
    check_only: bool = True,
) -> dict[str, Any]:
    """Execute restoration smoke verification against declared manifests and target root.

    Returns:
        A dictionary containing the restoration verification report.
    """
    _ = check_only
    disc: list[str] = []
    reasons: list[str] = []
    caps: dict[str, set[str]] = {
        "supported": set(),
        "unavailable": set(),
        "mismatch": set(),
        "blocked": set(),
    }
    identities = dict.fromkeys(
        (
            "manifest",
            "checksums",
            "schema",
            "source_commit",
            "config",
            "seed",
            "checkpoint",
            "rows",
            "environment",
        ),
        False,
    )
    if not manifest_path.is_file():
        caps["blocked"].add("manifest_loading")
        return _make_report(
            "blocked",
            "unknown",
            "unknown",
            caps,
            identities,
            [f"missing manifest: {manifest_path}"],
            ["manifest_not_found"],
        )
    manifest, m_err = _read_json(manifest_path)
    if m_err or not isinstance(manifest, dict):
        caps["blocked"].add("manifest_parsing")
        return _make_report(
            "fail",
            "unknown",
            "unknown",
            caps,
            identities,
            [f"manifest json error: {m_err}"],
            ["unsupported_schema"],
        )
    header_ok, commit = _verify_manifest_header(
        manifest, disc, reasons, caps["mismatch"], caps["supported"]
    )
    identities["manifest"] = identities["schema"] = manifest.get("schema") in (
        SUPPORTED_MANIFEST_SCHEMAS
    )
    identities["source_commit"] = header_ok and identities["manifest"]
    env_data, env_ok = _verify_environment(
        manifest,
        manifest_path.parent,
        env_path,
        disc,
        reasons,
        caps["supported"],
        caps["unavailable"],
    )
    identities["environment"] = env_ok
    if not env_ok:
        caps["blocked"].add("environment_identity")
    root.mkdir(parents=True, exist_ok=True)
    _hydrate_files(manifest, manifest_path.parent, root, disc, reasons)
    _inspect_security(root, manifest, env_data, disc, reasons)
    identities["checksums"] = _verify_file_checksums(
        manifest.get("files", []), root, disc, reasons, caps["mismatch"], caps["supported"]
    )
    if "row_identity" in manifest:
        identities["rows"] = _verify_rows(root, manifest["row_identity"], disc, reasons)
        (caps["supported"] if identities["rows"] else caps["mismatch"]).add("row_inspection")
    if "checkpoint_identity" in manifest:
        identities["checkpoint"] = _verify_checkpoint(
            root, manifest["checkpoint_identity"], disc, reasons
        )
        (caps["supported"] if identities["checkpoint"] else caps["mismatch"]).add(
            "checkpoint_metadata"
        )
    if "config_identity" in manifest:
        cfg_ok = _verify_config(root, manifest["config_identity"], disc, reasons)
        identities["config"] = identities["seed"] = cfg_ok
        (caps["supported"] if cfg_ok else caps["mismatch"]).add("config_resolution")
    s_file = manifest.get("summary_identity", {}).get("file")
    if s_file and (root / s_file).is_file():
        caps["supported"].add("summary_inspection")
    if identities["manifest"] and identities["checksums"]:
        caps["supported"].add("check_mode_analysis")
    is_blocked = any(r in reasons for r in BLOCKED_REASONS)
    verdict = "blocked" if is_blocked else ("fail" if disc else "pass")
    return _make_report(
        verdict,
        manifest.get("manifest_id", "unknown"),
        commit,
        caps,
        identities,
        disc,
        reasons,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for verify_restored_environment.py.

    Returns:
        Exit code: 0 for pass, 1 for fail, 2 for blocked.
    """
    parser = argparse.ArgumentParser(
        description="Verify artifact and environment restoration on a surviving host."
    )
    parser.add_argument(
        "--check", action="store_true", help="Perform read-only restoration smoke checks."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to transferred artifact manifest.",
    )
    parser.add_argument(
        "--environment-manifest",
        type=Path,
        default=None,
        help="Optional path to environment manifest.",
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path to fresh temporary restoration root.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format (json or text).",
    )
    args = parser.parse_args(argv)

    report = verify_restoration(
        manifest_path=args.manifest,
        env_path=args.environment_manifest,
        root=args.root,
        check_only=args.check,
    )
    if args.format == "json":
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(f"Verdict: {report['verdict'].upper()} [{report['label']}]\n")
        sys.stdout.write(f"Supported: {', '.join(report['capabilities']['supported']) or 'none'}\n")
        sys.stdout.write(
            f"Unavailable: {', '.join(report['capabilities']['unavailable']) or 'none'}\n"
        )
        if report["discrepancies"]:
            sys.stdout.write(f"Discrepancies: {'; '.join(report['discrepancies'])}\n")
    if report["verdict"] == "pass":
        return 0
    if report["verdict"] == "fail":
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
