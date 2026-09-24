#!/usr/bin/env python3
"""Build an immutable compute-window staging bundle plan for one authorized workload.

Reads one frozen ``compute_staging_request.v1`` JSON document, binds every declared
source/config/seed/lock/checkpoint input by SHA-256, and emits a deterministic
``compute_staging_bundle.v1`` receipt plus ``file_inventory.json``, ``SHA256SUMS``,
and ``transfer_instructions.json``.  It never copies, uploads, or mutates producer
bytes and records no result data.  Digests reuse
:mod:`robot_sf.benchmark.identity.hash_utils`; checkpoint staging receipts must
satisfy the canonical ``campaign-checkpoint-staging-receipt.v1`` contract; source
identity comes from the local git checkout.

Request schema (normalized repository-relative POSIX paths; tracked inputs must be
committed): ``issue``, ``owner``, ``source`` (commit, tree, dirty_policy=reject),
``environment`` slug, ``configs``, optional ``resolved_config``, ``seed_set``,
optional ``checkpoints`` (``local_file`` or ``staging_receipt``), optional
``model_registry``, ``dependencies.lockfile``, ``command_tokens``,
``expected_row_count``, ``resource_class``, ``output_contract``, ``capability_class``.
Stable fail-closed reason codes: missing_input, untracked_input, checksum_mismatch,
mutable_ref, stale_source, dirty_source, symlink_input, duplicate_member,
ambiguous_input, unauthorized_checkpoint, path_escape, nonportable_path,
private_locator, invalid_field, missing_field, unsupported_value, issue_mismatch,
and immutable_output_conflict.  ``--check`` is side-effect-free; ``--output-root``
writes the four artifacts under one explicit root and refuses to overwrite different
bytes.  Exit codes: 0 ready, 2 blocked, 3 malformed.  Claim boundary: operational
staging readiness only; no benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from robot_sf.benchmark.checkpoint_staging_receipt import CHECKPOINT_STAGING_RECEIPT_SCHEMA
from robot_sf.benchmark.identity.hash_utils import sha256_file, stable_hash

REQUEST_SCHEMA = "compute_staging_request.v1"
BUNDLE_SCHEMA = "compute_staging_bundle.v1"
INVENTORY_SCHEMA = "compute_staging_file_inventory.v1"
TRANSFER_SCHEMA = "compute_staging_transfer_instructions.v1"
MANIFEST_FILENAME = "compute_staging_bundle.v1.json"
INVENTORY_FILENAME = "file_inventory.json"
SUMS_FILENAME = "SHA256SUMS"
TRANSFER_FILENAME = "transfer_instructions.json"
LOCAL_FILE, STAGING_RECEIPT = "local_file", "staging_receipt"
SUMMARY_KEYS = ("path", "sha256", "byte_size", "tracked")
CLAIM_BOUNDARY = (
    "Operational staging readiness only. This receipt binds exact input identities by digest; "
    "it is not benchmark evidence and contains no result data."
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
ISSUE_URL_RE = re.compile(r"^https://github\.com/ll7/robot_sf_ll7/issues/(\d+)$")
PRIVATE_TOKEN_RE = re.compile(r"(^/|^[A-Za-z]:[\\/]|~|\$|://|/home/|/Users/|@github\.com)")


class StagingContractError(ValueError):
    """Raised when a request cannot be read as a staging contract at all."""


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=False
    )


def _summary(member: Mapping[str, Any]) -> dict[str, Any]:
    return {key: member[key] for key in SUMMARY_KEYS}


class _Builder:
    """Collect one request, its digest-pinned members, and fail-closed problems."""

    def __init__(self, repo_root: Path, expected_issue: int | None) -> None:
        self.root = repo_root
        self.expected_issue = expected_issue
        self.problems: list[dict[str, str]] = []

    def add(self, code: str, location: str, message: str) -> None:
        self.problems.append({"code": code, "location": location, "message": message})

    def rel(self, raw: Any, location: str) -> str | None:
        """Validate one normalized repository-relative POSIX path."""
        if not isinstance(raw, str) or not raw:
            return self.add("invalid_field", location, "path must be a non-empty string")
        if CONTROL_RE.search(raw) or "\\" in raw:
            return self.add("nonportable_path", location, "path is not portable")
        if "://" in raw or raw.startswith("~") or "$" in raw:
            return self.add("private_locator", location, "path contains a private locator")
        canonical = raw.rstrip("/")
        posix = PurePosixPath(canonical)
        if canonical != posix.as_posix() or posix.is_absolute() or ".." in posix.parts:
            return self.add("path_escape", location, "path must be normalized and relative")
        if not posix.parts or posix.parts[0].startswith("."):
            return self.add("private_locator", location, "path points at a metadata tree")
        return posix.as_posix()

    def digest(self, raw: Any, location: str) -> str | None:
        if not isinstance(raw, str) or SHA256_RE.fullmatch(raw) is None:
            return self.add("invalid_field", location, "sha256 must be 64 lowercase hex chars")
        return raw

    def slug(self, raw: Any, location: str) -> str | None:
        if not isinstance(raw, str) or SLUG_RE.fullmatch(raw) is None:
            return self.add("invalid_field", location, "value must be a lowercase slug")
        return raw

    def count(self, raw: Any, location: str) -> int | None:
        if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
            return self.add("invalid_field", location, "value must be a positive integer")
        return raw

    def resolve(self, relative_path: str, location: str) -> Path | None:
        """Resolve one input inside the root, rejecting symlinks and escapes."""
        current = self.root
        for part in PurePosixPath(relative_path).parts:
            current = current / part
            if current.is_symlink():
                return self.add("symlink_input", location, f"symlinked input: {relative_path}")
        try:
            resolved = current.resolve(strict=False)
        except OSError:
            return self.add("path_escape", location, f"unresolvable input: {relative_path}")
        if not resolved.is_relative_to(self.root):
            return self.add("path_escape", location, f"input escapes root: {relative_path}")
        if not current.is_file():
            return self.add("missing_input", location, f"missing input: {relative_path}")
        return resolved

    def is_tracked(self, relative_path: str) -> bool:
        return _git(self.root, "ls-files", "--error-unmatch", "--", relative_path).returncode == 0

    def pinned(
        self,
        raw: Any,
        role: str,
        location: str,
        *,
        require_tracked: bool = True,
        identity: str | None = None,
    ) -> dict[str, Any] | None:
        """Validate one digest-pinned input and return its inventory member."""
        if not isinstance(raw, Mapping):
            return self.add("invalid_field", location, "input needs path and sha256")
        relative_path = self.rel(raw.get("path"), f"{location}.path")
        digest = self.digest(raw.get("sha256"), f"{location}.sha256")
        if relative_path is None or digest is None:
            return None
        resolved = self.resolve(relative_path, location)
        if resolved is None:
            return None
        tracked = self.is_tracked(relative_path)
        if require_tracked and not tracked:
            return self.add("untracked_input", location, f"untracked input: {relative_path}")
        observed = sha256_file(resolved)
        if observed != digest:
            return self.add("checksum_mismatch", location, f"digest differs for {relative_path}")
        return {
            "path": relative_path,
            "sha256": observed,
            "byte_size": resolved.stat().st_size,
            "tracked": tracked,
            "roles": [role],
            "identities": [identity] if identity else [],
        }

    def source(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, Mapping):
            return self.add("missing_field", "source", "source identity object is required")
        commit, tree = raw.get("commit"), raw.get("tree")
        for key, value in (("commit", commit), ("tree", tree)):
            if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
                self.add("mutable_ref", f"source.{key}", f"source {key} must be a 40-hex commit")
        if raw.get("dirty_policy", "reject") != "reject":
            self.add("unsupported_value", "source.dirty_policy", "only 'reject' is supported")
        head = _git(self.root, "rev-parse", "HEAD")
        if head.returncode != 0:
            return self.add("missing_input", "source", "repository root is not a git checkout")
        if commit != head.stdout.strip():
            self.add("stale_source", "source.commit", "declared commit is not HEAD")
        if tree != _git(self.root, "rev-parse", "HEAD^{tree}").stdout.strip():
            self.add("stale_source", "source.tree", "declared tree is not HEAD tree")
        dirty = bool(_git(self.root, "status", "--porcelain").stdout.strip())
        if dirty:
            self.add("dirty_source", "source", "worktree is dirty; commit or stash first")
        return {"commit": commit, "tree": tree, "dirty_policy": "reject", "dirty": dirty}

    def receipt_member(
        self, relative_path: str, checkpoint_sha: str, config_digests: set[str], location: str
    ) -> dict[str, Any] | None:
        """Admit one canonical checkpoint-staging receipt as permitted staging evidence."""
        resolved = self.resolve(relative_path, location)
        if resolved is None:
            return None
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self.add("invalid_field", location, "staging receipt is not readable JSON")
        arms = payload.get("arms") if isinstance(payload, Mapping) else None
        pinned = (
            {arm.get("checkpoint_sha256") for arm in arms if isinstance(arm, Mapping)}
            if isinstance(arms, list)
            else set()
        )
        permitted = isinstance(payload, Mapping) and all(
            (
                payload.get("schema_version") == CHECKPOINT_STAGING_RECEIPT_SCHEMA,
                payload.get("status") == "ok",
                payload.get("mode") == "enforced_staged",
                payload.get("stage") is True,
                payload.get("submit_safe") is True,
                payload.get("campaign_config_sha256") in config_digests,
                checkpoint_sha in pinned,
            )
        )
        if not permitted:
            return self.add("unauthorized_checkpoint", location, "staging receipt not permitted")
        return {
            "path": relative_path,
            "sha256": sha256_file(resolved),
            "byte_size": resolved.stat().st_size,
            "tracked": self.is_tracked(relative_path),
            "roles": ["staging_receipt"],
            "identities": [],
        }

    def checkpoints(  # noqa: C901 - one bounded validation pass per checkpoint
        self, raw: Any, config_digests: set[str]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        members: list[dict[str, Any]] = []
        seen: set[str] = set()
        if raw is None:
            return records, members
        if not isinstance(raw, list):
            self.add("invalid_field", "checkpoints", "checkpoints must be a list")
            return records, members
        for index, item in enumerate(raw):
            location = f"checkpoints[{index}]"
            if not isinstance(item, Mapping):
                self.add("invalid_field", location, "checkpoint must be an object")
                continue
            identity = self.slug(item.get("identity"), f"{location}.identity")
            digest = self.digest(item.get("sha256"), f"{location}.sha256")
            resolution = item.get("resolution", LOCAL_FILE)
            member, receipt_path = None, None
            if resolution == LOCAL_FILE:
                member = self.pinned(
                    {"path": item.get("path"), "sha256": item.get("sha256")},
                    "checkpoint",
                    location,
                    identity=identity,
                )
            elif resolution == STAGING_RECEIPT:
                receipt_path = self.rel(item.get("staging_receipt"), f"{location}.staging_receipt")
                if receipt_path is not None and digest is not None:
                    member = self.receipt_member(receipt_path, digest, config_digests, location)
            else:
                self.add(
                    "unsupported_value",
                    f"{location}.resolution",
                    "resolution must be local_file or staging_receipt",
                )
            if identity is not None:
                if identity in seen:
                    self.add("ambiguous_input", location, f"duplicate identity: {identity}")
                seen.add(identity)
            records.append(
                {
                    "identity": identity,
                    "sha256": digest,
                    "resolution": resolution,
                    "bytes_present": resolution == LOCAL_FILE and member is not None,
                    "path": member["path"] if resolution == LOCAL_FILE and member else None,
                    "staging_receipt": receipt_path,
                }
            )
            if member is not None:
                members.append(member)
        return records, members

    def configs(self, raw: Any) -> list[dict[str, Any]]:
        members: list[dict[str, Any]] = []
        if not isinstance(raw, list) or not raw:
            self.add("missing_field", "configs", "configs must be a non-empty list")
            return members
        for index, item in enumerate(raw):
            member = self.pinned(item, "config", f"configs[{index}]")
            if member is None:
                continue
            if any(existing["path"] == member["path"] for existing in members):
                self.add("duplicate_member", f"configs[{index}]", "duplicate config path")
            members.append(member)
        return members

    def seed_set(self, raw: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        if not isinstance(raw, Mapping):
            self.add("missing_field", "seed_set", "seed_set identity is required")
            return None, []
        identity = self.slug(raw.get("identity"), "seed_set.identity")
        seed_count = self.count(raw.get("seed_count"), "seed_set.seed_count")
        member = self.pinned(raw, "seed_set", "seed_set", identity=identity)
        if member is None:
            return None, []
        return {**_summary(member), "identity": identity, "seed_count": seed_count}, [member]

    def tokens(self, raw: Any) -> list[str]:
        tokens: list[str] = []
        if not isinstance(raw, list) or not raw:
            self.add("missing_field", "command_tokens", "command_tokens must be a non-empty list")
            return tokens
        for index, token in enumerate(raw):
            if not isinstance(token, str) or not token or CONTROL_RE.search(token):
                self.add("invalid_field", f"command_tokens[{index}]", "token must be a string")
            elif PRIVATE_TOKEN_RE.search(token):
                self.add("private_locator", f"command_tokens[{index}]", "private token locator")
            else:
                tokens.append(token)
        return tokens

    def output_contract(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, Mapping):
            return self.add("missing_field", "output_contract", "output_contract is required")
        local_root = self.rel(raw.get("local_root"), "output_contract.local_root")
        raw_paths = raw.get("required_paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            self.add("missing_field", "output_contract.required_paths", "required_paths missing")
            return None
        required = [
            self.rel(item, f"output_contract.required_paths[{i}]")
            for i, item in enumerate(raw_paths)
        ]
        if local_root is None or not all(required):
            return None
        return {"local_root": local_root, "required_paths": required}

    def issue(self, raw: Any) -> dict[str, Any]:
        number = raw.get("number") if isinstance(raw, Mapping) else None
        url = raw.get("url") if isinstance(raw, Mapping) else None
        if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
            self.add("invalid_field", "issue.number", "issue number must be a positive integer")
            number = None
        match = ISSUE_URL_RE.fullmatch(url) if isinstance(url, str) else None
        if match is None:
            self.add("invalid_field", "issue.url", "issue url must be the public repository url")
        elif number is not None and int(match.group(1)) != number:
            self.add("issue_mismatch", "issue", "issue number and url disagree")
        if self.expected_issue is not None and number is not None and number != self.expected_issue:
            self.add("issue_mismatch", "issue.number", "request issue does not match --issue")
        return {"number": number, "url": url if match is not None else None}


def _merge_members(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for member in members:
        existing = merged.setdefault(
            member["path"], {**member, "roles": list(member["roles"]), "identities": []}
        )
        for role in member["roles"]:
            if role not in existing["roles"]:
                existing["roles"].append(role)
        for identity in member["identities"]:
            if identity not in existing["identities"]:
                existing["identities"].append(identity)
    for member in merged.values():
        member["roles"] = sorted(member["roles"])
        member["identities"] = sorted(member["identities"])
    return [merged[path] for path in sorted(merged)]


def _finalize(report: dict[str, Any]) -> dict[str, Any]:
    report["status"] = "ready" if not report["problems"] else "blocked"
    report["reason_codes"] = sorted({problem["code"] for problem in report["problems"]})
    report["problems"] = sorted(
        report["problems"], key=lambda item: (item["location"], item["code"], item["message"])
    )
    report.pop("manifest_sha256", None)
    report["manifest_sha256"] = stable_hash(report)
    return report


def build_report(
    request: Any, repo_root: Path, expected_issue: int | None = None
) -> dict[str, Any]:
    """Build a deterministic staging bundle report, ready or blocked."""
    if not isinstance(request, Mapping):
        raise StagingContractError("staging request must be a JSON object")
    builder = _Builder(repo_root, expected_issue)
    if request.get("schema_version") != REQUEST_SCHEMA:
        builder.add("schema_version_mismatch", "schema_version", f"must be {REQUEST_SCHEMA}")
    issue_record = builder.issue(request.get("issue"))
    owner = builder.slug(request.get("owner"), "owner")
    source = builder.source(request.get("source"))
    environment = builder.slug(request.get("environment"), "environment")
    configs = builder.configs(request.get("configs"))
    resolved = registry = lockfile = None
    for key, tracked in (("resolved_config", False), ("model_registry", True)):
        raw_value = request.get(key)
        if raw_value is not None:
            value = builder.pinned(raw_value, key, key, require_tracked=tracked)
            if key == "resolved_config":
                resolved = value
            else:
                registry = value
    dependencies = request.get("dependencies")
    lock_raw = dependencies.get("lockfile") if isinstance(dependencies, Mapping) else None
    if lock_raw is None:
        builder.add("missing_field", "dependencies.lockfile", "lockfile identity is required")
    else:
        lockfile = builder.pinned(lock_raw, "lockfile", "dependencies.lockfile")
    seed_set, seed_members = builder.seed_set(request.get("seed_set"))
    checkpoints, checkpoint_members = builder.checkpoints(
        request.get("checkpoints"), {member["sha256"] for member in configs}
    )
    members = _merge_members(
        [
            *configs,
            *seed_members,
            *checkpoint_members,
            *([resolved] if resolved is not None else []),
            *([registry] if registry is not None else []),
            *([lockfile] if lockfile is not None else []),
        ]
    )
    report: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA,
        "issue": issue_record,
        "owner": owner,
        "source": source,
        "environment": environment,
        "configs": [_summary(member) for member in configs],
        "resolved_config": _summary(resolved) if resolved is not None else None,
        "seed_set": seed_set,
        "checkpoints": checkpoints,
        "model_registry": _summary(registry) if registry is not None else None,
        "dependencies": _summary(lockfile) if lockfile is not None else None,
        "command_tokens": builder.tokens(request.get("command_tokens")),
        "expected_row_count": builder.count(
            request.get("expected_row_count"), "expected_row_count"
        ),
        "resource_class": builder.slug(request.get("resource_class"), "resource_class"),
        "output_contract": builder.output_contract(request.get("output_contract")),
        "capability_class": builder.slug(request.get("capability_class"), "capability_class"),
        "members": members,
        "file_count": len(members),
        "total_bytes": sum(member["byte_size"] for member in members),
        "problems": builder.problems,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    report["bundle_id"] = stable_hash(
        {
            "schema_version": BUNDLE_SCHEMA,
            "issue": issue_record,
            "owner": owner,
            "source": source,
            "members": members,
        }
    )
    return _finalize(report)


def _artifact_texts(report: Mapping[str, Any]) -> dict[str, str]:
    commit = (report.get("source") or {}).get("commit")
    inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "bundle_id": report["bundle_id"],
        "source_commit": commit,
        "file_count": report["file_count"],
        "total_bytes": report["total_bytes"],
        "members": report["members"],
    }
    transfer = {
        "schema_version": TRANSFER_SCHEMA,
        "bundle_id": report["bundle_id"],
        "source_commit": commit,
        "manifest": MANIFEST_FILENAME,
        "inventory": INVENTORY_FILENAME,
        "sha256sums": SUMS_FILENAME,
        "verify_command_template": f"cd <staging_root> && sha256sum --check {SUMS_FILENAME}",
        "network_access": False,
        "uploads": False,
        "copy_policy": (
            "Transfer only bytes pinned by this receipt; verify with SHA256SUMS before execution. "
            "This bundle records no result data."
        ),
    }
    return {
        MANIFEST_FILENAME: json.dumps(report, indent=2, sort_keys=True) + "\n",
        INVENTORY_FILENAME: json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        SUMS_FILENAME: "".join(
            f"{member['sha256']}  {member['path']}\n" for member in report["members"]
        ),
        TRANSFER_FILENAME: json.dumps(transfer, indent=2, sort_keys=True) + "\n",
    }


def write_bundle(
    report: dict[str, Any], output_root: Path, repo_root: Path
) -> tuple[dict[str, Any], list[str]]:
    """Write the four receipt artifacts, refusing to overwrite different bytes."""
    if report["status"] != "ready":
        raise StagingContractError("refusing to write a blocked compute staging bundle")
    inside = (
        output_root.relative_to(repo_root).as_posix()
        if output_root.is_relative_to(repo_root)
        else ""
    )
    if output_root == repo_root or (inside and not inside.startswith("output/")):
        raise StagingContractError("--output-root must be outside the checkout or under output/")
    texts = _artifact_texts(report)
    for name, text in texts.items():
        target = output_root / name
        if target.exists() and target.read_text(encoding="utf-8") != text:
            report["problems"] = [
                *report["problems"],
                {
                    "code": "immutable_output_conflict",
                    "location": name,
                    "message": "existing artifact differs; refusing to overwrite immutable bytes",
                },
            ]
            return _finalize(report), []
    output_root.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (output_root / name).write_text(text, encoding="utf-8")
    return report, sorted(texts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True, help="compute_staging_request JSON.")
    parser.add_argument("--issue", type=int, required=True, help="Authorized public issue number.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument("--check", action="store_true", help="Report only; write nothing.")
    parser.add_argument("--output-root", type=Path, default=None, help="Explicit bundle root.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the staging-bundle builder and return the process exit code."""
    args = _parser().parse_args(argv)
    if args.check == (args.output_root is not None):
        print("error: pass exactly one of --check or --output-root", file=sys.stderr)
        return 3
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        report = build_report(request, args.repo_root.resolve(), args.issue)
        written: list[str] = []
        if report["status"] == "ready" and args.output_root is not None:
            report, written = write_bundle(
                report, args.output_root.resolve(), args.repo_root.resolve()
            )
    except (OSError, json.JSONDecodeError, StagingContractError) as exc:
        print(f"compute staging bundle: malformed request: {exc}", file=sys.stderr)
        return 3
    payload: dict[str, Any] = dict(report)
    if args.output_root is not None:
        payload["output_root"] = str(args.output_root.resolve())
        payload["written_files"] = written
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif report["status"] == "ready":
        print(f"compute staging bundle: ready ({report['file_count']} files)")
    else:
        print(f"compute staging bundle: blocked ({', '.join(report['reason_codes'])})")
        for problem in report["problems"]:
            print(f"  [{problem['code']}] {problem['location']}: {problem['message']}")
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
