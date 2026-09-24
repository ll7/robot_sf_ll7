#!/usr/bin/env python3
"""Export and verify restorable source bundles for admitted workloads (#8851).

A bundle records the exact source identity of one workload: repository URL
classification, commit, tree, parents, ref context, vendored subproject
revisions, generated-source provenance, admitted patch identity, and a compact
tracked-file inventory. Tracked dirty state requires an explicit checksum-bound
patch; untracked and ignored state is always rejected. Shallow repositories,
unsafe declared paths, and unsafe bundle members fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

MANIFEST_SCHEMA = "source_bundle_manifest.v1"
STATUS_SCHEMA = "source_bundle_status.v1"
VERIFICATION_SCHEMA = "source_bundle_verification.v1"

PUBLIC_HOSTS = frozenset({"github.com", "gitlab.com", "codeberg.org"})
CREDENTIAL_RE = re.compile(r"://[^/@\s]+@")
URL_WITH_QUERY_FRAGMENT_RE = re.compile(r"https?://[^\s\"']+[?#][^\s\"']*", re.IGNORECASE)
URL_CREDENTIAL_PARAMETER_RE = re.compile(
    r"(?:^|[?&#;/])(?:access[_-]?token|api[_-]?key|auth(?:orization)?|credential|"
    r"password|passwd|private[_-]?key|secret|sig(?:nature)?|token)(?:=|%3d)",
    re.IGNORECASE,
)
PRIVATE_PATH_RE = re.compile(
    r"(?:^|[\s\"'=])(?:/(?:[^/\s\"']+(?:/[^/\s\"']*)*)?|[a-zA-Z]:[/\\]|\\\\)",
    re.IGNORECASE,
)
BLOCKING_REASONS = frozenset(
    "dirty_tree_not_admitted untracked_state_not_admitted ignored_state_not_admitted "
    "worktree_state_unavailable shallow_clone_rejected shallow_state_unavailable path_escape "
    "output_unavailable missing_object credential_in_remote subproject_missing generated_file_missing "
    "patch_not_found patch_not_bound patch_state_unavailable output_contaminated "
    "output_member_collision output_write_failed git_subproject_rejected gitlink_rejected "
    "subproject_invalid generated_input_invalid generated_duplicate inventory_unavailable "
    "inventory_invalid source_identity_unavailable manifest_invalid manifest_unreadable "
    "checksum_invalid checksum_incomplete missing_checksums restore_workdir_unavailable "
    "bundle_clone_failed admitted_patch_missing admitted_patch_not_applicable".split()
)

STATUS_VALUES = frozenset({"pass", "fail", "blocked"})
RESERVED_BUNDLE_MEMBERS = frozenset({"source.bundle", "manifest.json", "SHA256SUMS"})
GENERATED_METADATA_KEYS = frozenset({"path", "producer", "source", "command"})


def _run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def _is_object_id(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _path_has_symlink_component(path: Path) -> bool:
    """Reject existing symlink components before reading or installing a path."""
    try:
        absolute = path.absolute()
        current = Path(absolute.anchor)
        for component in absolute.parts[1:]:
            current /= component
            if current.is_symlink():
                return True
    except (OSError, RuntimeError, ValueError):
        return True
    return False


def _is_safe_relative_path_text(value: object) -> bool:
    """Return whether *value* has a portable, traversal-free relative spelling."""
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    if "\x00" in text:
        return False
    try:
        posix = PurePosixPath(text)
        windows = PureWindowsPath(text)
        candidate = Path(text)
    except (TypeError, ValueError):
        return False
    return not (
        candidate.is_absolute()
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.root
        or windows.drive
        or ".." in posix.parts
        or ".." in windows.parts
    )


def _safe_relative_path(root: Path, value: object) -> Path | None:
    """Resolve a declared relative path without following it outside *root*."""
    if not _is_safe_relative_path_text(value):
        return None
    try:
        candidate = root / Path(str(value).strip())
        if candidate.is_symlink():
            # A direct symlink is not a stable custody reference, even when its
            # current target happens to remain below the root.
            return None
        resolved_root = root.resolve(strict=False)
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_relative_to(resolved_root) else None


def _record_path_escape(label: str, reasons: list[str], disc: list[str]) -> None:
    reasons.append("path_escape")
    disc.append(f"{label} is not a safe repository-relative path")


def _decoded_url_variants(value: str) -> list[str]:
    variants = [value]
    for _ in range(3):
        decoded = unquote(variants[-1])
        if decoded == variants[-1]:
            break
        variants.append(decoded)
    return variants


def _remote_has_credential_like_data(url: str) -> bool:
    """Return whether a remote contains authority credentials or URL suffix data."""
    for variant in _decoded_url_variants(url):
        if CREDENTIAL_RE.search(variant) or URL_CREDENTIAL_PARAMETER_RE.search(variant):
            return True
        try:
            parsed = urlsplit(variant)
        except ValueError:
            return True
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            return True
    return False


def _sanitize_remote(url: str) -> tuple[str | None, bool]:
    """Return (sanitized_url_or_None, is_public); private or credentialed URLs stay hidden."""
    if not url:
        return None, False
    if any(character.isspace() for character in url):
        return None, False
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return None, False
    had_credentials = _remote_has_credential_like_data(url)
    public = parsed.scheme.lower() == "https" and host in PUBLIC_HOSTS and not had_credentials
    sanitized = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return (sanitized if public else None), public


def _directory_digest(root: Path) -> str:
    rows = []
    resolved_root = root.resolve(strict=True)
    for path in sorted(root.rglob("*")):
        try:
            resolved = path.resolve(strict=False)
        except (OSError, RuntimeError):
            raise ValueError("directory contains an unresolved path") from None
        if not resolved.is_relative_to(resolved_root):
            raise ValueError("directory contains a symlink escaping its root")
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            rows.append(f"{relative} symlink")
        elif path.is_file():
            rows.append(f"{relative} {_sha256_file(path)}")
    return _digest_bytes("\n".join(rows).encode("utf-8"))


def _resolve_identity(
    repo: Path, ref: str, reasons: list[str], disc: list[str]
) -> tuple[str, str, list[str]] | None:
    try:
        commit_proc = _run_git(repo, "rev-parse", f"{ref}^{{commit}}", check=False)
    except (OSError, subprocess.SubprocessError):
        reasons.append("source_identity_unavailable")
        return None
    if commit_proc.returncode != 0:
        reasons.append("missing_object")
        disc.append(f"{ref} does not resolve to a commit")
        return None
    commit = commit_proc.stdout.strip()
    try:
        tree_proc = _run_git(repo, "rev-parse", f"{commit}^{{tree}}", check=False)
        parent_proc = _run_git(repo, "rev-list", "--parents", "-n", "1", commit, check=False)
    except (OSError, subprocess.SubprocessError):
        reasons.append("source_identity_unavailable")
        return None
    if tree_proc.returncode or parent_proc.returncode:
        reasons.append("source_identity_unavailable")
        return None
    tree = tree_proc.stdout.strip()
    parent_fields = parent_proc.stdout.split()
    parents = parent_fields[1:]
    if (
        not _is_object_id(commit)
        or not _is_object_id(tree)
        or any(not _is_object_id(parent) for parent in parents)
    ):
        reasons.append("source_identity_unavailable")
        return None
    return commit, tree, parents


def _is_shallow_repository(repo: Path) -> bool:
    """Read shallow state through Git so linked worktrees use the common admin dir."""
    result = _run_git(repo, "rev-parse", "--is-shallow-repository", check=False)
    value = result.stdout.strip().lower()
    if result.returncode != 0 or value not in {"true", "false"}:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise RuntimeError(f"git shallow-state query failed: {detail}")
    return value == "true"


def _remote_identity(repo: Path, disc: list[str]) -> dict[str, Any]:
    try:
        remote_url = _run_git(
            repo, "config", "--get", "remote.origin.url", check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        remote_url = ""
    sanitized, public = _sanitize_remote(remote_url)
    if remote_url and not public and _remote_has_credential_like_data(remote_url):
        disc.append("remote URL contains credential-like data; recorded as private digest only")
    return {
        "remote_public": public,
        "remote": sanitized,
        "remote_digest": _digest_bytes(remote_url.encode("utf-8")) if remote_url else None,
    }


def _capture_worktree_diff(repo: Path) -> bytes:
    """Capture the exact tracked diff that an admitted patch must reproduce."""
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "--binary",
                "--no-ext-diff",
                "--no-color",
                "HEAD",
                "--",
            ],
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError("tracked diff query failed") from exc
    if result.returncode:
        raise RuntimeError("tracked diff query failed")
    return result.stdout


def _worktree_state(repo: Path) -> tuple[list[str], list[str], list[str]]:
    """Return tracked-dirty, untracked, and ignored entries from Git's full status."""
    status = _run_git(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored",
        check=False,
    )
    if status.returncode != 0:
        detail = status.stderr.strip() or status.stdout.strip() or "no diagnostic"
        raise RuntimeError(f"git status failed: {detail}")
    tracked: list[str] = []
    untracked: list[str] = []
    ignored: list[str] = []
    for line in status.stdout.splitlines():
        if not line.strip():
            continue
        code = line[:2]
        if code == "??":
            untracked.append(line)
        elif code == "!!":
            ignored.append(line)
        else:
            tracked.append(line)
    return tracked, untracked, ignored


def _patch_identity(
    patch_path: Path | None,
    captured_diff: bytes | None,
    base_commit: str,
    base_tree: str,
    reasons: list[str],
    disc: list[str],
) -> tuple[dict[str, Any] | None, bytes | None]:
    if patch_path is None:
        return None, None
    if not isinstance(patch_path, Path):
        reasons.append("patch_not_found")
        return None, None
    try:
        if patch_path.is_symlink():
            _record_path_escape("patch path", reasons, disc)
            return None, None
        if not patch_path.is_file():
            reasons.append("patch_not_found")
            return None, None
        patch_bytes = patch_path.read_bytes()
        patch_size = patch_path.stat().st_size
    except OSError:
        reasons.append("patch_not_found")
        return None, None
    if patch_path.name in RESERVED_BUNDLE_MEMBERS:
        reasons.append("output_member_collision")
        disc.append("admitted patch name is reserved by the bundle format")
        return None, None
    if captured_diff is None:
        reasons.append("patch_state_unavailable")
    elif patch_bytes != captured_diff:
        reasons.append("patch_not_bound")
        disc.append("admitted patch bytes do not equal the captured tracked diff")
    return {
        "path": patch_path.name,
        "sha256": _digest_bytes(patch_bytes),
        "size_bytes": patch_size,
        "captured_diff_sha256": _digest_bytes(captured_diff) if captured_diff is not None else None,
        "captured_diff_size_bytes": len(captured_diff) if captured_diff is not None else None,
        "base_commit": base_commit,
        "base_tree": base_tree,
    }, patch_bytes


def _subproject_rows(
    repo: Path, paths: list[str], reasons: list[str], disc: list[str]
) -> list[dict[str, Any]]:
    if not isinstance(paths, list):
        reasons.append("subproject_invalid")
        return []
    rows: list[dict[str, Any]] = []
    for relative in paths:
        path = _safe_relative_path(repo, relative)
        if path is None:
            _record_path_escape("subproject path", reasons, disc)
            rows.append({"path": relative, "kind": "invalid", "identity": None})
            continue
        if not path.exists():
            reasons.append("subproject_missing")
            rows.append({"path": relative, "kind": "missing", "identity": None})
        elif not path.is_dir():
            reasons.append("subproject_invalid")
            rows.append({"path": relative, "kind": "invalid", "identity": None})
        elif any(candidate.name == ".git" for candidate in path.rglob(".git")):
            reasons.append("git_subproject_rejected")
            rows.append({"path": relative, "kind": "git_rejected", "identity": None})
        else:
            try:
                identity = _directory_digest(path)
            except (OSError, RuntimeError, ValueError):
                _record_path_escape("subproject contents", reasons, disc)
                identity = None
            rows.append({"path": relative, "kind": "vendored", "identity": identity})
    return rows


def _generated_rows(
    repo: Path, entries: list[dict[str, str]], reasons: list[str], disc: list[str]
) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        reasons.append("generated_input_invalid")
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or any(
                key not in GENERATED_METADATA_KEYS or not isinstance(value, str)
                for key, value in entry.items()
            )
            or not isinstance(entry.get("path"), str)
        ):
            reasons.append("generated_input_invalid")
            rows.append({"path": None, "sha256": None})
            continue
        relative = entry["path"].strip()
        if relative in seen:
            reasons.append("generated_duplicate")
        seen.add(relative)
        path = _safe_relative_path(repo, relative)
        if path is None:
            _record_path_escape("generated path", reasons, disc)
            digest = None
        else:
            try:
                tracked = (
                    subprocess.run(
                        ["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", relative],
                        capture_output=True,
                        check=False,
                    ).returncode
                    == 0
                )
                digest = _sha256_file(path) if tracked and path.is_file() else None
            except OSError:
                digest = None
        if path is not None and digest is None:
            reasons.append("generated_file_missing")
        rows.append({**entry, "sha256": digest})
    return rows


def _inventory(
    repo: Path, commit: str, reasons: list[str], disc: list[str]
) -> list[dict[str, Any]]:
    try:
        result = _run_git(repo, "ls-tree", "-r", "-l", commit, check=False)
    except (OSError, subprocess.SubprocessError):
        reasons.append("inventory_unavailable")
        return []
    if result.returncode:
        reasons.append("inventory_unavailable")
        return []
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        meta, _, path = line.partition("\t")
        fields = meta.split()
        if len(fields) < 4 or not path:
            reasons.append("inventory_invalid")
            continue
        if fields[3] == "-":
            size: int | None = None
            reasons.append("gitlink_rejected")
        else:
            try:
                size = int(fields[3])
            except ValueError:
                reasons.append("inventory_invalid")
                continue
        rows.append({"path": path, "mode": fields[0], "size_bytes": size})
    return rows


def _create_bundle(
    repo: Path, out_dir: Path, ref: str, reasons: list[str], disc: list[str]
) -> None:
    if set(reasons) & BLOCKING_REASONS:
        return
    bundle_path = _safe_relative_path(out_dir, "source.bundle")
    if bundle_path is None:
        _record_path_escape("source bundle output path", reasons, disc)
        return
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        created = _run_git(repo, "bundle", "create", str(bundle_path), ref, check=False)
    except (OSError, subprocess.SubprocessError):
        reasons.append("output_write_failed")
        return
    if created.returncode != 0:
        reasons.append("missing_object")
        disc.append(created.stderr.strip().splitlines()[-1] if created.stderr else "")


def _validate_bundle_output(
    out_dir: Path, patch_identity: dict[str, Any] | None, reasons: list[str], disc: list[str]
) -> None:
    """Require a new, uncontaminated output root with a trusted parent."""
    try:
        if _path_has_symlink_component(out_dir):
            _record_path_escape("bundle output directory", reasons, disc)
        elif out_dir.exists():
            reasons.append("output_contaminated")
            disc.append("bundle output root must not already exist")
        elif not out_dir.parent.is_dir():
            reasons.append("output_unavailable")
            disc.append("bundle output parent is unavailable")
    except (OSError, RuntimeError, ValueError):
        reasons.append("output_unavailable")
    if set(reasons) & BLOCKING_REASONS:
        return
    if patch_identity is not None and patch_identity.get("path") in RESERVED_BUNDLE_MEMBERS:
        reasons.append("output_member_collision")
        disc.append("admitted patch name is reserved by the bundle format")


def _collect_source_state(repo: Path, reasons: list[str], disc: list[str]) -> list[str]:
    """Record shallow and full worktree state, returning tracked dirty entries."""
    try:
        if _is_shallow_repository(repo):
            reasons.append("shallow_clone_rejected")
    except RuntimeError as exc:
        reasons.append("shallow_state_unavailable")
        disc.append(str(exc))
    try:
        tracked_dirty, untracked, ignored = _worktree_state(repo)
    except RuntimeError as exc:
        reasons.append("worktree_state_unavailable")
        disc.append(str(exc))
        tracked_dirty, untracked, ignored = [], [], []
    if tracked_dirty:
        reasons.append("dirty_tree_not_admitted")
        disc.append(f"tracked changes present: {len(tracked_dirty)}")
    if untracked:
        reasons.append("untracked_state_not_admitted")
        disc.append(f"untracked entries present: {len(untracked)}")
    if ignored:
        reasons.append("ignored_state_not_admitted")
        disc.append(f"ignored entries present: {len(ignored)}")
    return tracked_dirty


def export_bundle(
    repo: Path,
    out_dir: Path,
    workload_id: str,
    patch_path: Path | None = None,
    subprojects: list[str] | None = None,
    generated: list[dict[str, str]] | None = None,
    ref: str = "HEAD",
) -> dict[str, Any]:
    """Export the exact source state of one ref into a restorable bundle."""
    reasons: list[str] = []
    disc: list[str] = []
    identity = _resolve_identity(repo, ref, reasons, disc)
    if identity is None:
        return {
            "schema": STATUS_SCHEMA,
            "status": "blocked",
            "reasons": sorted(set(reasons)),
            "discrepancies": disc,
        }
    commit, tree, parents = identity
    tracked_dirty = _collect_source_state(repo, reasons, disc)
    captured_diff: bytes | None = None
    if patch_path is not None:
        try:
            captured_diff = _capture_worktree_diff(repo)
        except RuntimeError as exc:
            reasons.append("patch_state_unavailable")
            disc.append(str(exc))
    patch_identity, patch_bytes = _patch_identity(
        patch_path, captured_diff, commit, tree, reasons, disc
    )
    if patch_identity is not None:
        reasons = [r for r in reasons if r != "dirty_tree_not_admitted"]
        if tracked_dirty:
            disc.append("admitted patch covers the tracked changes")
    subproject_rows = _subproject_rows(repo, subprojects or [], reasons, disc)
    generated_rows = _generated_rows(repo, generated or [], reasons, disc)
    _validate_bundle_output(out_dir, patch_identity, reasons, disc)
    inventory = _inventory(repo, commit, reasons, disc)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "workload_id": workload_id,
        "commit": commit,
        "tree": tree,
        "parents": parents,
        "ref_context": "resolved_commit",
        **_remote_identity(repo, disc),
        "subprojects": subproject_rows,
        "generated": generated_rows,
        "patch": patch_identity,
        "inventory_count": len(inventory),
        "inventory": inventory,
        "private_state_rejected": bool(
            {
                "dirty_tree_not_admitted",
                "untracked_state_not_admitted",
                "ignored_state_not_admitted",
            }
            & set(reasons)
        ),
        "status": "blocked" if set(reasons) & BLOCKING_REASONS else ("fail" if reasons else "pass"),
        "reasons": sorted(set(reasons)),
        "discrepancies": disc,
    }
    if manifest["status"] == "pass" and _validate_manifest(manifest, require_complete=True):
        reasons.append("manifest_invalid")
    elif manifest["status"] == "pass":
        _write_staged_bundle(
            repo, out_dir, ref, patch_identity, patch_bytes, manifest, reasons, disc
        )
    manifest["reasons"] = sorted(set(reasons))
    manifest["discrepancies"] = disc
    manifest["status"] = (
        "blocked" if set(reasons) & BLOCKING_REASONS else ("fail" if reasons else "pass")
    )
    return manifest


def _write_bundle_files(out_dir: Path, manifest: dict[str, Any]) -> None:
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    members = sorted(path for path in out_dir.iterdir() if path.name != "SHA256SUMS")
    if any(path.is_symlink() or not path.is_file() for path in members):
        raise OSError("bundle staging contains an unsafe member")
    rows = [f"{_sha256_file(path)}  {path.name}" for path in members]
    with (out_dir / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(rows) + "\n")


def _write_staged_bundle(
    repo: Path,
    out_dir: Path,
    ref: str,
    patch: dict[str, Any] | None,
    patch_bytes: bytes | None,
    manifest: dict[str, Any],
    reasons: list[str],
    disc: list[str],
) -> None:
    """Build privately, then install one complete bundle or leave no output."""
    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=out_dir.parent))
        _create_bundle(repo, staging, ref, reasons, disc)
        if set(reasons) & BLOCKING_REASONS:
            return
        if patch is not None:
            if patch_bytes is None or _digest_bytes(patch_bytes) != patch.get("sha256"):
                reasons.append("patch_not_bound")
                return
            patch_path = _safe_relative_path(staging, patch.get("path"))
            if patch_path is None:
                reasons.append("path_escape")
                return
            with patch_path.open("xb") as handle:
                handle.write(patch_bytes)
        _write_bundle_files(staging, manifest)
        os.rename(staging, out_dir)
        staging = None
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        reasons.append("output_write_failed")
        disc.append(f"bundle writer failed: {type(exc).__name__}")
    finally:
        if staging is not None:
            try:
                shutil.rmtree(staging)
            except OSError:
                pass


def _contains_public_sensitive_text(value: str) -> bool:
    """Return whether a projected string contains a path, credential, or URL suffix."""
    for variant in _decoded_url_variants(value):
        if (
            PRIVATE_PATH_RE.search(variant)
            or re.search(r"(?:^|[/\\])\.\.(?:[/\\]|$)", variant)
            or CREDENTIAL_RE.search(variant)
            or URL_WITH_QUERY_FRAGMENT_RE.search(variant)
            or URL_CREDENTIAL_PARAMETER_RE.search(variant)
        ):
            return True
    return False


def _public_value(value: Any, placeholder: str) -> tuple[Any, bool]:
    """Return a value safe for the public projection and whether it was redacted."""
    if isinstance(value, str) and _contains_public_sensitive_text(value):
        return placeholder, True
    if value is None or isinstance(value, (bool, int, float, str)):
        return value, False
    return placeholder, True


def _public_sequence(value: object) -> tuple[Any, bool]:
    """Sanitize a public scalar or sequence of public scalar values."""
    if not isinstance(value, list):
        return _public_value(value, "[REDACTED]")
    values: list[Any] = []
    sanitized = False
    for item in value:
        safe_item, item_redacted = _public_value(item, "[REDACTED]")
        values.append(safe_item)
        sanitized |= item_redacted
    return values, sanitized


def _public_relative_path(value: object) -> tuple[str, bool]:
    """Return a safe path label without consulting or exposing the local filesystem."""
    if (
        not _is_safe_relative_path_text(value)
        or not isinstance(value, str)
        or "?" in value
        or "#" in value
        or _contains_public_sensitive_text(value)
    ):
        return "[REDACTED_PATH]", True
    return value, False


def _public_reasons(value: object) -> tuple[list[str], bool]:
    """Keep only machine-readable public reason codes."""
    if not isinstance(value, list):
        return [], True
    reasons = [
        reason
        for reason in value
        if isinstance(reason, str) and re.fullmatch(r"[a-z0-9_]+", reason)
    ]
    return reasons, len(reasons) != len(value)


def _public_remote(manifest: dict[str, Any]) -> tuple[bool, str | None, bool]:
    """Return public remote fields and whether the source projection was unsafe."""
    declared_public = manifest.get("remote_public") is True
    if not declared_public:
        return False, None, False
    raw_remote = manifest.get("remote")
    remote, public = _sanitize_remote(raw_remote) if isinstance(raw_remote, str) else (None, False)
    is_public = public and remote == raw_remote
    return is_public, remote if is_public else None, not is_public


def _public_digest(value: object) -> tuple[str | None, bool]:
    """Keep only a well-formed remote SHA-256 digest in public output."""
    if value is None:
        return None, False
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        return value, False
    return None, True


def _public_subprojects(value: object) -> tuple[list[dict[str, Any]], bool]:
    """Project subproject rows without exposing unsafe declaration paths."""
    if not isinstance(value, list):
        return [], True
    rows: list[dict[str, Any]] = []
    sanitized = False
    for row in value:
        if not isinstance(row, dict):
            sanitized = True
            continue
        path, path_redacted = _public_relative_path(row.get("path"))
        identity, identity_redacted = _public_value(row.get("identity"), "[REDACTED_IDENTITY]")
        kind, kind_redacted = _public_value(row.get("kind"), "[REDACTED_KIND]")
        sanitized |= path_redacted or identity_redacted or kind_redacted
        rows.append({"path": path, "kind": kind, "identity": identity})
    return rows, sanitized


def _validate_manifest(manifest: object, require_complete: bool) -> list[str]:  # noqa: C901,PLR0912,PLR0915
    """Validate fields before a manifest can authorize public pass or restore."""
    if not isinstance(manifest, dict):
        return ["manifest_invalid"]
    errors: list[str] = []
    required = {
        "schema",
        "status",
        "workload_id",
        "commit",
        "tree",
        "parents",
        "ref_context",
        "remote_public",
        "remote",
        "remote_digest",
        "subprojects",
        "generated",
        "patch",
        "inventory_count",
        "inventory",
        "private_state_rejected",
        "reasons",
        "discrepancies",
    }
    if manifest.get("schema") != MANIFEST_SCHEMA or (require_complete and set(manifest) < required):
        errors.append("manifest_invalid")
        return errors
    if not isinstance(manifest.get("status"), str) or manifest.get("status") not in STATUS_VALUES:
        errors.append("manifest_invalid")
    if require_complete and manifest.get("status") != "pass":
        errors.append("manifest_invalid")
    if not isinstance(manifest.get("workload_id"), str):
        errors.append("manifest_invalid")
    if not _is_object_id(manifest.get("commit")) or not _is_object_id(manifest.get("tree")):
        errors.append("manifest_invalid")
    parents = manifest.get("parents")
    if not isinstance(parents, list) or any(not _is_object_id(parent) for parent in parents):
        errors.append("manifest_invalid")
    if manifest.get("ref_context") != "resolved_commit":
        errors.append("manifest_invalid")
    if not isinstance(manifest.get("remote_public"), bool):
        errors.append("manifest_invalid")
    elif manifest["remote_public"]:
        remote = manifest.get("remote")
        safe_remote, public = _sanitize_remote(remote) if isinstance(remote, str) else (None, False)
        if not public or remote != safe_remote:
            errors.append("manifest_invalid")
    elif manifest.get("remote") is not None:
        errors.append("manifest_invalid")
    if manifest.get("remote_digest") is not None and not _is_sha256(manifest.get("remote_digest")):
        errors.append("manifest_invalid")

    subprojects = manifest.get("subprojects")
    if not isinstance(subprojects, list):
        errors.append("manifest_invalid")
    else:
        for row in subprojects:
            if (
                not isinstance(row, dict)
                or not _is_safe_relative_path_text(row.get("path"))
                or row.get("kind") != "vendored"
                or not _is_sha256(row.get("identity"))
            ):
                errors.append("manifest_invalid")

    generated = manifest.get("generated")
    generated_paths: set[str] = set()
    if not isinstance(generated, list):
        errors.append("manifest_invalid")
    else:
        for row in generated:
            if not isinstance(row, dict) or not _is_safe_relative_path_text(row.get("path")):
                errors.append("manifest_invalid")
                continue
            if row["path"] in generated_paths:
                errors.append("manifest_invalid")
            generated_paths.add(row["path"])
            if not _is_sha256(row.get("sha256")) or any(
                key not in GENERATED_METADATA_KEYS | {"sha256"} or not isinstance(value, str)
                for key, value in row.items()
            ):
                errors.append("manifest_invalid")

    patch = manifest.get("patch")
    if patch is not None and (
        not isinstance(patch, dict)
        or not _is_safe_relative_path_text(patch.get("path"))
        or patch.get("path") in RESERVED_BUNDLE_MEMBERS
        or not _is_sha256(patch.get("sha256"))
        or patch.get("sha256") != patch.get("captured_diff_sha256")
        or patch.get("base_commit") != manifest.get("commit")
        or patch.get("base_tree") != manifest.get("tree")
        or not all(
            isinstance(patch.get(field), int) and patch.get(field) >= 0
            for field in ("size_bytes", "captured_diff_size_bytes")
        )
    ):
        errors.append("manifest_invalid")

    inventory = manifest.get("inventory")
    count = manifest.get("inventory_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        errors.append("manifest_invalid")
    if not isinstance(inventory, list) or count != len(inventory):
        errors.append("manifest_invalid")
    else:
        paths: set[str] = set()
        for row in inventory:
            path = row.get("path") if isinstance(row, dict) else None
            size = row.get("size_bytes") if isinstance(row, dict) else None
            valid = (
                isinstance(row, dict)
                and _is_safe_relative_path_text(path)
                and isinstance(row.get("mode"), str)
                and isinstance(size, int)
                and not isinstance(size, bool)
                and size >= 0
            )
            if not valid:
                errors.append("manifest_invalid")
                continue
            if path in paths:
                errors.append("manifest_invalid")
            paths.add(path)
    if require_complete and (
        manifest.get("private_state_rejected") is not False
        or manifest.get("reasons") != []
        or not isinstance(manifest.get("discrepancies"), list)
        or any(not isinstance(item, str) for item in manifest["discrepancies"])
    ):
        errors.append("manifest_invalid")
    return sorted(set(errors))


def public_status(manifest: dict[str, Any]) -> dict[str, Any]:
    """Sanitized projection: no private remote URL, paths, or credentials."""
    manifest = manifest if isinstance(manifest, dict) else {}
    reasons, sanitized = _public_reasons(manifest.get("reasons", []))

    workload_id, workload_redacted = _public_value(manifest.get("workload_id"), "[REDACTED]")
    sanitized |= workload_redacted
    raw_status = manifest.get("status")
    status_value = (
        raw_status
        if isinstance(raw_status, str)
        and raw_status
        in {
            "pass",
            "fail",
            "blocked",
        }
        else "blocked"
    )
    if status_value != raw_status:
        reasons.append("invalid_status")
        sanitized = True
    if _validate_manifest(manifest, require_complete=raw_status == "pass"):
        reasons.append("manifest_invalid")
        sanitized = True

    commit, commit_redacted = _public_value(manifest.get("commit"), "[REDACTED]")
    tree, tree_redacted = _public_value(manifest.get("tree"), "[REDACTED]")
    parents, parents_redacted = _public_sequence(manifest.get("parents"))
    sanitized |= commit_redacted or tree_redacted or parents_redacted
    remote_public, remote, remote_redacted = _public_remote(manifest)
    sanitized |= remote_redacted
    remote_digest, digest_redacted = _public_digest(manifest.get("remote_digest"))
    sanitized |= digest_redacted
    subprojects, subprojects_redacted = _public_subprojects(manifest.get("subprojects", []))
    sanitized |= subprojects_redacted
    raw_inventory_count = manifest.get("inventory_count", 0)
    if (
        isinstance(raw_inventory_count, int)
        and not isinstance(raw_inventory_count, bool)
        and raw_inventory_count >= 0
    ):
        inventory_count = raw_inventory_count
    else:
        inventory_count = 0
        sanitized = True

    status = {
        "schema": STATUS_SCHEMA,
        "workload_id": workload_id,
        "status": status_value,
        "commit": commit,
        "tree": tree,
        "parents": parents,
        "remote_public": remote_public,
        "remote": remote,
        "remote_digest": remote_digest,
        "subprojects": subprojects,
        "patch_present": manifest.get("patch") is not None,
        "inventory_count": inventory_count,
        "reasons": reasons,
    }
    if set(reasons) & BLOCKING_REASONS:
        status["status"] = "blocked"
    text = json.dumps(status, sort_keys=True)
    if sanitized or _contains_public_sensitive_text(text):
        status["status"] = "blocked"
        status["reasons"] = sorted({*status["reasons"], "sanitization_failed"})
    return status


def _verify_checksums(bundle_dir: Path, manifest: dict[str, Any]) -> list[str]:  # noqa: C901,PLR0912
    """Require one valid digest row for exactly every owned bundle member."""
    expected = {"manifest.json", "source.bundle"}
    patch = manifest.get("patch")
    if isinstance(patch, dict) and isinstance(patch.get("path"), str):
        expected.add(patch["path"])
    sums = bundle_dir / "SHA256SUMS"
    try:
        if sums.is_symlink() or not sums.is_file():
            return ["missing_checksums"]
        lines = sums.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return ["checksum_invalid"]
    rows: dict[str, str] = {}
    discrepancies: list[str] = []
    for line in lines:
        digest, separator, name = line.partition("  ")
        if not separator or not _is_sha256(digest) or not name or name in rows:
            discrepancies.append("checksum_invalid")
            continue
        if _safe_relative_path(bundle_dir, name) is None:
            discrepancies.append("path_escape")
            continue
        rows[name] = digest.lower()
    try:
        actual = {path.name for path in bundle_dir.iterdir() if path.name != "SHA256SUMS"}
        if any(
            path.name != "SHA256SUMS" and (path.is_symlink() or not path.is_file())
            for path in bundle_dir.iterdir()
        ):
            discrepancies.append("checksum_invalid")
    except OSError:
        return ["checksum_invalid"]
    if actual != expected or set(rows) != expected:
        discrepancies.append("checksum_incomplete")
    for name in expected & set(rows):
        target = _safe_relative_path(bundle_dir, name)
        try:
            if target is None or _sha256_file(target) != rows[name]:
                discrepancies.append("bundle_checksum_mismatch")
        except OSError:
            discrepancies.append("checksum_invalid")
    if isinstance(patch, dict) and isinstance(patch.get("path"), str):
        target = _safe_relative_path(bundle_dir, patch["path"])
        try:
            if target is None or not target.is_file():
                discrepancies.append("checksum_incomplete")
            elif rows.get(patch["path"]) != patch.get("sha256"):
                discrepancies.append("patch_digest_mismatch")
            elif target.stat().st_size != patch.get("size_bytes"):
                discrepancies.append("patch_size_mismatch")
        except OSError:
            discrepancies.append("checksum_invalid")
    return sorted(set(discrepancies))


def _compare_generated_rows(manifest: dict[str, Any], workdir: Path) -> list[str]:
    discrepancies: list[str] = []
    rows = manifest.get("generated", [])
    if not isinstance(rows, list):
        return ["manifest_invalid: generated rows"]
    for row in rows:
        if not isinstance(row, dict):
            discrepancies.append("manifest_invalid: generated row")
            continue
        path = _safe_relative_path(workdir, row.get("path"))
        if path is None:
            discrepancies.append("path_escape: generated path")
        else:
            try:
                matches = path.is_file() and _sha256_file(path) == row.get("sha256")
            except OSError:
                matches = False
            if not matches:
                discrepancies.append("generated_drift")
    return discrepancies


def _compare_subproject_rows(manifest: dict[str, Any], workdir: Path) -> list[str]:
    discrepancies: list[str] = []
    rows = manifest.get("subprojects", [])
    if not isinstance(rows, list):
        return ["manifest_invalid: subproject rows"]
    for row in rows:
        if not isinstance(row, dict):
            discrepancies.append("manifest_invalid: subproject row")
            continue
        path = _safe_relative_path(workdir, row.get("path"))
        if path is None:
            discrepancies.append("path_escape: subproject path")
            continue
        if row.get("kind") == "git_rejected":
            discrepancies.append("git_subproject_rejected")
            continue
        if row.get("kind") != "vendored":
            discrepancies.append("manifest_invalid")
            continue
        try:
            matches = path.exists() and _directory_digest(path) == row.get("identity")
        except (OSError, RuntimeError, ValueError):
            matches = False
            discrepancies.append("path_escape: subproject contents")
        if not matches:
            discrepancies.append("subproject_mismatch")
    return discrepancies


def _compare_admitted_patch(bundle_dir: Path, manifest: dict[str, Any], workdir: Path) -> list[str]:  # noqa: C901
    patch = manifest.get("patch")
    if not patch:
        return []
    if not isinstance(patch, dict):
        return ["manifest_invalid: patch"]
    patch_path = _safe_relative_path(bundle_dir, patch.get("path"))
    if patch_path is None:
        return ["path_escape: patch path"]
    try:
        patch_bytes = patch_path.read_bytes()
    except OSError:
        return ["admitted_patch_missing"]
    discrepancies = []
    digest = _digest_bytes(patch_bytes)
    if digest != patch.get("sha256"):
        discrepancies.append("patch_digest_mismatch")
    if len(patch_bytes) != patch.get("size_bytes"):
        discrepancies.append("patch_size_mismatch")
    if digest != patch.get("captured_diff_sha256"):
        discrepancies.append("patch_not_bound")
    if discrepancies:
        return discrepancies
    try:
        if _capture_worktree_diff(workdir):
            return discrepancies + ["restored_worktree_not_clean"]
        checked = _run_git(workdir, "apply", "--index", "--check", str(patch_path), check=False)
        if checked.returncode:
            return discrepancies + ["admitted_patch_not_applicable"]
        applied = _run_git(workdir, "apply", "--index", str(patch_path), check=False)
        if applied.returncode:
            return discrepancies + ["admitted_patch_replay_failed"]
        replayed = _capture_worktree_diff(workdir)
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return discrepancies + ["patch_state_unavailable"]
    return discrepancies + ([] if replayed == patch_bytes else ["admitted_patch_replay_mismatch"])


def _compare_restored(bundle_dir: Path, manifest: dict[str, Any], workdir: Path) -> list[str]:
    discrepancies: list[str] = []
    try:
        head = _run_git(workdir, "rev-parse", "HEAD^{commit}", check=False)
        tree = _run_git(workdir, "rev-parse", "HEAD^{tree}", check=False)
        status = _run_git(workdir, "status", "--porcelain", check=False)
    except (OSError, subprocess.SubprocessError):
        return ["restore_identity_unavailable"]
    if head.returncode or tree.returncode or status.returncode:
        return ["restore_identity_unavailable"]
    head_value = head.stdout.strip()
    tree_value = tree.stdout.strip()
    if head_value != manifest["commit"]:
        discrepancies.append("commit_mismatch")
    if tree_value != manifest["tree"]:
        discrepancies.append("tree_mismatch")
    if manifest.get("patch") is None and status.stdout.strip():
        discrepancies.append("restored_worktree_not_clean")
    discrepancies.extend(_compare_admitted_patch(bundle_dir, manifest, workdir))
    discrepancies.extend(_compare_generated_rows(manifest, workdir))
    discrepancies.extend(_compare_subproject_rows(manifest, workdir))
    return discrepancies


def _restore_and_compare(bundle_dir: Path, manifest: dict[str, Any], workdir: Path) -> list[str]:
    bundle_path = _safe_relative_path(bundle_dir, "source.bundle")
    if bundle_path is None:
        return ["path_escape: bundle file"]
    try:
        if not bundle_path.is_file():
            return ["missing_bundle_file"]
        if _path_has_symlink_component(workdir) or workdir.exists() or not workdir.parent.is_dir():
            return ["restore_workdir_unavailable"]
        clone = subprocess.run(
            ["git", "clone", "--no-local", "--quiet", str(bundle_path), str(workdir)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ["bundle_clone_failed"]
    if clone.returncode != 0:
        return ["bundle_clone_failed"]
    return _compare_restored(bundle_dir, manifest, workdir)


def verify_bundle(bundle_dir: Path, workdir: Path) -> dict[str, Any]:
    """Restore the bundle into a fresh repository and compare recorded identities."""
    if not isinstance(bundle_dir, Path) or not isinstance(workdir, Path):
        return {
            "schema": VERIFICATION_SCHEMA,
            "status": "blocked",
            "reasons": ["restore_workdir_unavailable"],
            "discrepancies": ["bundle and restore paths must be pathlib.Path values"],
        }
    try:
        unsafe_bundle = _path_has_symlink_component(bundle_dir)
        bundle_is_dir = bundle_dir.is_dir()
    except OSError:
        unsafe_bundle, bundle_is_dir = True, False
    if unsafe_bundle:
        return {
            "schema": VERIFICATION_SCHEMA,
            "status": "blocked",
            "reasons": ["path_escape"],
            "discrepancies": ["bundle directory is unsafe"],
        }
    if not bundle_is_dir:
        return {
            "schema": VERIFICATION_SCHEMA,
            "status": "blocked",
            "reasons": ["bundle_unavailable"],
            "discrepancies": ["bundle directory is unavailable"],
        }
    manifest_path = _safe_relative_path(bundle_dir, "manifest.json")
    try:
        manifest_exists = manifest_path is not None and manifest_path.is_file()
    except OSError:
        manifest_exists = False
    if manifest_path is None or not manifest_exists:
        return {
            "schema": VERIFICATION_SCHEMA,
            "status": "blocked",
            "reasons": ["path_escape" if manifest_path is None else "manifest_unreadable"],
            "discrepancies": ["unsafe or missing manifest path"],
        }
    manifest, err = _read_json(manifest_path)
    if err or manifest is None:
        return {
            "schema": VERIFICATION_SCHEMA,
            "status": "blocked",
            "reasons": ["manifest_unreadable"],
            "discrepancies": [err or "manifest is not an object"],
        }
    validation = _validate_manifest(manifest, require_complete=True)
    if validation:
        return {
            "schema": VERIFICATION_SCHEMA,
            "status": "blocked",
            "reasons": validation,
            "discrepancies": ["manifest failed complete-shape validation"],
        }
    discrepancies = _verify_checksums(bundle_dir, manifest)
    if not discrepancies:
        discrepancies.extend(_restore_and_compare(bundle_dir, manifest, workdir))
    reason_codes = sorted({d.split(":", 1)[0] for d in discrepancies})
    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "blocked"
        if set(reason_codes) & BLOCKING_REASONS
        else ("fail" if discrepancies else "pass"),
        "workload_id": manifest.get("workload_id"),
        "commit": manifest.get("commit"),
        "reasons": reason_codes,
        "discrepancies": discrepancies,
    }


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return None, str(exc)
    return (data, None) if isinstance(data, dict) else (None, "content is not a JSON object")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns 0 for pass, 1 for fail, 2 for blocked."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--workload-id", default="unnamed")
    parser.add_argument("--patch", type=Path, default=None)
    parser.add_argument("--subproject", action="append", default=[])
    parser.add_argument(
        "--generated",
        action="append",
        default=[],
        metavar="REPO_RELATIVE_PATH",
        help="tracked generated-source path relative to --repo; repeatable",
    )
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    args = parser.parse_args(argv)

    if args.verify:
        if args.bundle is None:
            print("--verify requires --bundle", file=sys.stderr)
            return 2
        workdir = args.workdir or Path(tempfile.mkdtemp(prefix="source_bundle_verify_")) / "restore"
        report = verify_bundle(args.bundle, workdir)
    elif args.export:
        if args.repo is None or args.out is None:
            print("--export requires --repo and --out", file=sys.stderr)
            return 2
        report = public_status(
            export_bundle(
                args.repo,
                args.out,
                args.workload_id,
                args.patch,
                args.subproject,
                [{"path": path} for path in args.generated],
            )
        )
    else:
        print("choose --export or --verify", file=sys.stderr)
        return 2
    if args.format == "json":
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(f"Status: {report['status'].upper()}\n")
        for reason in report.get("reasons", []):
            sys.stdout.write(f"- {reason}\n")
    return {"pass": 0, "fail": 1}.get(report["status"], 2)


if __name__ == "__main__":
    sys.exit(main())
