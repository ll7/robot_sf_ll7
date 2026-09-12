#!/usr/bin/env python3
"""Export and verify restorable source bundles for admitted workloads (#8851).

A bundle records the exact source identity of one workload: repository URL
classification, commit, tree, parents, ref context, vendored subproject
revisions, generated-source provenance, admitted patch identity, and a compact
tracked-file inventory. Dirty or untracked state is rejected unless an explicit
patch is admitted and checksum-bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "source_bundle_manifest.v1"
STATUS_SCHEMA = "source_bundle_status.v1"
VERIFICATION_SCHEMA = "source_bundle_verification.v1"

PUBLIC_HOSTS = frozenset({"github.com", "gitlab.com", "codeberg.org"})
CREDENTIAL_RE = re.compile(r"://[^/@\s]+@")
PRIVATE_PATH_RE = re.compile(
    r"(?:^|[\s\"'=])(/(?:home|root|private|scratch|work)/|[a-zA-Z]:[/\\]Users)", re.IGNORECASE
)
BLOCKING_REASONS = frozenset(
    "dirty_tree_not_admitted untracked_state_not_admitted shallow_clone_rejected missing_object "
    "credential_in_remote subproject_missing generated_file_missing patch_not_found".split()
)


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


def _sanitize_remote(url: str) -> tuple[str | None, bool]:
    """Return (sanitized_url_or_None, is_public); private or credentialed URLs stay hidden."""
    if not url:
        return None, False
    had_credentials = bool(CREDENTIAL_RE.search(url))
    sanitized = CREDENTIAL_RE.sub("://", url)
    host_match = re.search(r"://([^/@:\s]+)", sanitized) or re.search(r"@([^/:\s]+):", sanitized)
    host = host_match.group(1) if host_match else ""
    public = (not had_credentials) and host in PUBLIC_HOSTS and sanitized.startswith("https://")
    return (sanitized if public else None), public


def _directory_digest(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or path.is_file():
            rows.append(f"{relative} {_sha256_file(path) if path.is_file() else 'symlink'}")
    return _digest_bytes("\n".join(rows).encode("utf-8"))


def _resolve_identity(
    repo: Path, ref: str, reasons: list[str], disc: list[str]
) -> tuple[str, str, list[str]] | None:
    commit_proc = _run_git(repo, "rev-parse", f"{ref}^{{commit}}", check=False)
    if commit_proc.returncode != 0:
        reasons.append("missing_object")
        disc.append(f"{ref} does not resolve to a commit")
        return None
    commit = commit_proc.stdout.strip()
    tree = _run_git(repo, "rev-parse", f"{commit}^{{tree}}").stdout.strip()
    parents = _run_git(repo, "rev-list", "--parents", "-n", "1", commit).stdout.split()[1:]
    return commit, tree, parents


def _remote_identity(repo: Path, disc: list[str]) -> dict[str, Any]:
    remote_url = _run_git(repo, "config", "--get", "remote.origin.url", check=False).stdout.strip()
    sanitized, public = _sanitize_remote(remote_url)
    if remote_url and not public and CREDENTIAL_RE.search(remote_url):
        disc.append("remote URL contains credentials; recorded as private digest only")
    return {
        "remote_public": public,
        "remote": sanitized,
        "remote_digest": _digest_bytes(sanitized.encode("utf-8")) if sanitized else None,
    }


def _worktree_state(repo: Path) -> tuple[list[str], list[str]]:
    status = _run_git(repo, "status", "--porcelain").stdout.strip()
    tracked = [line for line in status.splitlines() if not line.startswith("??")]
    untracked = [line for line in status.splitlines() if line.startswith("??")]
    return tracked, untracked


def _patch_identity(patch_path: Path | None, reasons: list[str]) -> dict[str, Any] | None:
    if patch_path is None:
        return None
    if not patch_path.is_file():
        reasons.append("patch_not_found")
        return None
    return {
        "path": patch_path.name,
        "sha256": _sha256_file(patch_path),
        "size_bytes": patch_path.stat().st_size,
    }


def _subproject_rows(repo: Path, paths: list[str], reasons: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in paths:
        path = repo / relative
        if not path.exists():
            reasons.append("subproject_missing")
            rows.append({"path": relative, "kind": "missing", "identity": None})
        elif (path / ".git").exists():
            head = _run_git(path, "rev-parse", "HEAD^{commit}", check=False)
            remote = _run_git(path, "config", "--get", "remote.origin.url", check=False)
            url, public = (
                _sanitize_remote(remote.stdout.strip()) if remote.returncode == 0 else (None, False)
            )
            rows.append(
                {
                    "path": relative,
                    "kind": "git",
                    "identity": head.stdout.strip() if head.returncode == 0 else None,
                    "remote_public": public,
                    "remote": url,
                }
            )
        else:
            rows.append({"path": relative, "kind": "vendored", "identity": _directory_digest(path)})
    return rows


def _generated_rows(
    repo: Path, entries: list[dict[str, str]], reasons: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        path = repo / entry["path"]
        digest = _sha256_file(path) if path.is_file() else None
        if digest is None:
            reasons.append("generated_file_missing")
        rows.append({**entry, "sha256": digest})
    return rows


def _inventory(repo: Path, commit: str) -> list[dict[str, Any]]:
    rows = []
    for line in _run_git(repo, "ls-tree", "-r", "-l", commit).stdout.splitlines():
        meta, _, path = line.partition("\t")
        fields = meta.split()
        if len(fields) >= 4:
            rows.append({"path": path, "mode": fields[0], "size_bytes": int(fields[3])})
    return rows


def _create_bundle(
    repo: Path, out_dir: Path, ref: str, reasons: list[str], disc: list[str]
) -> None:
    if "shallow_clone_rejected" in reasons:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    created = _run_git(repo, "bundle", "create", str(out_dir / "source.bundle"), ref, check=False)
    if created.returncode != 0:
        reasons.append("missing_object")
        disc.append(created.stderr.strip().splitlines()[-1] if created.stderr else "")


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
    if (repo / ".git" / "shallow").exists():
        reasons.append("shallow_clone_rejected")
    tracked_dirty, untracked = _worktree_state(repo)
    if tracked_dirty:
        reasons.append("dirty_tree_not_admitted")
        disc.append(f"tracked changes present: {len(tracked_dirty)}")
    if untracked:
        reasons.append("untracked_state_not_admitted")
        disc.append(f"untracked entries present: {len(untracked)}")
    patch_identity = _patch_identity(patch_path, reasons)
    if patch_identity is not None:
        reasons = [r for r in reasons if r != "dirty_tree_not_admitted"]
        if tracked_dirty:
            disc.append("admitted patch covers the tracked changes")
    subproject_rows = _subproject_rows(repo, subprojects or [], reasons)
    generated_rows = _generated_rows(repo, generated or [], reasons)
    inventory = _inventory(repo, commit)
    _create_bundle(repo, out_dir, ref, reasons, disc)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "workload_id": workload_id,
        "commit": commit,
        "tree": tree,
        "parents": parents,
        "ref_context": _run_git(
            repo, "name-rev", "--name-only", commit, check=False
        ).stdout.strip(),
        **_remote_identity(repo, disc),
        "subprojects": subproject_rows,
        "generated": generated_rows,
        "patch": patch_identity,
        "inventory_count": len(inventory),
        "inventory": inventory,
        "private_state_rejected": bool(
            {"dirty_tree_not_admitted", "untracked_state_not_admitted"} & set(reasons)
        ),
        "status": "blocked" if set(reasons) & BLOCKING_REASONS else ("fail" if reasons else "pass"),
        "reasons": sorted(set(reasons)),
        "discrepancies": disc,
    }
    if manifest["status"] == "pass":
        if patch_path is not None and patch_path.is_file():
            (out_dir / patch_path.name).write_bytes(patch_path.read_bytes())
        _write_bundle_files(out_dir, manifest)
    return manifest


def _write_bundle_files(out_dir: Path, manifest: dict[str, Any]) -> None:
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = [
        f"{_sha256_file(path)}  {path.name}"
        for path in sorted(out_dir.iterdir())
        if path.name != "SHA256SUMS"
    ]
    (out_dir / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def public_status(manifest: dict[str, Any]) -> dict[str, Any]:
    """Sanitized projection: no private remote URL, paths, or credentials."""
    status = {
        "schema": STATUS_SCHEMA,
        "workload_id": manifest.get("workload_id"),
        "status": manifest.get("status"),
        "commit": manifest.get("commit"),
        "tree": manifest.get("tree"),
        "parents": manifest.get("parents"),
        "remote_public": manifest.get("remote_public"),
        "remote": manifest.get("remote") if manifest.get("remote_public") else None,
        "subprojects": [
            {"path": row["path"], "kind": row["kind"], "identity": row["identity"]}
            for row in manifest.get("subprojects", [])
        ],
        "patch_present": manifest.get("patch") is not None,
        "inventory_count": manifest.get("inventory_count", 0),
        "reasons": manifest.get("reasons", []),
    }
    text = json.dumps(status)
    if PRIVATE_PATH_RE.search(text) or CREDENTIAL_RE.search(text):
        status["status"] = "blocked"
        status["reasons"] = sorted({*status["reasons"], "sanitization_failed"})
    return status


def _verify_checksums(bundle_dir: Path) -> list[str]:
    discrepancies = []
    sums = bundle_dir / "SHA256SUMS"
    for row in sums.read_text(encoding="utf-8").splitlines() if sums.is_file() else []:
        digest, _, name = row.partition("  ")
        target = bundle_dir / name
        if not target.is_file() or _sha256_file(target) != digest:
            discrepancies.append(f"bundle_checksum_mismatch: {name}")
    return discrepancies


def _compare_restored(bundle_dir: Path, manifest: dict[str, Any], workdir: Path) -> list[str]:
    discrepancies: list[str] = []
    head = _run_git(workdir, "rev-parse", "HEAD^{commit}").stdout.strip()
    tree = _run_git(workdir, "rev-parse", "HEAD^{tree}").stdout.strip()
    if head != manifest["commit"]:
        discrepancies.append(f"commit_mismatch: {head}")
    if tree != manifest["tree"]:
        discrepancies.append(f"tree_mismatch: {tree}")
    if _run_git(workdir, "status", "--porcelain").stdout.strip():
        discrepancies.append("restored_worktree_not_clean")
    for row in manifest.get("generated", []):
        path = workdir / row["path"]
        if not path.is_file() or _sha256_file(path) != row["sha256"]:
            discrepancies.append(f"generated_drift: {row['path']}")
    for row in manifest.get("subprojects", []):
        path = workdir / row["path"]
        vendored_mismatch = row["kind"] == "vendored" and (
            not path.exists() or _directory_digest(path) != row["identity"]
        )
        if vendored_mismatch:
            discrepancies.append(f"subproject_mismatch: {row['path']}")
    patch = manifest.get("patch")
    if patch and (bundle_dir / patch["path"]).is_file():
        applied = _run_git(
            workdir, "apply", "--check", str(bundle_dir / patch["path"]), check=False
        )
        if applied.returncode != 0:
            discrepancies.append("admitted_patch_not_applicable")
    return discrepancies


def _restore_and_compare(bundle_dir: Path, manifest: dict[str, Any], workdir: Path) -> list[str]:
    bundle_path = bundle_dir / "source.bundle"
    if not bundle_path.is_file():
        return ["missing_bundle_file"]
    workdir.mkdir(parents=True, exist_ok=True)
    clone = subprocess.run(
        ["git", "clone", "--no-local", "--quiet", str(bundle_path), str(workdir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if clone.returncode != 0:
        return ["bundle_clone_failed"]
    return _compare_restored(bundle_dir, manifest, workdir)


def verify_bundle(bundle_dir: Path, workdir: Path) -> dict[str, Any]:
    """Restore the bundle into a fresh repository and compare recorded identities."""
    manifest, err = _read_json(bundle_dir / "manifest.json")
    if err or manifest is None or manifest.get("schema") != MANIFEST_SCHEMA:
        return {
            "schema": VERIFICATION_SCHEMA,
            "status": "blocked",
            "reasons": ["manifest_invalid"],
            "discrepancies": [err or "unsupported schema"],
        }
    discrepancies = _verify_checksums(bundle_dir) + _restore_and_compare(
        bundle_dir, manifest, workdir
    )
    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "fail" if discrepancies else "pass",
        "workload_id": manifest.get("workload_id"),
        "commit": manifest.get("commit"),
        "reasons": sorted({d.split(":")[0] for d in discrepancies}),
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
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    args = parser.parse_args(argv)

    if args.verify:
        if args.bundle is None:
            print("--verify requires --bundle", file=sys.stderr)
            return 2
        workdir = args.workdir or Path(tempfile.mkdtemp(prefix="source_bundle_verify_"))
        report = verify_bundle(args.bundle, workdir)
    elif args.export:
        if args.repo is None or args.out is None:
            print("--export requires --repo and --out", file=sys.stderr)
            return 2
        report = public_status(
            export_bundle(args.repo, args.out, args.workload_id, args.patch, args.subproject)
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
