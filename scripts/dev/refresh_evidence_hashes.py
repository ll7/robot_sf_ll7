"""Refresh stale artifact hashes declared in evidence-registry JSON files.
Every PR that edits a file pinned by an evidence-registry ``sha256`` declaration
(e.g. ``docs/RELEASE.md`` pinned by the issue_4683 assurance-case example) trips
``evidence-registry-ratchet`` with ``artifact_hash_mismatch``. The fix is a
one-line declared-hash refresh, not a baseline regeneration. This helper makes
that refresh deterministic:

- ``--check`` (default): report stale hashes in the tracked release-assurance
  example. Use ``--path`` to check another linter-reported evidence file.
- ``--write``: rewrite stale declared hashes in the declaring evidence files.
  Use ``--path`` to restrict the rewrite to named evidence files.

Fail-closed rules: baseline files (``scripts/validation/evidence_registry_baseline*``)
are never touched; files that do not parse as JSON are skipped; ambiguous
declarations (same artifact pinned twice with different values in one file)
are skipped; every rewrite is re-parsed before reporting success.
"""

# evidence-writer-exempt: hash-refresh tooling; declared-hash writes target
# tracked evidence JSON only, never baseline files (refused by path guard).

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
HASH_KEYS = ("sha256", "source_sha256")
ARTIFACT_PATH_KEYS = (
    "artifact_path",
    "artifact_uri",
    "file",
    "filename",
    "path",
    "source_path",
)
BASELINE_PREFIXES = ("scripts/validation/evidence_registry_baseline",)
DEFAULT_EXAMPLE = Path("docs/context/evidence/issue_4683_release_assurance_case_example.json")
DEFAULT_LINTER = Path("scripts/tools/lint_evidence_registry.py")
DEFAULT_REGISTRY_ROOT = Path("docs/context/evidence")
DEFAULT_DISPOSITION = Path("docs/context/evidence/evidence_registry_dispositions.yaml")


def _repo_root() -> Path:
    """Return the repository root for the current checkout."""
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("could not resolve repository root via git")
    return Path(proc.stdout.strip())


def _is_tracked(repo_root: Path, rel: str) -> bool:
    """Return whether a repo-relative path is tracked by Git."""
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def _run_linter(repo_root: Path) -> dict[str, Any]:
    """Run the canonical evidence-registry linter and return its report."""
    cmd = [
        sys.executable,
        str(repo_root / DEFAULT_LINTER),
        "--repo-root",
        str(repo_root),
        "--registry-root",
        str(DEFAULT_REGISTRY_ROOT),
        "--disposition-file",
        str(DEFAULT_DISPOSITION),
    ]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"linter exited {proc.returncode}: {proc.stderr[:1000]}")
    return json.loads(proc.stdout)


def _iter_mappings(value: Any):
    """Yield every JSON object in a document, depth-first."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_mappings(child)


def _actual_hash(repo_root: Path, artifact: str) -> str | None:
    """Return the SHA-256 of a tracked artifact, or None when unavailable."""
    candidate = (repo_root / artifact).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        return None
    if not candidate.is_file() or not _is_tracked(repo_root, artifact):
        return None
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def _mapping_candidates(mapping: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return (hash_key, declared, artifact) triples for one JSON object."""
    hash_keys = [k for k in HASH_KEYS if isinstance(mapping.get(k), str)]
    artifacts = [
        str(mapping[k])
        for k in ARTIFACT_PATH_KEYS
        if isinstance(mapping.get(k), str)
        and str(mapping[k])
        and not str(mapping[k]).startswith("configs/")
    ]
    triples = []
    for hash_key in hash_keys:
        declared = str(mapping[hash_key])
        if not SHA256_RE.fullmatch(declared):
            continue
        triples.extend((hash_key, declared, artifact) for artifact in artifacts)
    return triples


def _stale_declarations(
    repo_root: Path, evidence_path: Path
) -> tuple[list[dict[str, str]], list[str]]:
    """Find stale hash declarations in one evidence file.

    Returns (stale_entries, skipped_reasons). Each stale entry names the JSON
    object path, hash key, artifact, declared value, and actual value.
    """
    try:
        document = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"{evidence_path}: unreadable ({exc}); skipped"]
    candidates: dict[str, list[dict[str, str]]] = {}
    skipped: list[str] = []
    for mapping in _iter_mappings(document):
        if not isinstance(mapping, dict):
            continue
        for hash_key, declared, artifact in _mapping_candidates(mapping):
            actual = _actual_hash(repo_root, artifact)
            if actual is None or actual.lower() == declared.lower():
                continue
            dedupe = f"{artifact}\0{hash_key}"
            candidates.setdefault(dedupe, []).append(
                {
                    "artifact": artifact,
                    "hash_key": hash_key,
                    "declared": declared,
                    "actual": actual,
                }
            )
    stale: list[dict[str, str]] = []
    for entries in candidates.values():
        if len({entry["declared"].lower() for entry in entries}) > 1:
            skipped.append(
                f"{evidence_path}: {entries[0]['artifact']} pinned twice with "
                "different values; skipped as ambiguous"
            )
            continue
        stale.extend(entries)
    return stale, skipped


def _rewrite_file(evidence_path: Path, stale: list[dict[str, str]]) -> int:
    """Rewrite stale declared hashes by exact value replacement."""
    text = evidence_path.read_text(encoding="utf-8")
    replacements: list[tuple[str, str]] = []
    for entry in stale:
        occurrences = text.count(entry["declared"])
        if occurrences != 1:
            print(
                f"  skip {entry['artifact']}: declared value occurs {occurrences}x (need exactly 1)"
            )
            return 0
        replacements.append((entry["declared"], entry["actual"]))
    for declared, actual in replacements:
        text = text.replace(declared, actual)
    if replacements:
        evidence_path.write_text(text, encoding="utf-8")
        json.loads(evidence_path.read_text(encoding="utf-8"))
    return len(replacements)


def _mismatch_files(repo_root: Path) -> list[Path]:
    """Return evidence files with linter artifact_hash_mismatch findings."""
    report = _run_linter(repo_root)
    paths: list[str] = []
    findings = report.get("findings", report.get("issues", []))
    if isinstance(findings, dict):
        items = findings.items()
    elif isinstance(findings, list):
        items = [(f.get("path", ""), f) for f in findings if isinstance(f, dict)]
    else:
        items = []
    for path, finding in items:
        if isinstance(finding, dict):
            codes = finding.get("codes", {finding.get("code", ""): 1})
        else:
            codes = {}
        if "artifact_hash_mismatch" in codes and path:
            paths.append(path)
    ordered = sorted(dict.fromkeys(paths))
    return [repo_root / p for p in ordered]


def main(argv: list[str] | None = None) -> int:
    """Run the hash-refresh check or write pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Check stale declared hashes without writing (default).",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Rewrite stale declared hashes (default is check-only).",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Restrict to a repo-relative evidence file (repeatable).",
    )
    args = parser.parse_args(argv)
    repo_root = _repo_root()
    targets = [repo_root / p for p in args.path] if args.path else [repo_root / DEFAULT_EXAMPLE]
    for target in targets:
        rel = (
            target.relative_to(repo_root).as_posix()
            if target.is_relative_to(repo_root)
            else str(target)
        )
        if rel.startswith(BASELINE_PREFIXES):
            print(f"refusing to touch baseline file: {rel}")
            return 2
    total_stale = 0
    skipped_count = 0
    rewrite_failures = 0
    for target in targets:
        stale, skipped = _stale_declarations(repo_root, target)
        for reason in skipped:
            print(reason)
        skipped_count += len(skipped)
        rel = target.relative_to(repo_root).as_posix()
        if not stale:
            print(f"{rel}: no stale hashes")
            continue
        for entry in stale:
            print(
                f"{rel}: {entry['artifact']} {entry['hash_key']} "
                f"{entry['declared'][:12]} -> {entry['actual'][:12]}"
            )
        total_stale += len(stale)
        if args.write:
            refreshed = _rewrite_file(target, stale)
            print(f"{rel}: refreshed {refreshed}/{len(stale)} declaration(s)")
            rewrite_failures += len(stale) - refreshed
    if total_stale and not args.write:
        print("\nRefresh with: uv run python scripts/dev/refresh_evidence_hashes.py --write")
        return 1
    if skipped_count or rewrite_failures:
        print(
            "\nERROR: one or more stale declarations could not be refreshed "
            "unambiguously; no successful write is reported.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
