#!/usr/bin/env python3
"""License-aware assistant for staging pinned external git repositories.

The helper deliberately stages external code as local-only git clones rather than vendoring it into
Robot SF. Each registered repository must declare an explicit pinned commit, license decision,
redistribution decision, intended use, and validation command before it can be staged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_ROOT = REPO_ROOT / "third_party" / "external_repos"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "output" / "external_repos" / "manifests"


class ExternalRepoError(RuntimeError):
    """Raised when an external repository cannot be safely staged or validated."""


@dataclass(frozen=True)
class RepoSpec:
    """Static registry entry for one supported external repository."""

    name: str
    title: str
    upstream_url: str
    fork_url: str | None
    pinned_sha: str
    stage_path: Path
    source_access_date: str
    license_note: str
    license_compatibility_decision: str
    redistribution_decision: str
    intended_use: str
    validation_command: str
    related_issues: tuple[int, ...] = ()


REPOS: tuple[RepoSpec, ...] = (
    RepoSpec(
        name="sicnav",
        title="Safe Interactive Crowd Navigation reference implementation",
        upstream_url="https://github.com/sepsamavi/safe-interactive-crowdnav",
        fork_url=None,
        pinned_sha="c702fb8ac9ba6439ca61da7dde68b8524bbc6a1f",
        stage_path=DEFAULT_STAGE_ROOT / "sicnav",
        source_access_date="2026-06-21",
        license_note="MIT License, as reported by GitHub repository license metadata.",
        license_compatibility_decision="compatible for local research/reference staging",
        redistribution_decision=(
            "public fork allowed by MIT after maintainer review; this registry stages from "
            "upstream until an ll7 fork is created"
        ),
        intended_use=(
            "Reference checkout for SICNav wrapper/provenance work; wrapper smoke and benchmark "
            "eligibility remain separate gates."
        ),
        validation_command=(
            "uv run pytest tests/baselines/test_external_mpc_wrappers.py "
            "-k sicnav_skip_without_external_repo -q"
        ),
        related_issues=(3347, 3366),
    ),
    RepoSpec(
        name="crowdnav_pred_attng",
        title=(
            "CrowdNav_Prediction_AttnGraph: intention-aware crowd navigation with an "
            "attention-based interaction graph (ICRA 2023) reference implementation"
        ),
        upstream_url="https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph",
        fork_url=None,
        pinned_sha="390773137be04ed14e27620dc5fd7c5e1a5b1f62",
        stage_path=DEFAULT_STAGE_ROOT / "crowdnav_pred_attng",
        source_access_date="2026-07-08",
        license_note=(
            "MIT License, as reported by GitHub repository license metadata (LICENSE file at the "
            "pinned commit)."
        ),
        license_compatibility_decision=(
            "compatible for local research/reference staging; MIT permits academic benchmark reuse"
        ),
        redistribution_decision=(
            "public fork allowed by MIT after maintainer review; this registry stages from "
            "upstream until an ll7 fork is created"
        ),
        intended_use=(
            "Reference checkout for the CrowdNav_Prediction_AttnGraph learned-baseline feasibility "
            "smoke (issue #4871). Only the PyTorch RL navigation policy path is exercised; the "
            "TensorFlow GST trajectory-predictor path and OpenAI Baselines/crowd_sim stack are out "
            "of scope for the smoke. Benchmark roster eligibility remains a separate maintainer "
            "gate."
        ),
        validation_command=(
            "uv run pytest tests/planner/test_crowdnav_pred_attng.py "
            "-k crowdnav_pred_attng_skip_without_external_repo -q"
        ),
        related_issues=(4871, 1617),
    ),
)


def list_repos() -> tuple[RepoSpec, ...]:
    """Return the supported external repository registry."""
    return REPOS


def _get_repo(name: str) -> RepoSpec:
    """Resolve a repository name or raise a user-facing error."""
    for repo in REPOS:
        if repo.name == name:
            return repo
    supported = ", ".join(repo.name for repo in REPOS) or "none registered yet"
    raise ExternalRepoError(f"Unknown repository '{name}'. Supported repositories: {supported}")


def _run_git(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a git command and return captured text output."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def _relative_to_repo(path: Path, repo_root: Path) -> Path | None:
    """Return a repo-relative path when path is inside repo_root."""
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None


def _git_check_ignored(path: Path, repo_root: Path) -> bool:
    """Return whether git ignore rules cover a repo-local path."""
    rel = _relative_to_repo(path, repo_root)
    if rel is None:
        return True
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", rel.as_posix()],
        cwd=repo_root,
        check=False,
    )
    return result.returncode == 0


def _ensure_stage_path_ignored(stage_path: Path, repo_root: Path) -> None:
    """Fail closed if the clone destination is repo-local and not gitignored."""
    if _relative_to_repo(stage_path, repo_root) is None:
        return
    if not _git_check_ignored(stage_path, repo_root):
        rel = _relative_to_repo(stage_path, repo_root)
        raise ExternalRepoError(
            "External repository destination is not covered by gitignore: "
            f"{rel}. Add an explicit ignore rule before staging."
        )


def _assert_license_decisions_recorded(spec: RepoSpec) -> None:
    """Fail closed when the registry entry does not record license decisions."""
    missing: list[str] = []
    if not spec.source_access_date:
        missing.append("source_access_date")
    if not spec.license_note:
        missing.append("license_note")
    if not spec.license_compatibility_decision:
        missing.append("license_compatibility_decision")
    if not spec.redistribution_decision:
        missing.append("redistribution_decision")
    if not spec.intended_use:
        missing.append("intended_use")
    if not spec.validation_command:
        missing.append("validation_command")
    if missing:
        raise ExternalRepoError(
            f"{spec.name} is missing required registry fields: {', '.join(missing)}"
        )


def _assert_pinned_sha_reachable(spec: RepoSpec) -> None:
    """Fail closed unless git can fetch the pinned commit from the declared source."""
    with tempfile.TemporaryDirectory(prefix="robot-sf-external-repo-") as temp_dir:
        probe = Path(temp_dir)
        _run_git(["init"], cwd=probe)
        _run_git(["remote", "add", "origin", spec.fork_url or spec.upstream_url], cwd=probe)
        result = _run_git(
            ["fetch", "--depth=1", "origin", spec.pinned_sha],
            cwd=probe,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise ExternalRepoError(
                f"Pinned SHA is unreachable for {spec.name}: {spec.pinned_sha}. {detail}"
            )


def _checkout_pinned_clone(spec: RepoSpec, stage_path: Path) -> None:
    """Clone or update stage_path and detach it at the pinned commit."""
    source_url = spec.fork_url or spec.upstream_url
    if stage_path.exists():
        if not (stage_path / ".git").exists():
            raise ExternalRepoError(
                f"Destination already exists and is not a git clone: {stage_path}"
            )
    else:
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["clone", "--no-checkout", source_url, str(stage_path)], cwd=stage_path.parent)
    _run_git(["fetch", "--depth=1", "origin", spec.pinned_sha], cwd=stage_path)
    _run_git(["checkout", "--detach", spec.pinned_sha], cwd=stage_path)
    staged_commit = _run_git(["rev-parse", "HEAD"], cwd=stage_path).stdout.strip()
    if staged_commit != spec.pinned_sha:
        raise ExternalRepoError(
            f"Staged commit mismatch for {spec.name}: expected {spec.pinned_sha}, got "
            f"{staged_commit}"
        )


def _git_tracked_files(stage_path: Path) -> list[Path]:
    """Return tracked files in deterministic order for the staged commit."""
    result = _run_git(["ls-tree", "-r", "--name-only", "HEAD"], cwd=stage_path)
    return [stage_path / line for line in result.stdout.splitlines() if line]


def _tree_checksum(stage_path: Path) -> dict[str, Any]:
    """Compute an aggregate checksum over tracked files in the staged repository."""
    digest = hashlib.sha256()
    sample_files: list[dict[str, Any]] = []
    total_size = 0
    files = _git_tracked_files(stage_path)
    for file_path in files:
        file_digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        size = file_path.stat().st_size
        total_size += size
        rel = file_path.relative_to(stage_path).as_posix()
        file_sha = file_digest.hexdigest()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\0")
        if len(sample_files) < 20:
            sample_files.append({"path": rel, "size_bytes": size, "sha256": file_sha})
    return {
        "file_count": len(files),
        "total_size_bytes": total_size,
        "tree_sha256": digest.hexdigest(),
        "sample_files": sample_files,
    }


def check_repo(
    name: str | None = None,
    *,
    spec: RepoSpec | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate local staging status for one external repository."""
    repo = spec or _get_repo(name or "")
    stage_path = repo.stage_path.expanduser().resolve()
    report: dict[str, Any] = {
        "name": repo.name,
        "title": repo.title,
        "upstream_url": repo.upstream_url,
        "fork_url": repo.fork_url,
        "pinned_sha": repo.pinned_sha,
        "stage_path": str(stage_path),
        "source_access_date": repo.source_access_date,
        "license_note": repo.license_note,
        "license_compatibility_decision": repo.license_compatibility_decision,
        "redistribution_decision": repo.redistribution_decision,
        "intended_use": repo.intended_use,
        "validation_command": repo.validation_command,
        "related_issues": list(repo.related_issues),
        "ok": False,
        "status": "missing",
    }
    if not stage_path.exists():
        report["action"] = (
            "Missing local clone. Run "
            f"`uv run python scripts/tools/manage_external_repos.py stage {repo.name}`."
        )
        return report
    if not (stage_path / ".git").exists():
        report["status"] = "invalid"
        report["action"] = "Staging path exists but is not a git clone."
        return report
    staged_commit = _run_git(["rev-parse", "HEAD"], cwd=stage_path).stdout.strip()
    report["staged_commit"] = staged_commit
    if staged_commit != repo.pinned_sha:
        report["status"] = "wrong-sha"
        report["action"] = (
            f"Clone is at {staged_commit}; restage or checkout pinned SHA {repo.pinned_sha}."
        )
        return report
    if not _git_check_ignored(stage_path, repo_root):
        report["status"] = "not-gitignored"
        report["action"] = "Staging path is not covered by gitignore; add an ignore rule."
        return report
    report["ok"] = True
    report["status"] = "available"
    report["action"] = "Pinned clone satisfies the declared local staging contract."
    return report


def check_all(*, repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    """Validate every registered external repository."""
    return [check_repo(spec=repo, repo_root=repo_root) for repo in REPOS]


def stage_repo(
    name: str | None = None,
    *,
    spec: RepoSpec | None = None,
    manifest_out: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Clone the pinned repository and write a compact provenance manifest."""
    repo = spec or _get_repo(name or "")
    _assert_license_decisions_recorded(repo)
    stage_path = repo.stage_path.expanduser().resolve()
    _ensure_stage_path_ignored(stage_path, repo_root)
    _assert_pinned_sha_reachable(repo)
    try:
        _checkout_pinned_clone(repo, stage_path)
        checksum = _tree_checksum(stage_path)
    except Exception:
        if stage_path.exists() and not any(stage_path.iterdir()):
            shutil.rmtree(stage_path)
        raise
    staged_commit = _run_git(["rev-parse", "HEAD"], cwd=stage_path).stdout.strip()
    manifest = {
        "schema": "robot_sf_external_repo_manifest.v1",
        "name": repo.name,
        "title": repo.title,
        "upstream_url": repo.upstream_url,
        "fork_url": repo.fork_url,
        "pinned_sha": repo.pinned_sha,
        "staged_commit": staged_commit,
        "stage_path": str(stage_path),
        "source_access_date": repo.source_access_date,
        "license_note": repo.license_note,
        "license_compatibility_decision": repo.license_compatibility_decision,
        "redistribution_decision": repo.redistribution_decision,
        "intended_use": repo.intended_use,
        "validation_command": repo.validation_command,
        "file_count": checksum["file_count"],
        "total_size_bytes": checksum["total_size_bytes"],
        "tree_sha256": checksum["tree_sha256"],
        "checksum_policy": (
            "aggregate_tree_sha256 over tracked relative path, size, and sha256 at the pinned "
            "commit; sample file hashes are included for review while the clone remains local."
        ),
        "sample_files": checksum["sample_files"],
        "related_issues": list(repo.related_issues),
        "created_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
    }
    output_path = manifest_out or DEFAULT_MANIFEST_DIR / f"{repo.name}.provenance.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _print_json(payload: Any) -> None:
    """Print a stable JSON payload."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def _repo_summary(repo: RepoSpec, *, include_status: bool) -> dict[str, Any]:
    """Return one list/explain payload."""
    payload: dict[str, Any] = {
        "name": repo.name,
        "title": repo.title,
        "upstream_url": repo.upstream_url,
        "fork_url": repo.fork_url,
        "pinned_sha": repo.pinned_sha,
        "stage_path": str(repo.stage_path),
        "source_access_date": repo.source_access_date,
        "license_note": repo.license_note,
        "license_compatibility_decision": repo.license_compatibility_decision,
        "redistribution_decision": repo.redistribution_decision,
        "intended_use": repo.intended_use,
        "validation_command": repo.validation_command,
        "related_issues": list(repo.related_issues),
    }
    if include_status:
        status = check_repo(spec=repo)
        payload["status"] = status["status"]
        payload["ok"] = status["ok"]
        payload["action"] = status["action"]
    return payload


def _print_repo_summary(payload: dict[str, Any]) -> None:
    """Print a concise human-readable repository summary."""
    print(f"{payload['name']}: {payload['title']}")
    if "status" in payload:
        print(f"  status: {payload['status']}")
        print(f"  action: {payload['action']}")
    print(f"  stage path: {payload['stage_path']}")
    print(f"  upstream: {payload['upstream_url']}")
    print(f"  fork: {payload['fork_url'] or 'none'}")
    print(f"  pinned sha: {payload['pinned_sha']}")
    print(f"  access date: {payload['source_access_date']}")
    print(f"  license: {payload['license_note']}")
    print(f"  compatibility: {payload['license_compatibility_decision']}")
    print(f"  redistribution: {payload['redistribution_decision']}")
    print(f"  intended use: {payload['intended_use']}")
    print(f"  validation: {payload['validation_command']}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List supported repositories and local status.")

    explain_parser = subparsers.add_parser("explain", help="Explain one repository contract.")
    explain_parser.add_argument("name")

    check_parser = subparsers.add_parser("check", help="Check one staged repository.")
    check_parser.add_argument("name", nargs="?", default="all")

    stage_parser = subparsers.add_parser("stage", help="Clone a pinned repo and write manifest.")
    stage_parser.add_argument("name")
    stage_parser.add_argument("--manifest-out", type=Path)

    return parser.parse_args()


def _emit_repo_entries(payload: list[dict[str, Any]], *, as_json: bool) -> None:
    """Print list/check payload entries."""
    if as_json:
        _print_json(payload)
        return
    if not payload:
        print("No external repositories are registered yet.")
        return
    for index, entry in enumerate(payload):
        if index:
            print("")
        _print_repo_summary(entry)


def _handle_list(args: argparse.Namespace) -> int:
    """Handle the list subcommand."""
    payload = [_repo_summary(repo, include_status=True) for repo in REPOS]
    _emit_repo_entries(payload, as_json=args.json)
    return 0


def _handle_explain(args: argparse.Namespace) -> int:
    """Handle the explain subcommand."""
    repo = _get_repo(args.name)
    payload = _repo_summary(repo, include_status=False)
    if args.json:
        _print_json(payload)
        return 0
    _print_repo_summary(payload)
    return 0


def _handle_check(args: argparse.Namespace) -> int:
    """Handle the check subcommand."""
    if args.name == "all":
        payload = check_all()
        _emit_repo_entries(payload, as_json=args.json)
        return 0 if all(entry["ok"] for entry in payload) else 2
    payload = check_repo(args.name)
    if args.json:
        _print_json(payload)
    else:
        _print_repo_summary(payload)
    return 0 if payload["ok"] else 2


def _handle_stage(args: argparse.Namespace) -> int:
    """Handle the stage subcommand."""
    payload = stage_repo(args.name, manifest_out=args.manifest_out)
    if args.json:
        _print_json(payload)
        return 0
    output_path = args.manifest_out or DEFAULT_MANIFEST_DIR / (payload["name"] + ".provenance.json")
    print(f"wrote manifest for {payload['name']}: {output_path}")
    return 0


def _handle_error(exc: ExternalRepoError, *, as_json: bool) -> int:
    """Print one user-facing CLI error."""
    if as_json:
        _print_json({"ok": False, "error": str(exc)})
    else:
        print(f"error: {exc}")
    return 2


def main() -> int:
    """Run the external-repository assistant."""
    args = parse_args()
    handlers = {
        "list": _handle_list,
        "explain": _handle_explain,
        "check": _handle_check,
        "stage": _handle_stage,
    }
    try:
        return handlers[args.command](args)
    except ExternalRepoError as exc:
        return _handle_error(exc, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
