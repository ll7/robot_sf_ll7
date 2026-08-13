#!/usr/bin/env python3
"""Build compact routed-worker artifact manifests.

The manifest records route evidence only. A zero exit code, successful wrapper
run, or complete artifact set is not task acceptance; the parent orchestrator
still needs local diff review and validation.
"""

from __future__ import annotations

import argparse
import enum
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

SCHEMA_VERSION = "routed_worker_manifest.v2"
ROUTE_EVIDENCE_WARNING = (
    "Wrapper success, zero exit, and manifest presence are route evidence only; "
    "they are not task acceptance. The orchestrator must still inspect the diff "
    "and run the required local validation."
)
REQUIRED_ARTIFACTS = {
    "result_json": "result.json",
    "result_md": "RESULT.md",
    "diffstat": "diffstat.txt",
    "status": "status.txt",
    "validation": "validation.txt",
}


class TerminalFailure(enum.StrEnum):
    """Compact terminal failure classification for route outcomes."""

    NONE = "none"
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    NON_ZERO_EXIT = "non_zero_exit"
    MISSING_ARTIFACT = "missing_artifact"
    ROUTE_NOT_STARTED = "route_not_started"
    SCOPE_VIOLATION = "scope_violation"
    UNAVAILABLE = "unavailable"


def classify_terminal_state(
    *,
    returncode: int | None = None,
    failure_class: str | None = None,
    artifact_presence: dict[str, ArtifactPresence] | None = None,
    has_run_dir: bool = True,
) -> TerminalFailure:
    """Derive the compact terminal state for a route attempt.

    Absent details remain explicit rather than inferred: an attempt with no
    run_dir is ``ROUTE_NOT_STARTED``; missing artifacts yield
    ``MISSING_ARTIFACT``; non-zero returncode or known failure classes map to
    their respective terminal states.
    """
    if not has_run_dir:
        return TerminalFailure.ROUTE_NOT_STARTED
    if failure_class == "timeout" or returncode == 124:
        return TerminalFailure.TIMEOUT
    if failure_class in {"exception", "error"} or (returncode is not None and returncode < 0):
        return TerminalFailure.EXCEPTION
    if returncode is not None and returncode != 0:
        return TerminalFailure.NON_ZERO_EXIT
    if artifact_presence is not None:
        required_missing = [
            key
            for key, entry in artifact_presence.items()
            if not entry.present and entry.reason == "missing"
        ]
        if required_missing:
            return TerminalFailure.MISSING_ARTIFACT
    if failure_class not in {None, "none", "success"}:
        return TerminalFailure.UNAVAILABLE
    if returncode is None and failure_class is None:
        return TerminalFailure.UNAVAILABLE
    return TerminalFailure.NONE


@dataclass(frozen=True, slots=True)
class ArtifactPresence:
    """Compact presence record for one expected worker artifact."""

    present: bool
    path: str
    reason: str | None
    size_bytes: int | None


@dataclass(frozen=True, slots=True)
class TargetWorktreeCheck:
    """Result of validating the target checkout through Git."""

    requested_worktree: str
    resolved_worktree: str
    git_top_level: str | None
    common_git_dir: str | None
    ok: bool
    failure: str | None


@dataclass(frozen=True, slots=True)
class ScopeCheck:
    """Result of validating a run directory and its reported artifacts."""

    ok: bool
    resolved_run_dir: str
    failure: str | None
    spill_detected: bool
    spill_detail: str | None
    authorized_root: str | None = None
    spill_paths: tuple[str, ...] = ()


def _git_rev_parse(target_repo: Path, *arguments: str) -> tuple[str | None, str | None]:
    """Run one bounded Git query against a target checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(target_repo), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return None, str(exc)
    if result.returncode != 0:
        return None, (result.stderr or result.stdout).strip() or f"exit {result.returncode}"
    value = result.stdout.strip()
    return (value or None), None


def validate_target_worktree(target_repo: str | Path = ".") -> TargetWorktreeCheck:
    """Validate the absolute target worktree and discover its shared Git directory."""
    requested = Path(target_repo).resolve(strict=False)
    if not requested.is_dir():
        return TargetWorktreeCheck(
            requested_worktree=str(requested),
            resolved_worktree=str(requested),
            git_top_level=None,
            common_git_dir=None,
            ok=False,
            failure="target_repo is not an existing directory",
        )

    git_top_level, error = _git_rev_parse(requested, "rev-parse", "--show-toplevel")
    if error or git_top_level is None:
        return TargetWorktreeCheck(
            requested_worktree=str(requested),
            resolved_worktree=str(requested),
            git_top_level=None,
            common_git_dir=None,
            ok=False,
            failure=f"git rev-parse --show-toplevel failed: {error or 'empty output'}",
        )

    resolved_top_level = Path(git_top_level).resolve(strict=False)
    if resolved_top_level != requested:
        return TargetWorktreeCheck(
            requested_worktree=str(requested),
            resolved_worktree=str(requested),
            git_top_level=str(resolved_top_level),
            common_git_dir=None,
            ok=False,
            failure=(
                f"target worktree mismatch: git top-level {resolved_top_level} "
                f"does not match target_repo {requested}"
            ),
        )

    common_git_dir, error = _git_rev_parse(
        requested, "rev-parse", "--path-format=absolute", "--git-common-dir"
    )
    if error or common_git_dir is None:
        return TargetWorktreeCheck(
            requested_worktree=str(requested),
            resolved_worktree=str(requested),
            git_top_level=str(resolved_top_level),
            common_git_dir=None,
            ok=False,
            failure=f"git common directory lookup failed: {error or 'empty output'}",
        )

    return TargetWorktreeCheck(
        requested_worktree=str(requested),
        resolved_worktree=str(requested),
        git_top_level=str(resolved_top_level),
        common_git_dir=str(Path(common_git_dir).resolve(strict=False)),
        ok=True,
        failure=None,
    )


def _artifact_root(target: TargetWorktreeCheck) -> Path:
    """Return the shared Git-dir root authorized for agent-run artifacts."""
    if not target.common_git_dir:
        raise ValueError("target worktree has no shared Git directory")
    return Path(target.common_git_dir) / "codex-agent-runs"


def _authorized_run_roots(target: TargetWorktreeCheck) -> tuple[tuple[Path, str], ...]:
    """Return worktree and shared artifact roots accepted for a run directory."""
    roots: list[tuple[Path, str]] = []
    if target.common_git_dir:
        roots.append((_artifact_root(target), "shared_git_artifacts"))
    roots.append((Path(target.resolved_worktree), "target_worktree"))
    return tuple(roots)


def _resolve_run_dir(
    run_dir: str | Path,
    *,
    target_repo: Path,
    target_worktree: TargetWorktreeCheck | None = None,
) -> Path:
    """Resolve a run directory inside the worktree or authorized artifact root."""
    target = target_worktree or validate_target_worktree(target_repo)
    if not target.ok:
        raise ValueError(target.failure or "target worktree validation failed")

    path = Path(run_dir)
    if path.is_absolute():
        unresolved_path = path
    else:
        unresolved_path = target_repo / path
    if unresolved_path.is_symlink():
        raise ValueError("run_dir must not be a symlink")
    resolved = unresolved_path.resolve(strict=False)
    if not any(resolved.is_relative_to(root) for root, _root_name in _authorized_run_roots(target)):
        raise ValueError(
            "run_dir must resolve inside target_repo or the shared Git codex-agent-runs root"
        )
    return resolved


def validate_run_dir_scope(  # noqa: C901, PLR0912
    run_dir: str | Path,
    *,
    target_repo: str | Path = ".",
    artifact_paths: Iterable[object] | None = None,
) -> ScopeCheck:
    """Validate a run directory and reported artifacts against route scope.

    A run directory may be inside the target worktree or inside the shared
    Git-dir ``codex-agent-runs`` root. Reported artifact paths are resolved
    relative to the run directory and must remain inside that assigned bundle.
    Direct bundle files found in the shared artifact root are also classified
    as spill, covering the common failure where a worker writes ``RESULT.md``
    beside the per-run directory.
    """
    repo_root = Path(target_repo).resolve(strict=False)
    target = validate_target_worktree(repo_root)
    if not target.ok:
        return ScopeCheck(
            ok=False,
            resolved_run_dir="",
            failure=target.failure,
            spill_detected=True,
            spill_detail=f"target worktree scope violation: {target.failure}",
        )

    path = Path(run_dir)
    unresolved_path = path if path.is_absolute() else repo_root / path
    if unresolved_path.is_symlink():
        return ScopeCheck(
            ok=False,
            resolved_run_dir="",
            failure="run_dir must not be a symlink",
            spill_detected=True,
            spill_detail="scope violation: run_dir must not be a symlink",
        )

    resolved = unresolved_path.resolve(strict=False)
    authorized_root: str | None = None
    for root, root_name in _authorized_run_roots(target):
        if resolved.is_relative_to(root):
            authorized_root = root_name
            break
    if authorized_root is None:
        common_dir = Path(target.common_git_dir) if target.common_git_dir else None
        if common_dir and resolved.is_relative_to(common_dir):
            detail = (
                f"run_dir resolves inside shared Git directory {common_dir} but outside "
                "codex-agent-runs"
            )
        else:
            detail = (
                f"run_dir {resolved} is outside target worktree {repo_root} and the "
                "shared Git codex-agent-runs root"
            )
        return ScopeCheck(
            ok=False,
            resolved_run_dir=str(resolved),
            failure="run_dir is outside authorized route scope",
            spill_detected=True,
            spill_detail=detail,
            authorized_root=None,
        )

    spill_paths: list[str] = []
    reported = artifact_paths
    if reported is not None:
        if isinstance(reported, (str, bytes, Path)):
            reported_values: Iterable[object] = (reported,)
        else:
            reported_values = reported
        for raw_path in reported_values:
            if not isinstance(raw_path, (str, Path)) or not str(raw_path):
                spill_paths.append(f"<invalid artifact path: {raw_path!r}>")
                continue
            artifact_path = Path(raw_path)
            candidate = artifact_path if artifact_path.is_absolute() else resolved / artifact_path
            candidate_resolved = candidate.resolve(strict=False)
            if not candidate_resolved.is_relative_to(resolved):
                spill_paths.append(str(candidate_resolved))

    if target.common_git_dir and resolved.is_relative_to(_artifact_root(target)):
        for root in (_artifact_root(target), Path(target.common_git_dir)):
            for filename in REQUIRED_ARTIFACTS.values():
                candidate = root / filename
                if candidate.is_file():
                    spill_paths.append(str(candidate.resolve(strict=False)))

    unique_spill_paths = tuple(dict.fromkeys(spill_paths))
    if unique_spill_paths:
        detail = "artifact spill detected outside assigned run directory"
        return ScopeCheck(
            ok=False,
            resolved_run_dir=str(resolved),
            failure=detail,
            spill_detected=True,
            spill_detail=f"{detail}: {', '.join(unique_spill_paths)}",
            authorized_root=authorized_root,
            spill_paths=unique_spill_paths,
        )

    return ScopeCheck(
        ok=True,
        resolved_run_dir=str(resolved),
        failure=None,
        spill_detected=False,
        spill_detail=None,
        authorized_root=authorized_root,
    )


def scan_artifact_presence(
    run_dir: str | Path,
    *,
    target_repo: str | Path = ".",
    artifact_filenames: dict[str, str] | None = None,
) -> dict[str, ArtifactPresence]:
    """Scan expected artifact files in a worker run directory.

    Relative run directories are resolved against ``target_repo`` so wrappers
    write manifests into the repository being operated on, not into the
    orchestrator tooling checkout.
    """
    repo_root = Path(target_repo).resolve()
    run_root = _resolve_run_dir(run_dir, target_repo=repo_root)
    filenames = artifact_filenames or REQUIRED_ARTIFACTS
    presence: dict[str, ArtifactPresence] = {}
    for key, filename in filenames.items():
        artifact_path = run_root / filename
        relative_path = str(Path(run_dir) / filename)
        if artifact_path.is_file():
            presence[key] = ArtifactPresence(
                present=True,
                path=relative_path,
                reason=None,
                size_bytes=artifact_path.stat().st_size,
            )
        else:
            presence[key] = ArtifactPresence(
                present=False,
                path=relative_path,
                reason="missing",
                size_bytes=None,
            )
    return presence


def _jsonable_presence(presence: dict[str, ArtifactPresence]) -> dict[str, dict[str, Any]]:
    """Return JSON-ready artifact presence entries."""
    return {key: asdict(entry) for key, entry in presence.items()}


def build_routing_manifest(
    attempts: list[dict[str, Any]],
    *,
    chosen_index: int,
    target_repo: str | Path = ".",
    task_class: str | None = None,
) -> dict[str, Any]:
    """Build a routed-worker manifest for every attempt and the chosen route."""
    if not attempts:
        raise ValueError("at least one route attempt is required")
    if chosen_index < 0 or chosen_index >= len(attempts):
        raise IndexError("chosen_index is outside attempts")

    repo_root = Path(target_repo).resolve(strict=False)
    target_worktree = validate_target_worktree(repo_root)
    manifest_attempts: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts):
        run_dir = attempt.get("run_dir")
        scope_check: ScopeCheck | None = None
        if run_dir:
            scope_check = validate_run_dir_scope(
                run_dir,
                target_repo=repo_root,
                artifact_paths=attempt.get("artifact_paths"),
            )
            if not scope_check.ok:
                compact_artifacts = _jsonable_presence(
                    {
                        key: ArtifactPresence(
                            present=False,
                            path=filename,
                            reason="scope_violation",
                            size_bytes=None,
                        )
                        for key, filename in REQUIRED_ARTIFACTS.items()
                    }
                )
                terminal_state = TerminalFailure.SCOPE_VIOLATION
            else:
                compact_artifacts = _jsonable_presence(
                    scan_artifact_presence(run_dir, target_repo=repo_root)
                )
                artifact_presence = {
                    key: ArtifactPresence(**val) for key, val in compact_artifacts.items()
                }
                terminal_state = classify_terminal_state(
                    returncode=attempt.get("returncode"),
                    failure_class=attempt.get("failure_class"),
                    artifact_presence=artifact_presence,
                    has_run_dir=True,
                )
        else:
            compact_artifacts = _jsonable_presence(
                {
                    key: ArtifactPresence(
                        present=False,
                        path=filename,
                        reason=attempt.get("missing_reason") or "not-run",
                        size_bytes=None,
                    )
                    for key, filename in REQUIRED_ARTIFACTS.items()
                }
            )
            terminal_state = classify_terminal_state(
                returncode=attempt.get("returncode"),
                failure_class=attempt.get("failure_class"),
                has_run_dir=False,
            )
        scope_dict = asdict(scope_check) if scope_check is not None else None
        manifest_attempts.append(
            {
                "attempt_index": index,
                "route": attempt.get("route"),
                "returncode": attempt.get("returncode"),
                "failure_class": attempt.get("failure_class"),
                "terminal_state": terminal_state.value,
                "run_dir": run_dir,
                "artifact_paths": attempt.get("artifact_paths"),
                "compact_artifacts": compact_artifacts,
                "scope_check": scope_dict,
            }
        )

    chosen_attempt = manifest_attempts[chosen_index]
    return {
        "schema": SCHEMA_VERSION,
        "task_class": task_class,
        "target_worktree": asdict(target_worktree),
        "route_evidence_only": True,
        "warning": ROUTE_EVIDENCE_WARNING,
        "attempted_routes": manifest_attempts,
        "chosen_route": chosen_attempt["route"],
        "chosen_run_dir": chosen_attempt["run_dir"],
        "chosen_terminal_state": chosen_attempt["terminal_state"],
        "chosen_scope_check": chosen_attempt["scope_check"],
        "compact_artifacts": chosen_attempt["compact_artifacts"],
    }


def write_routing_manifest(
    attempts: list[dict[str, Any]],
    *,
    chosen_index: int,
    target_repo: str | Path = ".",
    task_class: str | None = None,
    filename: str = "routing_manifest.json",
) -> Path:
    """Write the routing manifest into the chosen attempt run directory."""
    manifest = build_routing_manifest(
        attempts,
        chosen_index=chosen_index,
        target_repo=target_repo,
        task_class=task_class,
    )
    chosen_run_dir = manifest["chosen_run_dir"]
    if not chosen_run_dir:
        raise ValueError("chosen route has no run_dir; cannot write manifest")
    run_root = _resolve_run_dir(chosen_run_dir, target_repo=Path(target_repo).resolve())
    run_root.mkdir(parents=True, exist_ok=True)
    output_path = run_root / filename
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempts-json", required=True, help="JSON file containing route attempts."
    )
    parser.add_argument("--chosen-index", type=int, required=True)
    parser.add_argument("--target-repo", default=".")
    parser.add_argument("--task-class")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = _parse_args()
    attempts = json.loads(Path(args.attempts_json).read_text(encoding="utf-8"))
    output_path = write_routing_manifest(
        attempts,
        chosen_index=args.chosen_index,
        target_repo=args.target_repo,
        task_class=args.task_class,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
