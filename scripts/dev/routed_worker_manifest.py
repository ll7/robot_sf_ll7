#!/usr/bin/env python3
"""Build compact routed-worker artifact manifests.

The manifest records route evidence only. A zero exit code, successful wrapper
run, or complete artifact set is not task acceptance; the parent orchestrator
still needs local diff review and validation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "routed_worker_manifest.v1"
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


@dataclass(frozen=True, slots=True)
class ArtifactPresence:
    """Compact presence record for one expected worker artifact."""

    present: bool
    path: str
    reason: str | None
    size_bytes: int | None


def _resolve_run_dir(run_dir: str | Path, *, target_repo: Path) -> Path:
    """Resolve a run directory relative to the target repository."""
    repo_resolved = target_repo.resolve(strict=False)
    path = Path(run_dir)
    if path.is_absolute():
        unresolved_path = path
    else:
        unresolved_path = target_repo / path
    if unresolved_path.is_symlink():
        raise ValueError("run_dir must not be a symlink")
    resolved = unresolved_path.resolve(strict=False)
    if not resolved.is_relative_to(repo_resolved):
        raise ValueError("run_dir must resolve inside target_repo")
    return resolved


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

    repo_root = Path(target_repo).resolve()
    manifest_attempts: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts):
        run_dir = attempt.get("run_dir")
        if run_dir:
            compact_artifacts = _jsonable_presence(
                scan_artifact_presence(run_dir, target_repo=repo_root)
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
        manifest_attempts.append(
            {
                "attempt_index": index,
                "route": attempt.get("route"),
                "returncode": attempt.get("returncode"),
                "failure_class": attempt.get("failure_class"),
                "run_dir": run_dir,
                "compact_artifacts": compact_artifacts,
            }
        )

    chosen_attempt = manifest_attempts[chosen_index]
    return {
        "schema": SCHEMA_VERSION,
        "task_class": task_class,
        "route_evidence_only": True,
        "warning": ROUTE_EVIDENCE_WARNING,
        "attempted_routes": manifest_attempts,
        "chosen_route": chosen_attempt["route"],
        "chosen_run_dir": chosen_attempt["run_dir"],
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
