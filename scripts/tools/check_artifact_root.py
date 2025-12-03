"""Guard script ensuring artifact producers respect the canonical `output/` root."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from robot_sf.common.artifact_paths import (
    get_artifact_root,
    get_legacy_migration_plan,
    get_repository_root,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass
class Violation:
    """Represents a legacy artifact detected at the repository root."""

    path: str
    remediation: str


@dataclass
class GuardResult:
    """Result container returned by the guard check."""

    violations: list[Violation]
    artifact_root: str

    @property
    def exit_code(self) -> int:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return 0 if not self.violations else 1

    def to_dict(self) -> dict[str, object]:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return {
            "violations": [asdict(v) for v in self.violations],
            "artifact_root": self.artifact_root,
            "exit_code": self.exit_code,
        }


def _normalize_allowlist(entries: Iterable[str | Path], repo_root: Path) -> set[Path]:
    """TODO docstring. Document this function.

    Args:
        entries: TODO docstring.
        repo_root: TODO docstring.

    Returns:
        TODO docstring.
    """
    normalized: set[Path] = set()
    for entry in entries:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        normalized.add(candidate)
    return normalized


def check_artifact_root(
    *,
    source_root: Path | None = None,
    artifact_root: Path | None = None,
    allowlist: Iterable[str | Path] | None = None,
) -> GuardResult:
    """Inspect `source_root` for legacy artifacts and report violations."""

    repo_root = (source_root or get_repository_root()).resolve()
    effective_artifact_root = (artifact_root or get_artifact_root()).resolve()
    allow = _normalize_allowlist(allowlist or (), repo_root)

    violations: list[Violation] = []
    plan = get_legacy_migration_plan()

    for relative_legacy_path, target_relative in plan.items():
        legacy_path = (repo_root / relative_legacy_path).resolve()
        if legacy_path in allow:
            logger.debug("Allowlisted legacy path detected: {}", legacy_path)
            continue
        if legacy_path.exists():
            destination = (effective_artifact_root / target_relative).resolve()
            remediation = (
                f"Relocate {legacy_path} to {destination} "
                "(run robot-sf-migrate-artifacts or update the producer)."
            )
            violations.append(Violation(path=str(legacy_path), remediation=remediation))

    return GuardResult(violations=violations, artifact_root=str(effective_artifact_root))


def _build_parser() -> argparse.ArgumentParser:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Override the repository root to scan for legacy artifacts",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Override the canonical artifact root used for remediation hints",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        help="Relative or absolute path to temporarily tolerate (can be repeated)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        dest="json_report",
        help="Write the guard result JSON to the given path",
    )
    return parser


def _write_json_report(result: GuardResult, path: Path) -> None:
    """TODO docstring. Document this function.

    Args:
        result: TODO docstring.
        path: TODO docstring.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    logger.info("Guard result written to {path}", path=path)


def main(argv: list[str] | None = None) -> int:
    """TODO docstring. Document this function.

    Args:
        argv: TODO docstring.

    Returns:
        TODO docstring.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    result = check_artifact_root(
        source_root=args.repo_root,
        artifact_root=args.artifact_root,
        allowlist=args.allow,
    )

    if result.violations:
        for violation in result.violations:
            logger.error("Legacy artifact detected: {path}", path=violation.path)
            logger.error("  Remedy: {remedy}", remedy=violation.remediation)
        logger.error(
            "Detected {count} legacy artifact(s) outside {root}",
            count=len(result.violations),
            root=result.artifact_root,
        )
    else:
        logger.info("Artifact tree clean; all outputs under {root}", root=result.artifact_root)

    if args.json_report:
        _write_json_report(result, args.json_report.resolve())

    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
