"""Migration helper to consolidate legacy artifacts under the canonical `output/` root."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger

from robot_sf.common.artifact_paths import (
    ensure_canonical_tree,
    find_legacy_artifact_paths,
    get_artifact_root,
    get_legacy_migration_plan,
    get_repository_root,
)


@dataclass
class RelocatedItem:
    legacy_path: str
    destination: str
    item_type: str


@dataclass
class SkippedItem:
    legacy_path: str
    reason: str


@dataclass
class MigrationReport:
    relocated: list[RelocatedItem]
    skipped: list[SkippedItem]
    warnings: list[str]
    artifact_root: str

    def to_dict(self) -> dict[str, object]:
        return {
            "relocated": [asdict(item) for item in self.relocated],
            "skipped": [asdict(item) for item in self.skipped],
            "warnings": list(self.warnings),
            "artifact_root": self.artifact_root,
        }


def _serialize_report(report: MigrationReport, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    logger.info("Migration report written to {path}", path=report_path)


def _iter_legacy_sources(source_root: Path) -> list[Path]:
    return sorted(find_legacy_artifact_paths(source_root))


def migrate_artifacts(
    *,
    dry_run: bool = False,
    source_root: Path | None = None,
    artifact_root: Path | None = None,
    report_path: Path | None = None,
) -> MigrationReport:
    """Relocate known legacy artifacts into the canonical tree."""

    resolved_source = (source_root or get_repository_root()).resolve()
    resolved_artifact_root = (artifact_root or get_artifact_root()).resolve()

    if not dry_run:
        resolved_artifact_root.mkdir(parents=True, exist_ok=True)

    plan = get_legacy_migration_plan()
    relocated: list[RelocatedItem] = []
    skipped: list[SkippedItem] = []
    warnings: list[str] = []

    for legacy_path in _iter_legacy_sources(resolved_source):
        relative = legacy_path.relative_to(resolved_source)
        target_relative = plan.get(relative)
        if target_relative is None:
            reason = f"No migration target defined for {relative}"
            logger.warning(reason)
            warnings.append(reason)
            skipped.append(SkippedItem(legacy_path=str(legacy_path), reason="unknown-path"))
            continue

        destination = (resolved_artifact_root / target_relative).resolve()
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
        item_type = "directory" if legacy_path.is_dir() else "file"

        if dry_run:
            logger.info(
                "[dry-run] Would relocate {src} → {dst}",
                src=legacy_path,
                dst=destination,
            )
            relocated.append(
                RelocatedItem(
                    legacy_path=str(legacy_path),
                    destination=str(destination),
                    item_type=item_type,
                ),
            )
            continue

        if destination.exists():
            reason = "destination-exists"
            logger.warning(
                "Skipping {src} because destination already exists at {dst}",
                src=legacy_path,
                dst=destination,
            )
            skipped.append(SkippedItem(legacy_path=str(legacy_path), reason=reason))
            continue

        logger.info("Relocating {src} → {dst}", src=legacy_path, dst=destination)
        shutil.move(str(legacy_path), str(destination))
        relocated.append(
            RelocatedItem(
                legacy_path=str(legacy_path),
                destination=str(destination),
                item_type=item_type,
            ),
        )

    if not dry_run:
        ensure_canonical_tree(resolved_artifact_root)

    report = MigrationReport(
        relocated=relocated,
        skipped=skipped,
        warnings=warnings,
        artifact_root=str(resolved_artifact_root),
    )

    if report_path and (report.relocated or report.skipped or report.warnings):
        _serialize_report(report, report_path.resolve())

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without modifying files",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Override the repository root where legacy artifacts currently live",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Override the destination artifact root (defaults to canonical root)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Write the migration report JSON to the given path (default: <artifact_root>/migration-report.json)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable writing the JSON migration report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    resolved_source = args.source_root or get_repository_root()
    resolved_artifact_root = args.artifact_root or get_artifact_root()
    default_report = (resolved_artifact_root / "migration-report.json").resolve()
    report_path = None if args.no_report else (args.report_path or default_report)

    report = migrate_artifacts(
        dry_run=args.dry_run,
        source_root=resolved_source,
        artifact_root=resolved_artifact_root,
        report_path=report_path,
    )

    logger.info(
        "Relocated: {relocated} | Skipped: {skipped} | Warnings: {warnings}",
        relocated=len(report.relocated),
        skipped=len(report.skipped),
        warnings=len(report.warnings),
    )

    if not report.relocated and not report.skipped:
        logger.info("No legacy artifacts detected under {root}", root=resolved_source)

    return 0


if __name__ == "__main__":
    sys.exit(main())
