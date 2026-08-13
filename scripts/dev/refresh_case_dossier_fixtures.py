#!/usr/bin/env python3
"""Refresh and validate the compact Chapter 7 case-dossier renderer fixtures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

from robot_sf.benchmark.case_dossier_figure import render_case_dossier

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests/fixtures/benchmark/case_dossier_v1"
TEST_BUILDER_PATH = REPOSITORY_ROOT / "tests/benchmark/test_case_dossier_figure.py"
FIXTURE_SPECS: tuple[tuple[str, str], ...] = (
    ("matched_seed118", "matched_start_planner"),
    ("doorway_seeds113_114", "same_cell_seed_sensitivity"),
)


def _load_test_builder() -> ModuleType:
    """Load the canonical synthetic fixture builder used by the renderer contract tests."""
    spec = importlib.util.spec_from_file_location("case_dossier_test_builder", TEST_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixture builder: {TEST_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for one fixture file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    """Write deterministic sorted JSON with the repository fixture newline convention."""
    path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_fixture_file(source: Path, destination: Path) -> None:
    """Copy one generated fixture file into the tracked package."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, json.loads(source.read_text(encoding="utf-8")))


def _set_relative_source_refs(payload: dict[str, Any], target: Path) -> None:
    """Replace temporary absolute source paths with package-local paths and fresh digests."""
    portfolio = target / "portfolio.json"
    atlas = target / "campaign_atlas.json"
    process_left = target / "process_left.json"
    process_right = target / "process_right.json"
    payload["dossier_id"] = f"{payload['case_id']}-dossier"
    payload["sources"]["portfolio"] = {
        "path": "portfolio.json",
        "sha256": _sha256(portfolio),
    }
    payload["sources"]["campaign_atlas"] = {
        "path": "campaign_atlas.json",
        "sha256": _sha256(atlas),
        "source_class": "release_statistics",
    }
    process_refs = payload["sources"]["process_traces"]
    for source, path in zip(process_refs, (process_left, process_right), strict=True):
        source["path"] = path.name
        source["sha256"] = _sha256(path)


def _refresh_one_fixture(
    builder: ModuleType,
    folder: str,
    grammar: str,
    *,
    fixture_root: Path | None = None,
) -> Path:
    """Generate one fixture package and its two deliberate negative inputs."""
    target_root = FIXTURE_ROOT if fixture_root is None else fixture_root
    target = target_root / folder
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="case-dossier-refresh-") as raw_dir:
        generated_root = Path(raw_dir)
        generated_input = builder._write_input(generated_root, grammar=grammar)
        generated_names = (
            ("portfolio.json", "portfolio.json"),
            ("atlas.json", "campaign_atlas.json"),
            ("goal-process.json", "process_left.json"),
            ("ppo-process.json", "process_left.json"),
            ("ppo-right-process.json", "process_right.json"),
        )
        copied: set[str] = set()
        for source_name, destination_name in generated_names:
            source = generated_root / source_name
            if source.exists() and destination_name not in copied:
                _copy_fixture_file(source, target / destination_name)
                copied.add(destination_name)
        payload = json.loads(generated_input.read_text(encoding="utf-8"))
        _set_relative_source_refs(payload, target)
        _write_json(target / "input.json", payload)

        if folder == "matched_seed118":
            bad_hash = copy.deepcopy(payload)
            bad_hash["sources"]["process_traces"][0]["sha256"] = "0" * 64
            _write_json(target / "bad_source_hash.input.json", bad_hash)
        else:
            bad_difference = copy.deepcopy(payload)
            bad_difference["comparison_options"]["difference_curve"] = True
            _write_json(target / "bad_difference_curve.input.json", bad_difference)
    return target / "input.json"


def _fixture_tree_digest(root: Path) -> str:
    """Return a stable digest over a fixture package's relative paths and bytes."""
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _fixture_display_path(path: Path) -> str:
    """Prefer repository-relative paths while keeping temporary test roots printable."""
    try:
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _first_differing_relative_path(generated: Path, tracked: Path) -> str | None:
    """Return the first sorted relative file path whose presence or bytes differ."""
    generated_paths = {
        path.relative_to(generated).as_posix() for path in generated.rglob("*") if path.is_file()
    }
    tracked_paths = {
        path.relative_to(tracked).as_posix() for path in tracked.rglob("*") if path.is_file()
    }
    for relative in sorted(generated_paths | tracked_paths):
        generated_path = generated / relative
        tracked_path = tracked / relative
        if not generated_path.is_file() or not tracked_path.is_file():
            return relative
        if generated_path.read_bytes() != tracked_path.read_bytes():
            return relative
    if generated.is_dir() != tracked.is_dir():
        return "<package-root>"
    return None


def _compare_fixture_package(generated: Path, tracked: Path, folder: str) -> dict[str, Any]:
    """Compare one generated package with its committed counterpart."""
    generated_digest = _fixture_tree_digest(generated)
    tracked_digest = _fixture_tree_digest(tracked)
    relative_difference = _first_differing_relative_path(generated, tracked)
    report: dict[str, Any] = {
        "fixture": folder,
        "source_digest": generated_digest,
        "tracked_digest": tracked_digest,
    }
    if relative_difference is None:
        report["status"] = "ok"
        return report
    report.update(
        {
            "status": "drift",
            "first_differing_path": _fixture_display_path(tracked / relative_difference),
        }
    )
    return report


def _check_generated_fixtures(builder: ModuleType) -> list[dict[str, Any]]:
    """Regenerate both packages in temporary storage and compare them byte-for-byte."""
    with tempfile.TemporaryDirectory(prefix="case-dossier-fixture-check-") as raw_dir:
        generated_root = Path(raw_dir) / "fixtures"
        reports = []
        for folder, grammar in FIXTURE_SPECS:
            _refresh_one_fixture(builder, folder, grammar, fixture_root=generated_root)
            reports.append(
                _compare_fixture_package(
                    generated_root / folder,
                    FIXTURE_ROOT / folder,
                    folder,
                )
            )
        return reports


def _check_fixture(path: Path) -> dict[str, Any]:
    """Render one committed fixture in a disposable directory and return a compact report."""
    with tempfile.TemporaryDirectory(prefix="case-dossier-check-") as raw_dir:
        bundle = render_case_dossier(path, Path(raw_dir) / "rendered")
    return {
        "path": _fixture_display_path(path),
        "status": "ok",
        "case_id": bundle.manifest["selection"]["case_id"],
        "comparison_grammar": bundle.manifest["comparison_grammar"],
        "mode": bundle.manifest["mode"],
        "scientific_admission": bundle.manifest["scientific_admission"],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--write",
        action="store_true",
        help="Regenerate tracked fixture packages from the canonical test builder first.",
    )
    modes.add_argument(
        "--check",
        action="store_true",
        help=(
            "Regenerate fixture packages in temporary storage and fail on byte-level drift "
            "without modifying tracked files."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Refresh, compare, or validate the production-shaped fixture packages."""
    args = _parse_args(argv)
    if args.check:
        builder = _load_test_builder()
        reports = _check_generated_fixtures(builder)
        status = "ok" if all(report["status"] == "ok" for report in reports) else "drift"
        print(
            json.dumps(
                {"mode": "check", "status": status, "fixtures": reports}, indent=2, sort_keys=True
            )
        )
        return 0 if status == "ok" else 1

    builder = _load_test_builder() if args.write else None
    if builder is not None:
        for folder, grammar in FIXTURE_SPECS:
            _refresh_one_fixture(builder, folder, grammar)
    reports = [_check_fixture(FIXTURE_ROOT / folder / "input.json") for folder, _ in FIXTURE_SPECS]
    print(json.dumps({"status": "ok", "fixtures": reports}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
