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


def _refresh_one_fixture(builder: ModuleType, folder: str, grammar: str) -> Path:
    """Generate one fixture package and its two deliberate negative inputs."""
    target = FIXTURE_ROOT / folder
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


def _check_fixture(path: Path) -> dict[str, Any]:
    """Render one committed fixture in a disposable directory and return a compact report."""
    with tempfile.TemporaryDirectory(prefix="case-dossier-check-") as raw_dir:
        bundle = render_case_dossier(path, Path(raw_dir) / "rendered")
    return {
        "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "status": "ok",
        "case_id": bundle.manifest["selection"]["case_id"],
        "comparison_grammar": bundle.manifest["comparison_grammar"],
        "mode": bundle.manifest["mode"],
        "scientific_admission": bundle.manifest["scientific_admission"],
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate tracked fixture packages from the canonical test builder first.",
    )
    return parser.parse_args()


def main() -> int:
    """Refresh when requested, then validate every committed production-shaped fixture."""
    args = _parse_args()
    builder = _load_test_builder() if args.write else None
    if builder is not None:
        for folder, grammar in FIXTURE_SPECS:
            _refresh_one_fixture(builder, folder, grammar)
    reports = [_check_fixture(FIXTURE_ROOT / folder / "input.json") for folder, _ in FIXTURE_SPECS]
    print(json.dumps({"status": "ok", "fixtures": reports}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
