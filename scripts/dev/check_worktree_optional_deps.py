#!/usr/bin/env python3
"""Check declared dependency imports without importing Robot SF project code.

This is a setup diagnostic for linked worktrees. It deliberately probes import
specifications only, so an incomplete environment is reported separately from a
failure while importing changed project code.

Exit codes:
    0: all requested optional imports are available.
    1: the dependency probe itself failed for at least one import.
    2: one or more optional imports are missing.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

EXTRA_MODULES = {
    "viz": ("pygame", "matplotlib", "PIL", "moviepy", "imageio_ffmpeg", "seaborn"),
    "maps": ("osmnx", "geopandas", "pyproj"),
    "benchmark": ("pandas", "scipy"),
    "gpu": ("torch",),
    "training": (
        "stable_baselines3",
        "torch",
        "sklearn",
        "optuna",
        "tensorboard",
        "wandb",
        "optuna_dashboard",
    ),
    "recurrent": ("sb3_contrib",),
    "rllib": ("ray",),
    "progress": ("tqdm",),
    "analytics": ("duckdb", "pyarrow"),
    "browser": ("playwright",),
    "sacadrl": ("tensorflow",),
    "orca": ("rvo2",),
    "socnav": ("cv2", "pyassimp", "OpenGL", "skfmm", "skimage"),
    "criticality": ("cma",),
}
CORE_MODULES = ("yaml",)
ALL_EXTRAS_MODULES = tuple(
    dict.fromkeys(module for modules in EXTRA_MODULES.values() for module in modules)
)
PROFILES = {"core": CORE_MODULES, "all-extras": ALL_EXTRAS_MODULES, **EXTRA_MODULES}
SCHEMA = "robot_sf.worktree_optional_deps.v1"


def _default_pyproject_path() -> Path:
    """Resolve the default pyproject.toml from the repo tree or current directory."""
    candidate = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if candidate.is_file():
        return candidate
    return Path.cwd() / "pyproject.toml"


def _parse_declared_entry_points(data: dict[str, Any]) -> list[dict[str, str]]:
    """Extract declared entry points from parsed pyproject table."""
    declared: list[dict[str, str]] = []
    project_table = data.get("project", {})
    if not isinstance(project_table, dict):
        return declared
    ep_table = project_table.get("entry-points", {})
    if not isinstance(ep_table, dict):
        return declared
    for group, entries in sorted(ep_table.items()):
        if isinstance(entries, dict):
            for name, value in sorted(entries.items()):
                declared.append(
                    {
                        "group": str(group),
                        "name": str(name),
                        "expected_value": str(value),
                    }
                )
    return declared


def _find_installed_distribution(name: str) -> importlib.metadata.Distribution | None:
    """Find installed distribution under canonical or hyphen/underscore variants."""
    candidates = (name, name.replace("-", "_"), name.replace("_", "-"))
    for candidate in dict.fromkeys(candidates):
        try:
            return importlib.metadata.distribution(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _partition_entry_points(
    declared: list[dict[str, str]],
    dist: importlib.metadata.Distribution,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Partition declared entry points into missing, mismatched, and matching."""
    installed_map: dict[tuple[str, str], str] = {
        (ep.group, ep.name): ep.value for ep in dist.entry_points
    }
    missing: list[dict[str, str]] = []
    mismatched: list[dict[str, str]] = []
    matching: list[dict[str, str]] = []

    for item in declared:
        key = (item["group"], item["name"])
        if key not in installed_map:
            missing.append(item)
        elif installed_map[key] != item["expected_value"]:
            mismatched.append({**item, "actual_value": installed_map[key]})
        else:
            matching.append({**item, "actual_value": installed_map[key]})
    return missing, mismatched, matching


def check_declared_entry_points(
    pyproject_path: Path | str,
    *,
    distribution_name: str = "robot-sf",
) -> dict[str, Any]:
    """Compare declared project entry points against the installed distribution.

    Args:
        pyproject_path: Path to the worktree's pyproject.toml.
        distribution_name: Installed package name to inspect (default: robot-sf).

    Returns:
        Structured report with status ("ready", "missing_entry_points",
        "mismatched_entry_points", or "check_failed"), declared entry points,
        and missing/mismatched/matching partitions.
    """
    path = Path(pyproject_path).resolve()
    if not path.is_file():
        return {
            "status": "check_failed",
            "exit_code": 1,
            "distribution": distribution_name,
            "pyproject_path": str(path),
            "declared_count": 0,
            "missing": [],
            "mismatched": [],
            "matching": [],
            "error": f"pyproject.toml not found at {path}",
        }

    try:
        with open(path, "rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return {
            "status": "check_failed",
            "exit_code": 1,
            "distribution": distribution_name,
            "pyproject_path": str(path),
            "declared_count": 0,
            "missing": [],
            "mismatched": [],
            "matching": [],
            "error": f"Failed to parse {path}: {exc}",
        }

    declared = _parse_declared_entry_points(data)
    dist = _find_installed_distribution(distribution_name)
    if dist is None:
        return {
            "status": "missing_entry_points" if declared else "ready",
            "exit_code": 2 if declared else 0,
            "distribution": distribution_name,
            "installed": False,
            "pyproject_path": str(path),
            "declared_count": len(declared),
            "missing": declared,
            "mismatched": [],
            "matching": [],
            "error": f"Package {distribution_name!r} is not installed in the active environment.",
        }

    missing, mismatched, matching = _partition_entry_points(declared, dist)
    if missing:
        status = "missing_entry_points"
        exit_code = 2
    elif mismatched:
        status = "mismatched_entry_points"
        exit_code = 2
    else:
        status = "ready"
        exit_code = 0

    return {
        "status": status,
        "exit_code": exit_code,
        "distribution": distribution_name,
        "installed": True,
        "pyproject_path": str(path),
        "declared_count": len(declared),
        "missing": missing,
        "mismatched": mismatched,
        "matching": matching,
    }


def check_modules(
    modules: Iterable[str],
    *,
    profile: str,
    entry_points_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a dependency-only availability report for *modules*."""
    checks: list[dict[str, Any]] = []
    for module in dict.fromkeys(modules):
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            checks.append(
                {
                    "module": module,
                    "available": False,
                    "status": "check_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        checks.append(
            {
                "module": module,
                "available": available,
                "status": "available" if available else "missing_optional",
            }
        )

    missing = [check["module"] for check in checks if check["status"] == "missing_optional"]
    failures = [check["module"] for check in checks if check["status"] == "check_failed"]
    if failures:
        status = "check_failed"
        exit_code = 1
    elif missing:
        status = "missing_optional"
        exit_code = 2
    elif entry_points_report is not None and entry_points_report["status"] != "ready":
        status = entry_points_report["status"]
        exit_code = int(entry_points_report["exit_code"])
    else:
        status = "ready"
        exit_code = 0

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "profile": profile,
        "status": status,
        "exit_code": exit_code,
        "checked_count": len(checks),
        "missing_optional": missing,
        "check_failures": failures,
        "checks": checks,
        "project_imports_performed": False,
    }
    if entry_points_report is not None:
        report["entry_points"] = entry_points_report
    return report


def _render_entry_points_report(lines: list[str], ep_info: dict[str, Any]) -> None:
    """Render the entry-points section of the human report."""
    dist_name = ep_info.get("distribution", "robot-sf")
    lines.append(f"Entry-point metadata ({dist_name}): {ep_info['status']}")
    if ep_info.get("error"):
        lines.append(f"  Error: {ep_info['error']}")
    if ep_info.get("missing"):
        lines.append("Missing declared entry points:")
        for ep in ep_info["missing"]:
            lines.append(f"  - [{ep['group']}] {ep['name']} = {ep['expected_value']}")
    if ep_info.get("mismatched"):
        lines.append("Mismatched declared entry points:")
        for ep in ep_info["mismatched"]:
            lines.append(
                f"  - [{ep['group']}] {ep['name']}: expected {ep['expected_value']}, "
                f"found {ep.get('actual_value')}"
            )
    if ep_info.get("missing") or ep_info.get("mismatched"):
        lines.append(
            "Remedy: install this checkout into the active environment: "
            "'uv pip install --no-deps -e .' or 'uv sync --all-extras --reinstall-package robot-sf'."
        )


def _render_human(report: dict[str, Any]) -> str:
    """Render a concise setup-only report."""
    status = report["status"]
    lines = [
        f"Worktree optional dependency preflight: {status} ({report['profile']})",
        f"Checked {report['checked_count']} import probes without importing project code.",
    ]
    if report["missing_optional"]:
        missing = ", ".join(report["missing_optional"])
        lines.append(f"Missing optional imports: {missing}")
        lines.append(
            "This is environment/setup evidence, not a changed-code failure. "
            "Rerun bootstrap or sync the requested extras."
        )
    if report["check_failures"]:
        lines.append(f"Dependency probe failures: {', '.join(report['check_failures'])}")

    if "entry_points" in report:
        _render_entry_points_report(lines, report["entry_points"])

    if status == "ready":
        lines.append("Requested optional imports are available.")
        if "entry_points" in report:
            lines.append("Declared project entry-point metadata matches installed distribution.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the dependency-only worktree preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="all-extras",
        help="Dependency import profile to check (default: all-extras).",
    )
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        metavar="IMPORT",
        help="Check one import instead of the selected profile; repeatable.",
    )
    parser.add_argument(
        "--check-entry-points",
        action="store_true",
        help="Verify declared pyproject.toml [project.entry-points] match installed package metadata.",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=None,
        help="Path to pyproject.toml to check (default: repo root pyproject.toml).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    modules = args.modules if args.modules else PROFILES[args.profile]
    profile = "custom" if args.modules else args.profile

    entry_points_report = None
    if args.check_entry_points:
        pyproject_path = args.pyproject or _default_pyproject_path()
        entry_points_report = check_declared_entry_points(pyproject_path)

    report = check_modules(modules, profile=profile, entry_points_report=entry_points_report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_human(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
