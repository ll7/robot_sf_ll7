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
import importlib.util
import json
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

EXTRA_MODULES = {
    "viz": ("pygame", "matplotlib", "PIL", "moviepy", "seaborn"),
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


def check_modules(modules: Iterable[str], *, profile: str) -> dict[str, Any]:
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
    else:
        status = "ready"
        exit_code = 0

    return {
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
    if status == "ready":
        lines.append("Requested optional imports are available.")
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
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    modules = args.modules if args.modules else PROFILES[args.profile]
    profile = "custom" if args.modules else args.profile
    report = check_modules(modules, profile=profile)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_human(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
