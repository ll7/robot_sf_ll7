"""Local runtime diagnostics for Robot SF developer environments."""

from __future__ import annotations

import importlib.metadata
import importlib.util as importlib_util
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import robot_sf

if TYPE_CHECKING:
    from collections.abc import Sequence

CRITICAL_BINARIES = ("git", "uv")
OPTIONAL_BINARIES = ("ffmpeg", "gh", "docker", "jq")
OPTIONAL_IMPORTS = ("gymnasium", "pygame", "matplotlib", "numpy")
OPTIONAL_ENV_VARS = ("MPLBACKEND", "SDL_VIDEODRIVER", "DISPLAY")
DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

# Optional dependency groups that unlock larger feature slices of the project.
OPTIONAL_EXTRAS = ("training", "gpu", "orca", "socnav", "rllib", "analysis")
# Map deps are pulled in by osmnx-based OSM map authoring examples.
MAP_DEP_IMPORTS = ("osmnx", "shapely")
# Bundled model artifacts the quickstart examples rely on.
MODEL_ARTIFACTS = (
    Path("model/pedestrian/ppo_ped_01.zip"),
    Path("model/pedestrian/ppo_ped_02.zip"),
    Path("model/pedestrian/ppo_corner.zip"),
    Path("model/pedestrian/ppo_headon.zip"),
    Path("model/pedestrian/ppo_intersection.zip"),
)
# Quickstart examples the doctor can sanity-check for presence/runnability.
QUICKSTART_EXAMPLES = (
    Path("examples/quickstart/01_basic_robot.py"),
    Path("examples/quickstart/02_trained_model.py"),
    Path("examples/quickstart/03_custom_map.py"),
    Path("examples/quickstart/04_occupancy_grid.py"),
)
UV_BOOTSTRAP_HINT = (
    "Install uv (https://docs.astral.sh/uv/getting-started/) with one of:\n"
    "  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
    "  python -m pip install uv\n"
    "  brew install uv"
)


@dataclass(frozen=True)
class DoctorCheck:
    """Single doctor check result."""

    name: str
    status: str
    details: dict[str, Any]
    required: bool = False

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "name": self.name,
            "status": self.status,
            "required": self.required,
            "details": self.details,
        }


def _check_python_version() -> DoctorCheck:
    """Check the active Python version.

    Returns:
        DoctorCheck: Python version check result.
    """
    version = platform.python_version()
    ok = sys.version_info >= (3, 11)
    return DoctorCheck(
        name="python",
        status="ok" if ok else "failed",
        required=True,
        details={
            "version": version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "requires": ">=3.11",
        },
    )


def _check_package_source() -> DoctorCheck:
    """Report the installed package version and import source.

    Returns:
        DoctorCheck: Package import/source check result.
    """
    try:
        version = importlib.metadata.version("robot-sf")
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed"

    return DoctorCheck(
        name="robot_sf_package",
        status="ok",
        details={
            "version": version,
            "source": str(Path(robot_sf.__file__).resolve()),
        },
    )


def _check_binary(name: str, *, required: bool) -> DoctorCheck:
    """Check whether an executable is present on PATH.

    Returns:
        DoctorCheck: Binary availability check result.
    """
    path = shutil.which(name)
    return DoctorCheck(
        name=f"binary:{name}",
        status="ok" if path else ("failed" if required else "missing_optional"),
        required=required,
        details={"path": path},
    )


def _check_optional_import(name: str) -> DoctorCheck:
    """Check whether an optional Python import is available.

    Returns:
        DoctorCheck: Optional import availability check result.
    """
    spec = importlib_util.find_spec(name)
    return DoctorCheck(
        name=f"import:{name}",
        status="ok" if spec else "missing_optional",
        required=False,
        details={"available": spec is not None},
    )


def _check_environment_variables() -> DoctorCheck:
    """Report environment variables that commonly affect headless runs.

    Returns:
        DoctorCheck: Environment-variable report.
    """
    return DoctorCheck(
        name="environment",
        status="ok",
        details={name: os.environ.get(name) for name in OPTIONAL_ENV_VARS},
    )


def _resolve_workspace_path(path: Path, workspace_root: Path) -> Path:
    """Resolve a path relative to the workspace root when it is not absolute.

    Returns:
        Path: Absolute path for the requested workspace-relative path.
    """
    return path if path.is_absolute() else workspace_root / path


def _check_git_worktree(workspace_root: Path) -> DoctorCheck:
    """Report local machine context and git checkout state.

    Returns:
        DoctorCheck: Workspace state report.
    """
    local_machine = workspace_root / "local.machine.md"
    git_dir = workspace_root / ".git"
    return DoctorCheck(
        name="workspace",
        status="ok",
        details={
            "cwd": str(Path.cwd().resolve()),
            "workspace_root": str(workspace_root),
            "has_git_dir": git_dir.exists(),
            "local_machine_context": str(local_machine.resolve())
            if local_machine.exists()
            else None,
            "local_machine_readable": local_machine.is_file(),
        },
    )


def _check_artifact_root(root: Path) -> DoctorCheck:
    """Check whether the artifact root can accept a temporary file.

    Returns:
        DoctorCheck: Artifact-root writability check result.
    """
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".doctor-", dir=root, delete=True) as handle:
            handle.write(b"ok")
            handle.flush()
        return DoctorCheck(
            name="artifact_root",
            status="ok",
            details={"path": str(root.resolve()), "writable": True},
        )
    except OSError as exc:
        return DoctorCheck(
            name="artifact_root",
            status="failed",
            details={"path": str(root), "writable": False, "error": str(exc)},
        )


def _run_env_smoke() -> DoctorCheck:
    """Run one minimal reset/step smoke through the public environment factory.

    Returns:
        DoctorCheck: Environment reset/step smoke result.
    """
    try:
        from robot_sf.gym_env.environment_factory import make_robot_env  # noqa: PLC0415

        env = make_robot_env(debug=False, seed=0)
        try:
            env.reset(seed=0)
            action = env.action_space.sample()
            env.step(action)
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    except Exception as exc:
        return DoctorCheck(
            name="env_smoke",
            status="failed",
            details={"reset_step": False, "error": f"{type(exc).__name__}: {exc}"},
        )
    return DoctorCheck(
        name="env_smoke",
        status="ok",
        details={"reset_step": True},
    )


def _check_model_artifacts(workspace_root: Path) -> DoctorCheck:
    """Report whether bundled model artifacts the quickstart relies on are present.

    Returns:
        DoctorCheck: Model-artifact presence check result.
    """
    missing = [str(rel) for rel in MODEL_ARTIFACTS if not (workspace_root / rel).is_file()]
    status = "ok" if not missing else "missing_optional"
    return DoctorCheck(
        name="model_artifacts",
        status=status,
        required=False,
        details={
            "present": [str(rel) for rel in MODEL_ARTIFACTS if (workspace_root / rel).is_file()],
            "missing": missing,
            "hint": "Restore model artifacts from the release bundle or run without the pre-trained PPO demo."
            if missing
            else "Bundled model artifacts present.",
        },
    )


def _check_optional_extras() -> DoctorCheck:
    """Report which optional dependency groups are importable.

    Returns:
        DoctorCheck: Optional-extras availability check result.
    """
    details: dict[str, Any] = {}
    for extra in OPTIONAL_EXTRAS:
        # Each extra is dominated by a flagship import we can probe cheaply.
        probe = {
            "training": "stable_baselines3",
            "gpu": "torch",
            "orca": "rvo2",
            "socnav": "cv2",
            "rllib": "ray",
            "analysis": "seaborn",
        }[extra]
        available = importlib_util.find_spec(probe) is not None
        details[extra] = {"available": available, "probe": probe}
    map_imports = {name: importlib_util.find_spec(name) is not None for name in MAP_DEP_IMPORTS}
    details["map_deps"] = {"available": all(map_imports.values()), "imports": map_imports}
    status = "ok" if all(v["available"] for v in details.values()) else "missing_optional"
    return DoctorCheck(
        name="optional_extras",
        status=status,
        required=False,
        details={
            "extras": details,
            "hint": "Enable extras with: uv sync --extra <group>  (e.g. uv sync --extra training)",
        },
    )


def _check_quickstart(workspace_root: Path) -> DoctorCheck:
    """Report whether the quickstart examples are present and importable.

    Returns:
        DoctorCheck: Quickstart readiness check result.
    """
    missing = [str(rel) for rel in QUICKSTART_EXAMPLES if not (workspace_root / rel).is_file()]
    status = "ok" if not missing else "failed"
    return DoctorCheck(
        name="quickstart",
        status=status,
        required=True,
        details={
            "present": [
                str(rel) for rel in QUICKSTART_EXAMPLES if (workspace_root / rel).is_file()
            ],
            "missing": missing,
            "hint": "Quickstart examples are missing from the checkout."
            if missing
            else "Quickstart examples present; run with: uv run python examples/quickstart/01_basic_robot.py",
        },
    )


def _check_uv_bootstrap() -> DoctorCheck:
    """Report uv presence with a beginner-friendly bootstrap hint when absent.

    Returns:
        DoctorCheck: uv bootstrap check result.
    """
    path = shutil.which("uv")
    return DoctorCheck(
        name="uv_bootstrap",
        status="ok" if path else "failed",
        required=True,
        details={
            "path": path,
            "hint": UV_BOOTSTRAP_HINT if not path else "uv is available on PATH.",
        },
    )


def collect_doctor_report(
    *,
    artifact_root: Path = Path("output"),
    run_env_smoke: bool = True,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Collect local runtime diagnostics for issue reports and setup triage.

    Returns:
        dict[str, Any]: JSON-serializable doctor report.
    """
    resolved_workspace_root = (
        DEFAULT_WORKSPACE_ROOT if workspace_root is None else workspace_root.resolve()
    )
    resolved_artifact_root = _resolve_workspace_path(artifact_root, resolved_workspace_root)
    checks: list[DoctorCheck] = [
        _check_python_version(),
        _check_package_source(),
        _check_uv_bootstrap(),
        *[_check_binary(name, required=True) for name in CRITICAL_BINARIES],
        *[_check_binary(name, required=False) for name in OPTIONAL_BINARIES],
        *[_check_optional_import(name) for name in OPTIONAL_IMPORTS],
        _check_environment_variables(),
        _check_git_worktree(resolved_workspace_root),
        _check_artifact_root(resolved_artifact_root),
        _check_model_artifacts(resolved_workspace_root),
        _check_optional_extras(),
        _check_quickstart(resolved_workspace_root),
    ]
    if run_env_smoke:
        checks.append(_run_env_smoke())
    else:
        checks.append(
            DoctorCheck(
                name="env_smoke",
                status="skipped",
                details={"reset_step": None, "reason": "disabled by --skip-env-smoke"},
            )
        )

    overall = "ok"
    if any(check.required and check.status == "failed" for check in checks):
        overall = "failed"
    elif any(check.status in {"failed", "missing_optional"} for check in checks):
        overall = "warning"

    return {
        "schema": "robot_sf_bench.doctor.v1",
        "status": overall,
        "checks": [check.to_jsonable() for check in checks],
    }


def doctor_exit_code(report: dict[str, Any]) -> int:
    """Return the CLI exit code for a doctor report."""
    return 1 if report.get("status") == "failed" else 0


_STATUS_SYMBOL = {"ok": "PASS", "skipped": "SKIP", "missing_optional": "WARN", "failed": "FAIL"}


def _format_human(report: dict[str, Any]) -> str:
    """Render a doctor report as a categorized pass/warn/fail summary with remedies.

    Returns:
        str: Human-readable doctor report.
    """
    lines: list[str] = ["Robot SF environment check", ""]
    failures: list[str] = []
    for check in report["checks"]:
        symbol = _STATUS_SYMBOL.get(check["status"], check["status"].upper())
        lines.append(f"[{symbol}] {check['name']}")
        details = check.get("details", {}) or {}
        hint = details.get("hint")
        if hint and check["status"] in {"failed", "missing_optional"}:
            for hint_line in str(hint).splitlines():
                lines.append(f"      -> {hint_line}")
        if check["status"] == "failed":
            failures.append(check["name"])
    lines.append("")
    overall = report["status"]
    if overall == "ok":
        lines.append("All required checks passed.")
    elif overall == "warning":
        lines.append("Finished with warnings (optional capabilities missing).")
    else:
        lines.append("Hard failures detected; resolve the FAIL items above before continuing.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run doctor diagnostics as a standalone helper entrypoint.

    Returns:
        int: Process-style exit code.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    fmt = "friendly"
    rest: list[str] = []
    if "--format" in args:
        idx = args.index("--format")
        fmt = args[idx + 1]
        rest = args[:idx] + args[idx + 2 :]
    skip_smoke = "--skip-env-smoke" in rest
    rest = [a for a in rest if a != "--skip-env-smoke"]
    del argv
    report = collect_doctor_report(run_env_smoke=not skip_smoke)
    text = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
        if fmt == "json"
        else _format_human(report)
    )
    sys.stdout.write(text)
    return doctor_exit_code(report)
