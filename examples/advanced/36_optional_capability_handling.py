"""Handle unavailable optional capabilities without fallback or fabricated success.

Usage:
    uv run python examples/advanced/36_optional_capability_handling.py [--format text|json]

Prerequisites:
    - None (all unavailable states use deterministic fixtures; no network access,
      no heavy optional imports, no machine-specific missing assets).

Expected Output:
    - One status line per capability probe with a stable reason code and remedy hint.
    - ``--format json`` preserves the same reason codes in machine-readable form.

Limitations:
    - Tutorial only; it demonstrates branching on status, not package installation,
      model download, dataset acquisition, or runtime launch.

References:
    - robot_sf/common/optional_import.py
    - model/registry.md
    - docs/user-guide.md#8-troubleshoot
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_MISSING_EXTRA = "robot_sf_nonexistent_extra_xyz"
_MISSING_BACKEND = "nonexistent_backend_xyz"


@dataclass(frozen=True)
class CapabilityStatus:
    """One capability probe outcome with a stable reason code and remedy hint."""

    capability: str
    available: bool
    reason_code: str
    detail: str
    remedy: str


def check_core_capability() -> CapabilityStatus:
    """Probe an available core dependency through the canonical import guard."""
    from robot_sf.common.optional_import import try_import

    module = try_import("numpy")
    if module is None:  # pragma: no cover - numpy is a required dependency
        return CapabilityStatus(
            capability="core_numpy",
            available=False,
            reason_code="core_missing",
            detail="numpy import returned None.",
            remedy="Reinstall repository dependencies: uv sync --all-extras.",
        )
    return CapabilityStatus(
        capability="core_numpy",
        available=True,
        reason_code="core_available",
        detail=f"numpy {module.__version__} imported.",
        remedy="None required.",
    )


def check_optional_extra() -> CapabilityStatus:
    """Probe a missing optional extra without importing anything absent."""
    from robot_sf.common.optional_import import try_import

    if try_import(_MISSING_EXTRA) is None:
        return CapabilityStatus(
            capability="optional_extra",
            available=False,
            reason_code="extra_missing",
            detail=f"Optional module '{_MISSING_EXTRA}' is not installed.",
            remedy="Install the documented extra for this capability, then retry.",
        )
    return CapabilityStatus(  # pragma: no cover - fixture module never exists
        capability="optional_extra",
        available=True,
        reason_code="extra_available",
        detail="Optional module is installed.",
        remedy="None required.",
    )


def check_model_artifact() -> CapabilityStatus:
    """Resolve a local-only registry entry whose artifact is absent (no download)."""
    import yaml

    from robot_sf.models.registry import resolve_model_path

    with tempfile.TemporaryDirectory(prefix="tutorial_registry_") as tmpdir:
        registry_path = Path(tmpdir) / "registry.yaml"
        registry = {
            "version": 1,
            "models": [
                {
                    "model_id": "tutorial_missing_model",
                    "local_path": str(Path(tmpdir) / "absent_model.zip"),
                    "local_only": True,
                }
            ],
        }
        registry_path.write_text(yaml.safe_dump(registry))
        try:
            resolve_model_path(
                "tutorial_missing_model", registry_path=registry_path, allow_download=False
            )
        except FileNotFoundError as exc:
            return CapabilityStatus(
                capability="model_artifact",
                available=False,
                reason_code="model_unavailable",
                detail=f"{type(exc).__name__}: registered local model artifact is unavailable.",
                remedy="Stage the checkpoint per model/registry.md, then retry.",
            )
    return CapabilityStatus(  # pragma: no cover - fixture artifact never exists
        capability="model_artifact",
        available=True,
        reason_code="model_available",
        detail="Model artifact resolved locally.",
        remedy="None required.",
    )


def check_unknown_model_id() -> CapabilityStatus:
    """Distinguish an unknown registry id from a known-but-absent artifact."""
    import yaml

    from robot_sf.models.registry import get_registry_entry

    registry = {"version": 1, "models": []}
    with tempfile.TemporaryDirectory(prefix="tutorial_registry_") as tmpdir:
        registry_path = Path(tmpdir) / "registry.yaml"
        registry_path.write_text(yaml.safe_dump(registry))
        try:
            get_registry_entry("tutorial_unknown_model", registry_path)
        except KeyError as exc:
            return CapabilityStatus(
                capability="unknown_model_id",
                available=False,
                reason_code="model_unknown",
                detail=str(exc),
                remedy="Register the model id per model/registry.md, then retry.",
            )
    return CapabilityStatus(  # pragma: no cover - empty registry never contains the id
        capability="unknown_model_id",
        available=True,
        reason_code="model_available",
        detail="Model id is registered.",
        remedy="None required.",
    )


def check_dataset_artifact() -> CapabilityStatus:
    """Probe an unavailable external dataset through the canonical SDD loader."""
    from robot_sf.data.external.sdd_trajectories import (
        SddTrajectoryDataError,
        load_sdd_track_set,
    )

    with tempfile.TemporaryDirectory(prefix="tutorial_dataset_") as tmpdir:
        missing = Path(tmpdir) / "absent_annotations.txt"
        try:
            load_sdd_track_set(
                missing, scene="tutorial", split="train", frame_rate_hz=30.0, meters_per_pixel=0.1
            )
        except (SddTrajectoryDataError, OSError) as exc:
            return CapabilityStatus(
                capability="dataset_artifact",
                available=False,
                reason_code="dataset_unavailable",
                detail=f"{type(exc).__name__}: SDD dataset annotations are unavailable.",
                remedy="Stage the licensed dataset per the data-staging docs, then retry.",
            )
    return CapabilityStatus(  # pragma: no cover - fixture path never exists
        capability="dataset_artifact",
        available=True,
        reason_code="dataset_available",
        detail="Dataset annotations parsed.",
        remedy="None required.",
    )


def check_external_runtime() -> CapabilityStatus:
    """Probe an unsupported external runtime through the simulator registry."""
    from robot_sf.sim.registry import get_backend

    try:
        get_backend(_MISSING_BACKEND)
    except KeyError as exc:
        return CapabilityStatus(
            capability="external_runtime",
            available=False,
            reason_code="runtime_unsupported",
            detail=str(exc),
            remedy="Use a registered backend or install the runtime integration.",
        )
    return CapabilityStatus(  # pragma: no cover - fixture backend never registers
        capability="external_runtime",
        available=True,
        reason_code="runtime_available",
        detail="Backend is registered.",
        remedy="None required.",
    )


def collect_statuses() -> list[CapabilityStatus]:
    """Run every capability probe in deterministic order.

    Returns:
        Probe outcomes in a fixed order for stable output.
    """
    return [
        check_core_capability(),
        check_optional_extra(),
        check_model_artifact(),
        check_unknown_model_id(),
        check_dataset_artifact(),
        check_external_runtime(),
    ]


def format_text(statuses: list[CapabilityStatus]) -> str:
    """Render probe outcomes as friendly text lines.

    Args:
        statuses: Probe outcomes from :func:`collect_statuses`.

    Returns:
        Human-readable lines with reason codes and remedy hints.
    """
    fail_on_unknown_status(statuses)
    lines = []
    for status in statuses:
        state = "available" if status.available else "UNAVAILABLE"
        lines.append(f"[{state}] {status.capability} ({status.reason_code}): {status.detail}")
        lines.append(f"  remedy: {status.remedy}")
    return "\n".join(lines)


def format_json(statuses: list[CapabilityStatus]) -> str:
    """Render probe outcomes as machine-readable JSON.

    Args:
        statuses: Probe outcomes from :func:`collect_statuses`.

    Returns:
        JSON array preserving the same reason codes as the text format.
    """
    fail_on_unknown_status(statuses)
    return json.dumps([asdict(status) for status in statuses], indent=2, sort_keys=True)


def fail_on_unknown_status(statuses: list[CapabilityStatus]) -> None:
    """Fail closed when a probe reports an unrecognized reason code.

    Args:
        statuses: Probe outcomes from :func:`collect_statuses`.

    Raises:
        ValueError: If any reason code is outside the known vocabulary.
    """
    known = {
        "core_available",
        "core_missing",
        "extra_available",
        "extra_missing",
        "model_available",
        "model_unavailable",
        "model_unknown",
        "dataset_available",
        "dataset_unavailable",
        "runtime_available",
        "runtime_unsupported",
    }
    unknown = [status.reason_code for status in statuses if status.reason_code not in known]
    if unknown:
        raise ValueError(f"Unknown capability reason codes: {sorted(set(unknown))}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the tutorial.

    Args:
        argv: Argument list for testing; defaults to process arguments.

    Returns:
        Parsed arguments with the requested output format.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run every probe, print statuses, and fail closed on unknown codes.

    Args:
        argv: Argument list for testing; defaults to process arguments.

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)
    statuses = collect_statuses()
    fail_on_unknown_status(statuses)
    print(format_json(statuses) if args.format == "json" else format_text(statuses))
    return 0


if __name__ == "__main__":
    sys.exit(main())
