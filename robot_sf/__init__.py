"""Robot SF package bootstrap with lazily resolved telemetry exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

try:
    # Written at build time by hatch-vcs (see [tool.hatch.build.hooks.vcs] in
    # pyproject.toml). Present in built wheels and editable installs.
    from ._version import __version__
except ImportError:  # pragma: no cover - source checkout without a build
    # Fall back to installed package metadata (importlib.metadata is stdlib on
    # the supported Python >=3.11, so its import needs no guard).
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        __version__ = _pkg_version("robot_sf")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

if TYPE_CHECKING:  # pragma: no cover - static type information only
    from . import telemetry
    from .telemetry import ManifestWriter, RunRegistry, RunTrackerConfig, generate_run_id

__all__ = [
    "ManifestWriter",
    "RunRegistry",
    "RunTrackerConfig",
    "generate_run_id",
    "telemetry",
]

# __version__ (assigned above) is intentionally a plain module attribute, not an
# ``__all__`` export, so the telemetry export contract stays unchanged.
_TELEMETRY_EXPORTS = frozenset(__all__) - {"telemetry"}


def __getattr__(name: str) -> Any:
    """Resolve telemetry exports only when a caller requests them.

    Keeping package import light lets standalone tools such as the coverage
    comparator avoid importing optional visualization and TensorBoard backends.

    Returns:
        The requested telemetry module or export.
    """
    if name == "telemetry":
        value = import_module(".telemetry", __name__)
    elif name in _TELEMETRY_EXPORTS:
        value = getattr(import_module(".telemetry", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported telemetry names in interactive discovery.

    Returns:
        Available package attribute names.
    """
    return sorted(set(globals()) | set(__all__))
