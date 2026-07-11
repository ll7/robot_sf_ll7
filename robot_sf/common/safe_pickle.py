"""Restricted unpickler for RobotSF pickle loading.

This module provides a restricted unpickler that only allows loading of specific
known-safe classes and NumPy reconstruction symbols. Any other pickle globals
are rejected with a clear error message.

Security note: This is a mitigation, not a proof that pickle is safe. Restricted
unpickling reduces RCE surface for known payloads; it is not a general-purpose
safe pickle parser.
"""

from __future__ import annotations

import io
import pickle
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path
from robot_sf.errors import RobotSfError


class UnsafePickleError(RobotSfError, RuntimeError):
    """Raised when a restricted unpickler rejects a pickle global."""


class RestrictedUnpickler(pickle.Unpickler):
    """Restricted unpickler that only allows specific known-safe globals."""

    def __init__(
        self,
        file: io.BytesIO,
        *,
        allowed_globals: set[tuple[str, str]],
        label: str,
    ) -> None:
        """Initialize the restricted unpickler.

        Args:
            file: BytesIO stream to unpickle from.
            allowed_globals: Set of (module, name) tuples that are allowed.
            label: Human-readable label for error messages (e.g., "playback recording").
        """
        super().__init__(file)
        self._allowed_globals = frozenset(allowed_globals)
        self._label = label

    def find_class(self, module: str, name: str) -> Any:
        """Override to restrict which globals can be loaded.

        Args:
            module: Module name of the class to load.
            name: Class/function name to load.

        Returns:
            The class or function object if allowed.

        Raises:
            UnsafePickleError: If the (module, name) pair is not in the allowlist.
        """
        key = (module, name)
        if key in self._allowed_globals:
            return super().find_class(module, name)
        raise UnsafePickleError(
            f"Unsafe pickle global rejected for {self._label}: "
            f"{module}.{name} is not in the allowlist. "
            f"Allowed globals: {sorted(self._allowed_globals)}"
        )


# Allowlist for RobotSF playback recordings (states, map_def tuples).
PLAYBACK_RECORDING_ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        # RobotSF simulation state classes
        ("robot_sf.render.sim_state", "VisualizableSimState"),
        ("robot_sf.render.sim_state", "VisualizableAction"),
        ("robot_sf.render.sim_view", "VisualizableSimState"),
        ("robot_sf.render.sim_view", "VisualizableAction"),
        # RobotSF map and navigation classes
        ("robot_sf.nav.map_config", "MapDefinition"),
        ("robot_sf.nav.obstacle", "Obstacle"),
        ("robot_sf.nav.global_route", "GlobalRoute"),
        # NumPy array and scalar reconstruction (required for array data in recordings)
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.numeric", "scalar"),
        ("numpy.core.numeric", "_reconstruct"),
        # NumPy 2.x module paths
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        # Required for protocol-2 pickles with bytes
        ("_codecs", "encode"),
    }
)

# Allowlist for SocNavBench ETH traversible payloads.
SOCNAVBENCH_TRAVERSIBLE_ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        # NumPy array and scalar reconstruction only
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.numeric", "scalar"),
        ("numpy.core.numeric", "_reconstruct"),
        # NumPy 2.x module paths
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        # Required for protocol-2 pickles with bytes
        ("_codecs", "encode"),
    }
)


def restricted_pickle_load(
    file_obj: io.BytesIO,
    *,
    allowed_globals: frozenset[tuple[str, str]],
    label: str,
) -> Any:
    """Load a pickle from a file object using a restricted unpickler.

    Args:
        file_obj: BytesIO stream to unpickle from.
        allowed_globals: Set of (module, name) tuples that are allowed.
        label: Human-readable label for error messages.

    Returns:
        The unpickled Python object.

    Raises:
        UnsafePickleError: If the pickle contains disallowed globals.
    """
    unpickler = RestrictedUnpickler(
        file_obj,
        allowed_globals=allowed_globals,
        label=label,
    )
    return unpickler.load()


def restricted_pickle_load_path(
    path: Path,
    *,
    allowed_globals: frozenset[tuple[str, str]],
    label: str,
) -> Any:
    """Load a pickle from a file path using a restricted unpickler.

    Args:
        path: Path to the pickle file.
        allowed_globals: Set of (module, name) tuples that are allowed.
        label: Human-readable label for error messages.

    Returns:
        The unpickled Python object.

    Raises:
        UnsafePickleError: If the pickle contains disallowed globals.
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path is a directory.
    """
    with path.open("rb") as f:
        return restricted_pickle_load(
            io.BytesIO(f.read()),
            allowed_globals=allowed_globals,
            label=label,
        )
