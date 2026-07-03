"""Helpers for the vendored python-rvo2 CMake build wrapper."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_CMAKE_COMMAND_PREFIX = "CMAKE_COMMAND:INTERNAL="
_SYSTEM_CMAKE_CANDIDATES = (
    Path("/usr/bin/cmake"),
    Path("/usr/local/bin/cmake"),
    Path("/opt/homebrew/bin/cmake"),
)


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def read_cached_cmake_command(cache_path: Path) -> Path | None:
    """Return the cached CMake command path when ``CMakeCache.txt`` records one."""
    if not cache_path.exists():
        return None

    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(_CMAKE_COMMAND_PREFIX):
            command = line[len(_CMAKE_COMMAND_PREFIX) :].strip()
            return Path(command) if command else None
    return None


def cmake_cache_is_stale(build_dir: Path) -> bool:
    """Return true when the cached CMake executable is recorded but missing."""
    cached_command = read_cached_cmake_command(build_dir / "CMakeCache.txt")
    return cached_command is not None and not cached_command.exists()


def remove_stale_cmake_build_dir(build_dir: Path) -> bool:
    """Delete ``build_dir`` when its cache references a missing CMake executable.

    Returns:
        True when the build directory was deleted, otherwise False.
    """
    if not cmake_cache_is_stale(build_dir):
        return False

    shutil.rmtree(build_dir)
    return True


def resolve_cmake_executable() -> str:
    """Return a resolvable CMake executable, preferring persistent system paths."""
    for env_var in ("ROBOT_SF_CMAKE", "CMAKE_EXECUTABLE"):
        configured = os.environ.get(env_var)
        if configured:
            path = Path(configured)
            if _is_executable(path):
                return str(path)
            raise RuntimeError(f"{env_var} points to a non-executable CMake path: {configured}")

    for candidate in _SYSTEM_CMAKE_CANDIDATES:
        if _is_executable(candidate):
            return str(candidate)

    cmake = shutil.which("cmake")
    if cmake:
        return cmake

    raise RuntimeError("Could not find a CMake executable. Install cmake or set ROBOT_SF_CMAKE.")


def configure_and_build_rvo2(build_dir: Path) -> None:
    """Self-heal stale CMake state, then configure and build the vendored RVO2 library."""
    build_dir = build_dir.resolve()
    remove_stale_cmake_build_dir(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake = resolve_cmake_executable()
    if not (build_dir / "CMakeCache.txt").exists():
        subprocess.check_call([cmake, "../..", "-DCMAKE_CXX_FLAGS=-fPIC"], cwd=build_dir)
    subprocess.check_call([cmake, "--build", "."], cwd=build_dir)
