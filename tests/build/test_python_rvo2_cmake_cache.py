"""Tests for the vendored python-rvo2 CMake cache self-heal helpers."""

# ruff: noqa: D103

from __future__ import annotations

import importlib.util
import stat
from pathlib import Path

_HELPER_PATH = (
    Path(__file__).resolve().parents[2] / "third_party" / "python-rvo2" / "_rvo2_cmake_build.py"
)


def _load_helper():
    spec = importlib.util.spec_from_file_location("_rvo2_cmake_build_test", _HELPER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cmake_build = _load_helper()


def _write_cache(build_dir: Path, cmake: Path) -> Path:
    build_dir.mkdir(parents=True)
    cache_path = build_dir / "CMakeCache.txt"
    cache_path.write_text(f"CMAKE_COMMAND:INTERNAL={cmake}\n", encoding="utf-8")
    return cache_path


def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_read_cached_cmake_command_parses_internal_cache_entry(tmp_path: Path) -> None:
    cache_path = _write_cache(tmp_path / "build" / "RVO2", Path("/missing/cmake"))

    assert cmake_build.read_cached_cmake_command(cache_path) == Path("/missing/cmake")


def test_cmake_cache_is_stale_when_cached_executable_is_missing(tmp_path: Path) -> None:
    build_dir = tmp_path / "build" / "RVO2"
    _write_cache(build_dir, tmp_path / "missing" / "cmake")

    assert cmake_build.cmake_cache_is_stale(build_dir) is True


def test_cmake_cache_is_not_stale_when_cached_executable_exists(tmp_path: Path) -> None:
    cmake = _make_executable(tmp_path / "cmake")
    build_dir = tmp_path / "build" / "RVO2"
    _write_cache(build_dir, cmake)

    assert cmake_build.cmake_cache_is_stale(build_dir) is False


def test_remove_stale_cmake_build_dir_deletes_stale_cache(tmp_path: Path) -> None:
    build_dir = tmp_path / "build" / "RVO2"
    _write_cache(build_dir, tmp_path / "missing" / "cmake")
    (build_dir / "generated").write_text("stale", encoding="utf-8")

    assert cmake_build.remove_stale_cmake_build_dir(build_dir) is True
    assert not build_dir.exists()


def test_remove_stale_cmake_build_dir_preserves_valid_cache(tmp_path: Path) -> None:
    cmake = _make_executable(tmp_path / "cmake")
    build_dir = tmp_path / "build" / "RVO2"
    _write_cache(build_dir, cmake)
    marker = build_dir / "generated"
    marker.write_text("keep", encoding="utf-8")

    assert cmake_build.remove_stale_cmake_build_dir(build_dir) is False
    assert marker.read_text(encoding="utf-8") == "keep"


def test_resolve_cmake_executable_honors_environment_override(tmp_path: Path, monkeypatch) -> None:
    cmake = _make_executable(tmp_path / "cmake")
    monkeypatch.setenv("ROBOT_SF_CMAKE", str(cmake))

    assert cmake_build.resolve_cmake_executable() == str(cmake)


def test_configure_and_build_removes_stale_cache_before_configure(
    tmp_path: Path, monkeypatch
) -> None:
    build_dir = tmp_path / "build" / "RVO2"
    _write_cache(build_dir, tmp_path / "missing" / "cmake")
    cmake = _make_executable(tmp_path / "cmake")
    calls: list[tuple[list[str], Path]] = []

    def fake_check_call(args, cwd):
        calls.append((args, cwd))
        if args[1] == "../..":
            (Path(cwd) / "CMakeCache.txt").write_text(
                f"CMAKE_COMMAND:INTERNAL={cmake}\n", encoding="utf-8"
            )

    monkeypatch.setenv("ROBOT_SF_CMAKE", str(cmake))
    monkeypatch.setattr(cmake_build.subprocess, "check_call", fake_check_call)

    cmake_build.configure_and_build_rvo2(build_dir)

    assert calls == [
        ([str(cmake), "../..", "-DCMAKE_CXX_FLAGS=-fPIC"], build_dir),
        ([str(cmake), "--build", "."], build_dir),
    ]


def test_configure_and_build_clean_directory_configures_then_builds(
    tmp_path: Path, monkeypatch
) -> None:
    build_dir = tmp_path / "build" / "RVO2"
    cmake = _make_executable(tmp_path / "cmake")
    calls: list[list[str]] = []

    def fake_check_call(args, cwd):
        calls.append(args)
        if args[1] == "../..":
            (Path(cwd) / "CMakeCache.txt").write_text(
                f"CMAKE_COMMAND:INTERNAL={cmake}\n", encoding="utf-8"
            )

    monkeypatch.setenv("ROBOT_SF_CMAKE", str(cmake))
    monkeypatch.setattr(cmake_build.subprocess, "check_call", fake_check_call)

    cmake_build.configure_and_build_rvo2(build_dir)

    assert calls == [
        [str(cmake), "../..", "-DCMAKE_CXX_FLAGS=-fPIC"],
        [str(cmake), "--build", "."],
    ]


def test_configure_and_build_valid_existing_cache_builds_without_reconfigure(
    tmp_path: Path, monkeypatch
) -> None:
    cmake = _make_executable(tmp_path / "cmake")
    build_dir = tmp_path / "build" / "RVO2"
    _write_cache(build_dir, cmake)
    calls: list[list[str]] = []

    monkeypatch.setenv("ROBOT_SF_CMAKE", str(cmake))
    monkeypatch.setattr(cmake_build.subprocess, "check_call", lambda args, cwd: calls.append(args))

    cmake_build.configure_and_build_rvo2(build_dir)

    assert calls == [[str(cmake), "--build", "."]]
