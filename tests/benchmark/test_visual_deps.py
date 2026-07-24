"""Hermetic unit tests for optional-dependency detection helpers.

Tests ``has_pygame``, ``has_moviepy``, and ``ffmpeg_in_path`` from
``robot_sf.benchmark.full_classic.visual_deps`` using monkeypatching to
cover both available and unavailable branches without depending on the
actual external packages or binaries.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

_real_import_module = __import__("importlib").import_module


@pytest.fixture(autouse=True)
def _clear_visual_deps_caches():
    """Clear ``lru_cache`` on every helper so monkeypatches take effect."""
    import robot_sf.benchmark.full_classic.visual_deps as _vd

    _vd.has_pygame.cache_clear()
    _vd.has_moviepy.cache_clear()
    _vd.ffmpeg_in_path.cache_clear()
    _vd.moviepy_ready.cache_clear()
    _vd.simulation_view_ready.cache_clear()


def _pygame_available_import(name: str, *args: object, **kwargs: object) -> object:
    if name == "pygame":
        m = MagicMock()
        m.time.get_ticks = lambda: 0  # type: ignore[method-assign]
        return m
    return _real_import_module(name)


def _pygame_unavailable_import(name: str, *args: object, **kwargs: object) -> object:
    if name == "pygame":
        raise ImportError(f"No module named {name!r}")
    return _real_import_module(name)


def _moviepy_available_import(name: str, *args: object, **kwargs: object) -> object:
    if name == "moviepy":
        m = MagicMock()
        m.__version__ = "1.0.0"
        return m
    return _real_import_module(name)


def _moviepy_unavailable_import(name: str, *args: object, **kwargs: object) -> object:
    if name == "moviepy":
        raise ImportError(f"No module named {name!r}")
    return _real_import_module(name)


def _ffmpeg_which(cmd: str) -> str | None:
    return "/usr/bin/ffmpeg" if cmd == "ffmpeg" else None


def _ffmpeg_none_which(cmd: str) -> None:
    return None


# --- has_pygame ------------------------------------------------------------


def test_has_pygame_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import robot_sf.benchmark.full_classic.visual_deps as vd

    monkeypatch.setattr("importlib.import_module", _pygame_available_import)
    assert vd.has_pygame() is True


def test_has_pygame_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import robot_sf.benchmark.full_classic.visual_deps as vd

    monkeypatch.setattr("importlib.import_module", _pygame_unavailable_import)
    assert vd.has_pygame() is False


# --- has_moviepy -----------------------------------------------------------


def test_has_moviepy_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import robot_sf.benchmark.full_classic.visual_deps as vd

    monkeypatch.setattr("importlib.import_module", _moviepy_available_import)
    assert vd.has_moviepy() is True


def test_has_moviepy_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import robot_sf.benchmark.full_classic.visual_deps as vd

    monkeypatch.setattr("importlib.import_module", _moviepy_unavailable_import)
    assert vd.has_moviepy() is False


# --- ffmpeg_in_path --------------------------------------------------------


def test_ffmpeg_in_path_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import robot_sf.benchmark.full_classic.visual_deps as vd

    monkeypatch.setattr("shutil.which", _ffmpeg_which)
    assert vd.ffmpeg_in_path() is True


def test_ffmpeg_in_path_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import robot_sf.benchmark.full_classic.visual_deps as vd

    monkeypatch.setattr("shutil.which", _ffmpeg_none_which)
    assert vd.ffmpeg_in_path() is False


# --- moviepy_ready (composite helper) --------------------------------------


def test_moviepy_ready_both_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import robot_sf.benchmark.full_classic.visual_deps as vd

    monkeypatch.setattr("importlib.import_module", _moviepy_available_import)
    monkeypatch.setattr("shutil.which", _ffmpeg_which)
    assert vd.moviepy_ready() is True


def test_moviepy_ready_moviepy_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import robot_sf.benchmark.full_classic.visual_deps as vd

    monkeypatch.setattr("importlib.import_module", _moviepy_unavailable_import)
    monkeypatch.setattr("shutil.which", _ffmpeg_which)
    assert vd.moviepy_ready() is False


def test_moviepy_ready_ffmpeg_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import robot_sf.benchmark.full_classic.visual_deps as vd

    monkeypatch.setattr("importlib.import_module", _moviepy_available_import)
    monkeypatch.setattr("shutil.which", _ffmpeg_none_which)
    assert vd.moviepy_ready() is False


# --- __all__ contract ------------------------------------------------------


def test_all_names_exported() -> None:
    import robot_sf.benchmark.full_classic.visual_deps as vd

    expected = {
        "ffmpeg_in_path",
        "has_moviepy",
        "has_pygame",
        "moviepy_ready",
        "simulation_view_ready",
    }
    assert set(vd.__all__) == expected


def test_every_name_in_all_is_callable() -> None:
    import robot_sf.benchmark.full_classic.visual_deps as vd

    for name in vd.__all__:
        assert callable(getattr(vd, name)), f"{name} is not callable"
