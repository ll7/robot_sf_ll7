"""Optional encoder import failures must leave non-video consumers usable."""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_ENCODERS = (
    ("robot_sf.benchmark.full_classic.encode", "ImageSequenceClip", None),
    ("robot_sf.benchmark.full_classic.videos", "ImageSequenceClip", None),
    ("robot_sf.render.sim_view", "MOVIEPY_AVAILABLE", False),
)


def _load_with_moviepy_error(
    module_name: str, error: Exception, monkeypatch: pytest.MonkeyPatch
) -> ModuleType:
    """Execute checkout bytes in an isolated namespace with a denied encoder import."""
    original = importlib.import_module(module_name)
    source = Path(original.__file__).resolve()
    assert source.is_relative_to(Path(__file__).resolve().parents[2] / "robot_sf")
    probe_name = f"{module_name}_import_probe"
    spec = importlib.util.spec_from_file_location(probe_name, source)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, probe_name, probe)
    original_import = builtins.__import__
    attempts = []

    def denied_import(name, *args, **kwargs):
        if name in ("moviepy", "moviepy.video.io.ImageSequenceClip"):
            attempts.append(name)
            raise error
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as imports:
        imports.setattr(builtins, "__import__", denied_import)
        try:
            spec.loader.exec_module(probe)
        finally:
            assert len(attempts) == 1, "the optional encoder import must be exercised"
    return probe


@pytest.mark.parametrize(("module_name", "availability", "unavailable"), _ENCODERS)
def test_unreadable_moviepy_configuration_disables_encoder(
    module_name: str, availability: str, unavailable: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Permission denial at each import boundary disables its optional encoder."""
    module = _load_with_moviepy_error(
        module_name, PermissionError(13, "Permission denied", ".env"), monkeypatch
    )
    assert getattr(module, availability) is unavailable


@pytest.mark.parametrize("module_name", [case[0] for case in _ENCODERS])
def test_unexpected_moviepy_import_failure_propagates(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unrelated encoder defects must remain visible instead of becoming unavailable."""
    error = RuntimeError("unexpected encoder initialization failure")
    with pytest.raises(RuntimeError, match="unexpected encoder initialization failure") as caught:
        _load_with_moviepy_error(module_name, error, monkeypatch)
    assert caught.value is error
