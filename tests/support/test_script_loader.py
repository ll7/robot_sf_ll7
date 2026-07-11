"""Tests for the shared script-by-path loader (issue #5289).

These are CPU-only and pin the exact friction recorded in issue #5289: a
directly-loaded validation script that declares a ``@dataclass(frozen=True)``
with a bare ``InitVar`` string annotation (under
``from __future__ import annotations``) fails during ``exec_module`` because
``dataclasses._is_type`` looks up ``cls.__module__`` in ``sys.modules`` and the
bare direct-loader idiom never inserts the module there.

The shared helper in :mod:`tests.support.script_loader` registers the module
before ``exec_module`` so the annotation introspection succeeds; these tests
prove both halves (the failure mode and the fix).
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest

from tests.support.script_loader import load_script_module

if TYPE_CHECKING:
    from pathlib import Path

# A script that reproduces the issue #5289 failure surface: a frozen dataclass
# with a bare ``InitVar`` string annotation (the ``from dataclasses import
# InitVar`` form whose resolution hits ``dataclasses._is_type`` line 749).
_FROZEN_DATACLASS_SCRIPT = textwrap.dedent(
    """\
    from __future__ import annotations

    from dataclasses import dataclass, field, InitVar


    @dataclass(frozen=True)
    class ImmutableState:
        name: str
        init_config: InitVar[dict]
        regulary: int = 0

        def __post_init__(self, init_config: dict) -> None:
            self.__dict__["init_config"] = init_config
    """,
)


def _write_script(tmp_path: Path, body: str, filename: str = "frozen_script.py") -> Path:
    path = tmp_path / filename
    path.write_text(body)
    return path


def test_bare_direct_loader_breaks_frozen_dataclass(tmp_path: Path) -> None:
    """The bare direct-loader idiom must still reproduce the issue #5289 failure.

    This is the red side of the red/green pairing: it asserts that the
    historical three-line idiom (no ``sys.modules`` registration before
    ``exec_module``) raises ``AttributeError`` from
    ``dataclasses._is_type``. If this ever stops failing, the helper's fix is
    leaking into the reference path and the test must be re-tightened.
    """
    path = _write_script(tmp_path, _FROZEN_DATACLASS_SCRIPT)
    name = "issue_5289_bare_loader_red"

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    # Ensure a previously-loaded same-name module doesn't mask the failure.
    sys.modules.pop(name, None)

    with pytest.raises(AttributeError, match="__dict__"):
        spec.loader.exec_module(module)

    sys.modules.pop(name, None)


def test_load_script_module_supports_frozen_dataclass(tmp_path: Path) -> None:
    """The shared helper loads the issue #5289 script without error.

    This is the green side: :func:`load_script_module` registers the module in
    ``sys.modules`` before ``exec_module`` so ``dataclasses._is_type`` resolves
    ``cls.__module__`` and the frozen dataclass is constructed normally.
    """
    path = _write_script(tmp_path, _FROZEN_DATACLASS_SCRIPT)
    name = "issue_5289_helper_green"

    try:
        module = load_script_module(path, name=name)
    finally:
        sys.modules.pop(name, None)

    instance = module.ImmutableState(name="alpha", init_config={"k": 1})
    assert instance.name == "alpha"
    assert instance.regulary == 0
    assert instance.init_config == {"k": 1}
    # Frozen dataclass must actually be immutable (FrozenInstanceError subclasses
    # AttributeError, so a single AttributeError match pins the contract).
    with pytest.raises(AttributeError, match=r"cannot assign to field 'name'"):
        instance.name = "beta"  # type: ignore[misc]


def test_load_script_module_registers_module_in_sys_modules(tmp_path: Path) -> None:
    """The helper registers the loaded module under the requested name.

    Callers rely on the module being importable afterward (e.g. fixtures that
    cache a builder module across the session). The fix from issue #5289
    centrally depends on registration happening *before* ``exec_module``; this
    independently pins that the registration persists.
    """
    path = _write_script(tmp_path, "VALUE = 7\n", filename="plain_script.py")
    name = "issue_5289_register_check"

    try:
        module = load_script_module(path, name=name)
        assert sys.modules.get(name) is module
        assert module.VALUE == 7
        # Subsequent lookup by name returns the same object (no duplicate loads).
        assert sys.modules[name] is module
    finally:
        sys.modules.pop(name, None)


def test_load_script_module_default_name_is_stable(tmp_path: Path) -> None:
    """When no ``name`` is given, the derived name is deterministic and stable.

    Gradual migration means some callers will keep passing explicit names and
    others will adopt the default; the default name must not flake between runs
    for the same path so module identity (and ``sys.modules`` keys) stay stable.
    The exact string is an implementation detail (``{parent}.{stem}`` of the
    resolved path) and may change; this test pins the stable-for-a-given-path
    contract plus readability, not a fixed name.
    """
    from tests.support.script_loader import _module_name_from_path

    path = _write_script(tmp_path, "X = 1\n", filename="stable_named.py")

    module = load_script_module(path)
    try:
        # Same path -> same derived name across independent calls.
        assert module.__name__ == _module_name_from_path(path, None)
        assert sys.modules.get(module.__name__) is module
        # The stem must appear in the derived name (readable in tracebacks).
        assert "stable_named" in module.__name__
    finally:
        sys.modules.pop(module.__name__, None)


def test_load_script_module_default_name_distinguishes_same_named_scripts(tmp_path: Path) -> None:
    """Distinct scripts with the same parent name and stem cannot collide."""
    left = tmp_path / "left" / "scripts" / "shared.py"
    right = tmp_path / "right" / "scripts" / "shared.py"
    left.parent.mkdir(parents=True)
    right.parent.mkdir(parents=True)
    left.write_text("VALUE = 'left'\n")
    right.write_text("VALUE = 'right'\n")

    left_module = load_script_module(left)
    right_module = load_script_module(right)
    try:
        assert left_module.__name__ != right_module.__name__
        assert sys.modules[left_module.__name__] is left_module
        assert sys.modules[right_module.__name__] is right_module
        assert left_module.VALUE == "left"
        assert right_module.VALUE == "right"
    finally:
        sys.modules.pop(left_module.__name__, None)
        sys.modules.pop(right_module.__name__, None)


def test_load_script_module_missing_path_raises(tmp_path: Path) -> None:
    """A missing script path fails fast with a clear FileNotFoundError."""
    missing = tmp_path / "does_not_exist.py"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_script_module(missing)
