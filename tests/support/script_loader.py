"""Shared loader for validation/analysis scripts loaded by file path in tests.

Why this module exists
----------------------
Throughout the repository, focused tests load sibling scripts under
``scripts/`` directly by file path, e.g.::

    spec = importlib.util.spec_from_file_location("mymod", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

That three-line idiom is subtly broken for modules that define
``@dataclass(frozen=True)`` (or any dataclass) with a *bare* string-form
annotation such as ``InitVar`` / ``ClassVar`` imported via
``from dataclasses import InitVar`` (or ``from typing import ClassVar``).
CPython's :func:`dataclasses._is_type` resolves the bare annotation by looking
up the defining module in :data:`sys.modules`::

    ns = sys.modules.get(cls.__module__).__dict__  # dataclasses.py:749

When the module was never inserted into :data:`sys.modules` before
``exec_module`` ran, that lookup returns ``None`` and the assignment raises::

    AttributeError: 'NoneType' object has no attribute '__dict__'

This was recorded as friction in issue #5289 while loading
``scripts/validation/check_cross_host_source_staging.py`` from a focused test
under ``from __future__ import annotations``. The fix is the one
:mod:`importlib` documents for custom loaders: insert the freshly created
module into :data:`sys.modules` *before* calling ``exec_module`` so any code
executed during construction (notably ``@dataclass`` and its annotation
introspection) can find ``cls.__module__``.

This helper centralises that fixed idiom: tests and ``conftest.py`` files should
call :func:`load_script_module` instead of re-spelling the broken bootstrap.
Migration is gradual (issue #5289 filed the friction); new direct-loaders should
use this helper from the start.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types
    from pathlib import Path


def _module_name_from_path(path: Path | str, explicit_name: str | None) -> str:
    """Derive a deterministic module name for a script path.

    The name only has to be unique within :data:`sys.modules` for the loader to
    behave; callers may supply an explicit name to keep historical test module
    names stable during gradual migration. The default uses ``{parent}.{stem}``
    plus a short digest of the resolved path. That remains readable in
    tracebacks while avoiding collisions between distinct scripts that share a
    parent directory name and stem.
    """
    if explicit_name:
        return explicit_name
    import pathlib

    p = pathlib.Path(str(path)).resolve()
    path_digest = hashlib.sha256(str(p).encode()).hexdigest()[:12]
    return f"{p.parent.name}.{p.stem}_{path_digest}"


def load_script_module(
    path: Path | str,
    *,
    name: str | None = None,
    register: bool = True,
) -> types.ModuleType:
    """Load a Python script by file path into a fully usable module.

    This is the shared replacement for the bare
    ``spec_from_file_location`` / ``module_from_spec`` / ``exec_module`` idiom.
    It inserts the module into :data:`sys.modules` *before* ``exec_module`` so
    that ``@dataclass(frozen=True)`` (and any dataclass whose bare string
    annotations need :func:`dataclasses._is_type`) can resolve
    ``cls.__module__`` instead of hitting ``AttributeError`` (issue #5289).

    Args:
        path: Path to the ``.py`` script to load.
        name: Optional module name to register under. If omitted, a stable
            name is derived from ``path``. Supply an explicit name to keep a
            previously-hard-coded direct-loader name stable during migration.
        register: When ``True`` (default), insert the module into
            :data:`sys.modules` before ``exec_module`` and leave it registered,
            matching the behaviour of a real ``import``. Pass ``False`` only for
            short-lived diagnostics that must not pollute ``sys.modules``; the
            dataclass fix does *not* take effect with ``register=False``.

    Returns:
        The loaded module object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ImportError: If the spec cannot be created for the path.
    """
    import pathlib

    path_obj: pathlib.Path = pathlib.Path(str(path))
    if not path_obj.is_file():
        raise FileNotFoundError(f"script path does not exist: {path}")

    module_name = _module_name_from_path(path, name)

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for script: {path}")

    module = importlib.util.module_from_spec(spec)

    # The fix central to issue #5289: register before exec so dataclass
    # annotation introspection can find cls.__module__ in sys.modules.
    previously_registered = sys.modules.get(module_name)
    if register:
        sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except BaseException:
        # On failure, restore prior registration (typical importlib behaviour)
        # so a half-loaded module name doesn't shadow a retry. If we had
        # inserted a brand-new entry, drop it; otherwise keep the prior module.
        if register:
            if previously_registered is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previously_registered
        raise

    return module
