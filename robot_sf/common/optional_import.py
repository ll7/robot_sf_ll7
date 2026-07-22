"""Canonical optional-dependency import guard (see issue #4990).

Optional-dependency import guards in ``robot_sf`` had drifted into ~40 distinct
spellings. This module provides the single blessed spelling for the *common*
case so that reviews can focus scrutiny on the (few) guards that legitimately
catch more than ``ImportError``.

The common case
---------------
Most optional imports are of the form "import a module if it is available,
otherwise bind the name to ``None`` and degrade gracefully". Use
:func:`try_import`::

    from robot_sf.common.optional_import import try_import

    yaml = try_import("yaml")
    if yaml is None:
        ...  # degrade gracefully

:func:`try_import` catches **only** :class:`ImportError` (which includes
:class:`ModuleNotFoundError`). It never swallows broader failures, so a buggy
optional dependency cannot hide behind an "optional import".

When is a broader ``except`` legitimate?
----------------------------------------
Some optional imports intentionally catch more than ``ImportError``. Do **not**
use :func:`try_import` for these; keep the explicit broad ``except`` and add a
``# pragma: no cover`` plus a one-line justification:

* **Native/extension libraries** (OMPL, NVML/pynvml, pygame) may raise
  ``RuntimeError`` or ``OSError`` during a partially-broken install even when
  the package is importable by name.
* **Matplotlib backend selection** can raise ``RuntimeError``.
* **Compiled-backend entry points** (shapely/scipy) may raise ``AttributeError``
  when a native wheel component is absent.

The AST inventory ratchet in
``tests/test_optional_import_guard_inventory.py`` must be updated to bless any
new (justified) broad spelling; otherwise the test fails and blocks the change
in review.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def try_import(name: str) -> ModuleType | None:
    """Import ``name`` if available, otherwise return ``None``.

    This is the canonical spelling for the common optional-dependency case. It
    catches only :class:`ImportError` (which includes
    :class:`ModuleNotFoundError`) and never swallows broader failures such as
    ``RuntimeError``/``OSError``/``AttributeError`` -- those indicate a broken
    install and should surface, not be masked as "missing dependency".

    Parameters
    ----------
    name:
        Dotted module name to import, e.g. ``"yaml"`` or
        ``"shapely.geometry"``.

    Returns
    -------
    ModuleType | None
        The imported module, or ``None`` if it is not installed.

    Examples
    --------
    >>> yaml = try_import("yaml")
    >>> if yaml is None:
    ...     pass  # degrade gracefully without the YAML backend
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


# Map of optional dependency import names to the PEP 621 extra that provides
# them (Issue #5799). Keep this aligned with ``[project.optional-dependencies]``
# in pyproject.toml.
_EXTRA_FOR_DEPENDENCY: dict[str, str] = {
    # [viz]
    "pygame": "viz",
    "moviepy": "viz",
    "seaborn": "viz",
    # [maps]
    "osmnx": "maps",
    "geopandas": "maps",
    "pyproj": "maps",
    "svgelements": "maps",
    # [training]
    "stable_baselines3": "training",
    "torch": "training",
    "sklearn": "training",
    "optuna": "training",
    "tensorboard": "training",
    "wandb": "training",
    # [benchmark]
    "pandas": "benchmark",
    "duckdb": "benchmark",
    "pyarrow": "benchmark",
}


def missing_extra_error(name: str, extra: str | None = None) -> ModuleNotFoundError:
    """Build a clear ``ModuleNotFoundError`` pointing at the missing extra.

    Use this together with :func:`try_import` when a feature *requires* an
    optional dependency (i.e. it cannot degrade gracefully). The returned error
    carries an actionable ``install robot_sf[<extra>]`` hint so users see exactly
    what to install instead of a bare import failure.

    Parameters
    ----------
    name:
        The optional dependency import name (e.g. ``"pygame"``).
    extra:
        The extra that provides it. If ``None``, the known mapping in
        ``_EXTRA_FOR_DEPENDENCY`` is used, falling back to ``[all]``.

    Returns
    -------
    ModuleNotFoundError
        An error whose message tells the user which extra to install.

    Examples
    --------
    >>> pygame = try_import("pygame")
    >>> if pygame is None:
    ...     raise missing_extra_error("pygame", "viz")
    """
    resolved_extra = extra or _EXTRA_FOR_DEPENDENCY.get(name, "all")
    return ModuleNotFoundError(
        f"The optional dependency '{name}' is required for this feature but is "
        f"not installed. Install it with the '{resolved_extra}' extra, e.g.:\n"
        f'    uv pip install -e ".[{resolved_extra}]"   # editable worktree\n'
        f'    pip install "robot_sf[{resolved_extra}]"      # from a built wheel'
    )


def require_extra(name: str, extra: str | None = None) -> ModuleType:
    """Import and return an optional dependency, or raise a clear error.

    Combines :func:`try_import` with :func:`missing_extra_error`: it returns the
    module when installed, otherwise raises a ``ModuleNotFoundError`` with an
    actionable ``install robot_sf[<extra>]`` hint. Use this (rather than
    ``try_import``) for modules that genuinely need the extra and cannot degrade.

    This helper introduces no new ``except ImportError`` spelling at call sites,
    so it stays within the optional-import guard inventory ratchet
    (``tests/test_optional_import_guard_inventory.py``).

    Parameters
    ----------
    name:
        The optional dependency import name (e.g. ``"torch"``).
    extra:
        The extra that provides it. If ``None``, the known mapping in
        ``_EXTRA_FOR_DEPENDENCY`` is used, falling back to ``[all]``.

    Returns
    -------
    ModuleType
        The imported module.
    """
    module = try_import(name)
    if module is None:
        raise missing_extra_error(name, extra)
    return module
